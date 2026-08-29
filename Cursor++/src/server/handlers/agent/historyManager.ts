import type { AgentServerMessage } from '../../gen/agent_v1_pb'
import type { LLMContentBlock, LLMMessage } from '../llm/types'
import { logger } from '../../logger'
import { decodeBlob, encodeBlob } from './blob'
import { cacheBlob, getCachedBlob } from './blobStore'
import { normalizeBlobMessage, restoreBlobMessageToLLMMessage } from './transcript'
import { createRepairDiagnostics, hasRepairMutations, repairConversationHistory, type RepairDiagnostics } from '../llm/transformMessages'

export interface HistoryEntry {
  blobId: string
  raw: Record<string, unknown>
  message: LLMMessage
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return !!value && typeof value === 'object'
}

export function* sendAndCacheBlob(
  buildKvMessage: (id: number, blobId: string, blobData: string) => AgentServerMessage,
  id: number,
  data: { role: string, content: unknown, toolCallId?: string, toolName?: string, isError?: boolean, providerOptions?: Record<string, unknown> },
  blobIds: string[],
): Generator<AgentServerMessage, void, void> {
  const normalized = normalizeBlobMessage(data)
  const blob = encodeBlob(normalized)
  blobIds.push(blob.blobId)
  cacheBlob(blob.blobId, blob.blobData)
  yield buildKvMessage(id, blob.blobId, blob.blobData)
}

export function* flushMessageBlobs(
  buildKvMessage: (id: number, blobId: string, blobData: string) => AgentServerMessage,
  messages: LLMMessage[],
  startIndex: number,
  blobCounter: number,
  blobIds: string[],
): Generator<AgentServerMessage, { nextIndex: number, blobCounter: number }, void> {
  let nextIndex = startIndex
  let nextBlobCounter = blobCounter

  for (let i = startIndex; i < messages.length; i++) {
    const msg = messages[i]
    yield* sendAndCacheBlob(buildKvMessage, ++nextBlobCounter, {
      role: msg.role,
      content: msg.content,
      toolCallId: msg.toolCallId,
      toolName: msg.toolName,
      isError: msg.isError,
      providerOptions: msg.providerOptions,
    }, blobIds)
    nextIndex = i + 1
  }

  return { nextIndex, blobCounter: nextBlobCounter }
}

export function extractPlainTextContent(message: LLMMessage): string {
  if (typeof message.content === 'string')
    return message.content
  return message.content
    .filter((block): block is Extract<LLMContentBlock, { type: 'text' | 'thinking' }> => block.type === 'text' || block.type === 'thinking')
    .map(block => block.text)
    .join('')
}

export function extractComparableUserTexts(message: LLMMessage): string[] {
  const text = extractPlainTextContent(message).trim()
  if (!text)
    return []

  const values = new Set<string>([text])
  // eslint-disable-next-line regexp/no-super-linear-backtracking
  const userQueryMatches = text.matchAll(/<user_query>\s*([\s\S]*?)\s*<\/user_query>/g)
  for (const match of userQueryMatches) {
    const inner = match[1]?.trim()
    if (inner)
      values.add(inner)
  }

  return [...values]
}

export function hasSystemMessage(messages: LLMMessage[]): boolean {
  return messages.some(message => message.role === 'system')
}

export function isPreambleUserMessage(message: LLMMessage): boolean {
  if (message.role !== 'user' || typeof message.content !== 'string')
    return false
  return message.content.includes('<user_info>')
    && !message.content.includes('<user_query>')
}

export function hasPreambleUserMessage(messages: LLMMessage[]): boolean {
  return messages.some(isPreambleUserMessage)
}

function syncConversationScaffold(messages: LLMMessage[], systemMessage: LLMMessage, preambleUserMessage: LLMMessage): { messages: LLMMessage[], systemReplaced: boolean, preambleReplaced: boolean } {
  const next = [...messages]
  let systemReplaced = false
  let preambleReplaced = false

  const systemIndex = next.findIndex(message => message.role === 'system')
  if (systemIndex >= 0 && next[systemIndex]?.content !== systemMessage.content) {
    next[systemIndex] = systemMessage
    systemReplaced = true
  }

  const preambleIndex = next.findIndex(isPreambleUserMessage)
  if (preambleIndex >= 0 && next[preambleIndex]?.content !== preambleUserMessage.content) {
    next[preambleIndex] = preambleUserMessage
    preambleReplaced = true
  }

  return { messages: next, systemReplaced, preambleReplaced }
}

export function mergePrependUserMessages(
  messages: LLMMessage[],
  prependUserMessages: Array<{ text: string, messageId?: string }>,
): { messages: LLMMessage[], insertedTexts: string[] } {
  if (prependUserMessages.length === 0)
    return { messages, insertedTexts: [] }

  const existingUserTexts = new Set(
    messages
      .filter(message => message.role === 'user')
      .flatMap(extractComparableUserTexts)
      .filter(text => text.length > 0),
  )

  const missing = prependUserMessages
    .map(entry => entry.text)
    .filter(text => text.length > 0 && !existingUserTexts.has(text))

  if (missing.length === 0) {
    logger.info({ prependCount: prependUserMessages.length }, '[SESSION] prepend user messages already satisfied by history')
    return { messages, insertedTexts: [] }
  }

  const insertAt = messages.findIndex(message => message.role !== 'system' && !isPreambleUserMessage(message))
  const prefix = missing.map(text => ({ role: 'user' as const, content: text }))

  logger.info({
    prependCount: prependUserMessages.length,
    missingCount: missing.length,
    firstMissing: missing[0],
  }, '[SESSION] merging prepend user messages into history')

  if (insertAt === -1)
    return { messages: [...messages, ...prefix], insertedTexts: missing }
  return {
    messages: [...messages.slice(0, insertAt), ...prefix, ...messages.slice(insertAt)],
    insertedTexts: missing,
  }
}

/**
 * 摘要 blob 判定 — 双保险 (设计文档 §6 Q6):
 *   1. providerOptions.cursor.isSummary 语义标记 (修复后透传, 语义根治);
 *   2. 内容前缀 fallback: assistant 且以 `Previous conversation summary:` 开头
 *      (本插件格式) 或官方 `[Previous conversation summary]: ` 格式 ——
 *      对修复上线前的存量摘要 blob 立即生效 (标记已丢, 只剩前缀)。
 */
const SUMMARY_CONTENT_PREFIXES = [
  'Previous conversation summary:',
  '[Previous conversation summary]:',
] as const

function extractLeadingTextFromContent(content: unknown): string {
  if (typeof content === 'string')
    return content
  if (Array.isArray(content)) {
    const firstTextBlock = content.find(
      (block): block is Record<string, unknown> => isRecord(block) && block.type === 'text',
    )
    return typeof firstTextBlock?.text === 'string' ? firstTextBlock.text : ''
  }
  return ''
}

export function isSummaryBlobMessage(raw: Record<string, unknown>): boolean {
  const providerOptions = raw.providerOptions
  if (isRecord(providerOptions)) {
    const cursor = providerOptions.cursor
    if (isRecord(cursor) && cursor.isSummary === true)
      return true
  }
  if (raw.role === 'assistant') {
    const text = extractLeadingTextFromContent(raw.content).trimStart()
    return SUMMARY_CONTENT_PREFIXES.some(prefix => text.startsWith(prefix))
  }
  return false
}

export function hydrateHistoryEntries(blobIds: string[]): HistoryEntry[] {
  const entries: HistoryEntry[] = []
  for (const blobId of blobIds) {
    const blobData = getCachedBlob(blobId)
    if (!blobData)
      continue
    try {
      const decoded = decodeBlob(blobData)
      if (!isRecord(decoded))
        continue
      const restored = restoreBlobMessageToLLMMessage(decoded)
      if (!restored)
        continue
      entries.push({ blobId, raw: decoded, message: restored })
    }
    catch (error) {
      logger.warn({ blobId, error: (error as Error).message }, '[SESSION] failed to hydrate history entry')
    }
  }
  return entries
}

export function materializeHistoryEntries(messages: LLMMessage[]): HistoryEntry[] {
  return messages.map((message) => {
    const normalized = normalizeBlobMessage({
      role: message.role,
      content: message.content,
      toolCallId: message.toolCallId,
      toolName: message.toolName,
      isError: message.isError,
      providerOptions: message.providerOptions,
    })
    const blob = encodeBlob(normalized)
    cacheBlob(blob.blobId, blob.blobData)
    return {
      blobId: blob.blobId,
      raw: normalized as unknown as Record<string, unknown>,
      message,
    }
  })
}

function logHistoryRepair(stage: string, diagnostics: RepairDiagnostics, extra: Record<string, unknown> = {}): void {
  if (!hasRepairMutations(diagnostics))
    return
  logger.debug({
    stage,
    ...extra,
    ...diagnostics,
  }, '[HISTORY_REPAIR] canonicalized conversation history')
}

export function repairHistoryEntries(entries: HistoryEntry[]): HistoryEntry[] {
  const diagnostics = createRepairDiagnostics(entries.length)
  const repaired = repairConversationHistory(entries.map(entry => entry.message), diagnostics)
  logHistoryRepair('repairHistoryEntries', diagnostics, {
    entryCount: entries.length,
    inputBlobIds: entries.length,
  })
  return materializeHistoryEntries(repaired)
}

export function* rebuildConversationHistory(params: {
  historyBlobIds: string[]
  prependUserMessages: Array<{ text: string, messageId?: string }>
  systemMessage: LLMMessage
  preambleUserMessage: LLMMessage
  currentUserMessage: LLMMessage
  systemContent: string
  preambleUserContent: string
  sendSystemScaffoldBlob: (data: { role: string, content: unknown, toolCallId?: string, toolName?: string, isError?: boolean }) => Generator<AgentServerMessage, void, void>
  sendOrderedBlob: (data: { role: string, content: unknown, toolCallId?: string, toolName?: string, isError?: boolean }) => Generator<AgentServerMessage, void, void>
}): Generator<AgentServerMessage, { messages: LLMMessage[], insertedPrependUserTexts: string[] }, void> {
  let messages: LLMMessage[] = []
  let insertedPrependUserTexts: string[] = []

  if (params.historyBlobIds.length > 0) {
    const historyEntries = hydrateHistoryEntries(params.historyBlobIds)
    logger.info({
      requestedBlobs: params.historyBlobIds.length,
      resolvedBlobs: historyEntries.length,
      prependUserMessages: params.prependUserMessages.length,
    }, '[SESSION] history blobs from cache')

    messages = historyEntries.map(entry => entry.message)

    const scaffoldSynced = syncConversationScaffold(messages, params.systemMessage, params.preambleUserMessage)
    messages = scaffoldSynced.messages
    if (scaffoldSynced.systemReplaced || scaffoldSynced.preambleReplaced) {
      logger.debug({
        systemReplaced: scaffoldSynced.systemReplaced,
        preambleReplaced: scaffoldSynced.preambleReplaced,
      }, '[HISTORY_REPAIR] replaced provider-specific scaffold in restored history')
    }

    if (!hasSystemMessage(messages)) {
      messages.unshift(params.systemMessage)
      yield* params.sendSystemScaffoldBlob({ role: 'system', content: params.systemContent })
    }

    if (!hasPreambleUserMessage(messages)) {
      const insertAt = messages.findIndex(message => message.role !== 'system')
      if (insertAt === -1)
        messages.push(params.preambleUserMessage)
      else messages.splice(insertAt, 0, params.preambleUserMessage)
      yield* params.sendOrderedBlob({ role: 'user', content: params.preambleUserContent })
    }

    ({ messages, insertedTexts: insertedPrependUserTexts } = mergePrependUserMessages(messages, params.prependUserMessages))
    messages.push(params.currentUserMessage)
  }
  else {
    messages.push(params.systemMessage)
    yield* params.sendSystemScaffoldBlob({ role: 'system', content: params.systemContent })

    messages.push(params.preambleUserMessage)
    yield* params.sendOrderedBlob({ role: 'user', content: params.preambleUserContent });

    ({ messages, insertedTexts: insertedPrependUserTexts } = mergePrependUserMessages(messages, params.prependUserMessages))
    messages.push(params.currentUserMessage)
  }

  const diagnostics = createRepairDiagnostics(messages.length)
  const repaired = repairConversationHistory(messages, diagnostics)
  logHistoryRepair('rebuildConversationHistory', diagnostics, {
    historyBlobIds: params.historyBlobIds.length,
    insertedPrependUserTexts: insertedPrependUserTexts.length,
  })

  return {
    messages: repaired,
    insertedPrependUserTexts,
  }
}

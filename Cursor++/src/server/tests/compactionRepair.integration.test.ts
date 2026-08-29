import type { HistoryEntry } from '../handlers/agent/historyManager'
import type { LLMContentBlock, LLMMessage } from '../handlers/llm/types'
import { expect, it } from 'vitest'
import { planCompaction } from '../handlers/agent/compactionStrategy'
import { repairHistoryEntries } from '../handlers/agent/historyManager'
import { countTokens } from '../handlers/agent/tokenCounter'
import { repairConversationHistory } from '../handlers/llm/transformMessages'

function makeEntry(index: number, message: LLMMessage): HistoryEntry {
  return {
    blobId: `blob-${index}`,
    raw: { role: message.role, content: message.content as unknown },
    message,
  }
}

function hasToolUse(message: LLMMessage): boolean {
  return message.role === 'assistant'
    && typeof message.content !== 'string'
    && message.content.some((block: LLMContentBlock) => block.type === 'tool_use')
}

/** 多样化填充文本 (避免同字符长游程触发 tokenizer 估计路径) */
function variedFiller(chars: number): string {
  const words = ['alpha', 'beta', 'gamma', 'delta', 'epsilon']
  let text = ''
  let wordIndex = 0
  while (text.length < chars)
    text += `${words[wordIndex++ % words.length]} `
  return text
}

/**
 * 预算制 fixture: 大体积历史 + 小预算 (budgetOverride) 强制切分只保留尾部若干组。
 * assistant(tool_use) + 其 tool_result 组成一个原子组 —— 切点永不落在组内。
 */
function buildLegacyCompactionEntries(): HistoryEntry[] {
  return [
    makeEntry(0, { role: 'system', content: 'sys prompt' }),
    makeEntry(1, { role: 'user', content: '<user_info>env</user_info>' }),
    makeEntry(2, { role: 'user', content: `user-1 ${variedFiller(48_000)}` }),
    makeEntry(3, { role: 'assistant', content: `assistant-1 ${variedFiller(48_000)}` }),
    makeEntry(4, { role: 'user', content: 'user-2' }),
    makeEntry(5, {
      role: 'assistant',
      content: [
        { type: 'text', text: '先查一下。' },
        { type: 'tool_use', id: 'call_A', name: 'Read', input: { path: 'a.ts' } },
      ],
    }),
    makeEntry(6, {
      role: 'user',
      content: [
        { type: 'tool_result', toolUseId: 'call_A', toolName: 'Read', content: 'read result' },
      ],
    }),
    makeEntry(7, { role: 'assistant', content: 'assistant-tail' }),
  ]
}

function toEntries(messages: LLMMessage[]): HistoryEntry[] {
  return messages.map((message, index) => makeEntry(index, message))
}

function collectToolUseIds(entries: HistoryEntry[]): Set<string> {
  const ids = new Set<string>()
  for (const entry of entries) {
    if (!Array.isArray(entry.message.content))
      continue
    for (const block of entry.message.content) {
      if (block.type === 'tool_use')
        ids.add(block.id)
    }
  }
  return ids
}

function collectToolResultIds(entries: HistoryEntry[]): Set<string> {
  const ids = new Set<string>()
  for (const entry of entries) {
    if (entry.message.role === 'tool') {
      if (entry.message.toolCallId)
        ids.add(entry.message.toolCallId)
      continue
    }
    if (!Array.isArray(entry.message.content))
      continue
    for (const block of entry.message.content) {
      if (block.type === 'tool_result')
        ids.add(block.toolUseId)
    }
  }
  return ids
}

/** 两侧均无孤立 tool_use / tool_result (跨切点拆散) */
function expectNoSplitToolPairs(plan: ReturnType<typeof planCompaction>): void {
  const tailUseIds = collectToolUseIds(plan.keepTail)
  const tailResultIds = collectToolResultIds(plan.keepTail)
  const summarizeUseIds = collectToolUseIds(plan.summarizeEntries)
  const summarizeResultIds = collectToolResultIds(plan.summarizeEntries)
  for (const resultId of tailResultIds)
    expect(summarizeUseIds.has(resultId), `tool result ${resultId} split from its tool_use`).toBe(false)
  for (const useId of summarizeUseIds)
    expect(tailResultIds.has(useId) && !summarizeResultIds.has(useId), `tool_use ${useId} split from its result`).toBe(false)
  void tailUseIds
}

it('legacy anthropic 形态 (未 repair): 组划分把 user(tool_result) 并入 assistant(tool_use) 组, 切点不拆散配对', () => {
  const entries = buildLegacyCompactionEntries()
  // 小预算 (scanBudget ≈ 5.2K): 大条目 (≈10K tok/条) 单条即超 → 切点落在 user-2 组之前
  const plan = planCompaction(entries, { contextTokenLimit: 258_400, budgetOverride: 6_000 })

  expect(plan.mode).toBe('budget')
  expect(plan.leading.map(entry => entry.message.role)).toEqual(['system', 'user'])
  // user-1 / assistant-1 两条大消息进摘要侧; 摘要侧不含任何 tool_use
  expect(plan.summarizeEntries.map(entry => entry.message.role)).toEqual(['user', 'assistant'])
  // keepTail = [user-2, assistant(tool_use), user(tool_result 载体), assistant-tail] — 配对完整且相邻
  expect(plan.keepTail.map(entry => entry.message.role)).toEqual(['user', 'assistant', 'user', 'assistant'])
  expect(hasToolUse(plan.keepTail[1]!.message)).toBe(true)
  expect(Array.isArray(plan.keepTail[2]?.message.content)).toBe(true)
  expect(((plan.keepTail[2]?.message.content as LLMContentBlock[])[0] as Extract<LLMContentBlock, { type: 'tool_result' }>).type).toBe('tool_result')
  expectNoSplitToolPairs(plan)
})

it('repair 规范化后 (canonical tool role): 组原子性同样保证切点不拆散 assistant/tool 配对', () => {
  const repairedMessages = repairConversationHistory(buildLegacyCompactionEntries().map(entry => entry.message))
  const repairedEntries = toEntries(repairedMessages)
  const plan = planCompaction(repairedEntries, { contextTokenLimit: 258_400, budgetOverride: 6_000 })

  expect(plan.leading.map(entry => entry.message.role)).toEqual(['system', 'user'])
  expect(plan.summarizeEntries.map(entry => entry.message.role)).toEqual(['user', 'assistant'])
  expect(plan.summarizeEntries.some(entry => hasToolUse(entry.message))).toBe(false)
  // canonical 形态: tool_result 载体为 role='tool', 与其 tool_use 相邻同组
  expect(plan.keepTail.map(entry => entry.message.role)).toEqual(['user', 'assistant', 'tool', 'assistant'])
  expect(hasToolUse(plan.keepTail[1]!.message)).toBe(true)
  expect(plan.keepTail[2]?.message.toolCallId).toBe('call_A')
  expectNoSplitToolPairs(plan)
})

it('runtime helper repairHistoryEntries materializes canonicalized entries before compaction planning', () => {
  const plan = planCompaction(
    repairHistoryEntries(buildLegacyCompactionEntries()),
    { contextTokenLimit: 258_400, budgetOverride: 6_000 },
  )

  expect(plan.keepTail.map(entry => entry.message.role)).toEqual(['user', 'assistant', 'tool', 'assistant'])
  expect(hasToolUse(plan.keepTail[1]!.message)).toBe(true)
  expect(plan.keepTail[2]?.message.role).toBe('tool')
  expect(plan.keepTail[2]?.message.toolCallId).toBe('call_A')
  expectNoSplitToolPairs(plan)
})

it('providerOptions 端到端存续: 摘要 blob 经 hydrate → repair → plan 链路后标记与识别均存活', () => {
  // 构造带 isSummary 标记的摘要 blob, 缓存后经完整链路往返
  const summaryMessage: LLMMessage = {
    role: 'assistant',
    content: 'Previous conversation summary:\n- user asked X',
    providerOptions: { cursor: { isSummary: true } },
  }
  const repaired = repairHistoryEntries([makeEntry(0, summaryMessage)])
  expect(repaired.length).toBe(1)
  expect(repaired[0].message.providerOptions).toEqual({ cursor: { isSummary: true } })
  expect((repaired[0].raw as { providerOptions?: { cursor?: { isSummary?: boolean } } }).providerOptions?.cursor?.isSummary).toBe(true)
  // 计数器与 token 化 sanity
  expect(countTokens('Previous conversation summary:')).toBeGreaterThan(0)
})

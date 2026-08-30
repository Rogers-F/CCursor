import type { CompactionPlan } from '../handlers/agent/compactionStrategy'
/**
 * compactionBudget.test.ts — 第二阶段自动压缩修复 (keepTail 预算化) 测试
 *
 * 用例编号对应设计文档 §8 测试计划表 (#1-#34), 表为权威清单。
 * 分阶段落地: 阶段 1 (入口截断/image) → 阶段 5 (触发公式/重试) 逐批补齐。
 */
import type { HistoryEntry } from '../handlers/agent/historyManager'
import type { LLMContentBlock, LLMMessage } from '../handlers/llm/types'
import { readFileSync, rmSync } from 'node:fs'
import { fromBinary } from '@bufbuild/protobuf'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'
import { getSpillDir } from '../config/paths'
import { resetAgentDatabaseForTests } from '../database/sqlite'
import { ConversationSummaryArchiveSchema } from '../gen/agent_v1_pb'
import { encodeBlob } from '../handlers/agent/blob'
import { cacheBlob, resetBlobCacheForTests } from '../handlers/agent/blobStore'
import { getCompactionContentionCount, releaseCompactionLock, tryAcquireCompactionLock, waitForCompactionLockRelease } from '../handlers/agent/compactionLock'
import {
  buildSummarySource,

  computeSummaryHardCapTokens,
  computeSummaryMaxOutputTokens,
  computeSummarySourceBudgetChars,
  createCompactionArtifacts,
  estimateMessagesTokens,
  formatMessageForSummary,
  generateSummaryWithFallback,
  measureMessagesTokens,
  planCompaction,
  resolveSummaryThinkingLevel,
} from '../handlers/agent/compactionStrategy'
import { CONTEXT_LENGTH_RETRY_MAX } from '../handlers/agent/constants'
import { hydrateHistoryEntries, isSummaryBlobMessage, repairHistoryEntries } from '../handlers/agent/historyManager'
import { countTokens } from '../handlers/agent/tokenCounter'
import {
  buildTaskToolResultText,
  resolveTaskEntryCapTokens,
  TASK_ENTRY_CAP_MAX_TOKENS,
} from '../handlers/agent/toolkit/results/taskToolResults'
import { getAutoCompactThreshold, isContextLengthLimitError } from '../handlers/agent/usage'

// ─── helpers ───

export function makeBlobEntry(
  role: LLMMessage['role'],
  content: string | LLMContentBlock[],
  extra?: Record<string, unknown>,
): HistoryEntry {
  const raw: Record<string, unknown> = { role, content, ...extra }
  const blob = encodeBlob(raw)
  cacheBlob(blob.blobId, blob.blobData)
  const message: LLMMessage = { role, content }
  if (typeof extra?.toolCallId === 'string')
    message.toolCallId = extra.toolCallId
  if (typeof extra?.toolName === 'string')
    message.toolName = extra.toolName
  if (extra?.isError === true)
    message.isError = true
  if (isRecordValue(extra?.providerOptions))
    message.providerOptions = extra.providerOptions as Record<string, unknown>
  return {
    blobId: blob.blobId,
    raw,
    message,
  }
}

function isRecordValue(value: unknown): value is Record<string, unknown> {
  return !!value && typeof value === 'object' && !Array.isArray(value)
}

/** 生成指定 o200k token 量级的 ASCII 文本 */
export function makeTokenSizedText(tokens: number): string {
  // ASCII 下约 4 chars/token; 多造 5% 再精确裁剪
  let text = 'a'.repeat(Math.ceil(tokens * 4.2))
  while (countTokens(text) > tokens) {
    text = text.slice(0, Math.max(0, text.length - Math.ceil(tokens / 16) - 1))
  }
  return text
}

/** 生成指定字符量级的多样化自然文本 (模拟真实 Task 报告, 无同字符长游程) */
export function makeVariedReportText(chars: number): string {
  const lines: string[] = []
  let total = 0
  let index = 0
  while (total < chars) {
    const line = `Step ${index}: subagent examined module-${index} (src/module-${index % 97}.ts) and recorded findings, edge cases, plus follow-up questions about the implementation details.`
    lines.push(line)
    total += line.length + 1
    index++
  }
  return lines.join('\n')
}

/** 解码 archive blob 的 summarizedMessages 为 blobId 列表 (双保险归档排除断言用) */
function decodeArchivedBlobIds(artifacts: ReturnType<typeof createCompactionArtifacts>): string[] {
  return artifacts.archiveBlobs.flatMap((archiveBlob) => {
    const archiveMessage = fromBinary(ConversationSummaryArchiveSchema, Buffer.from(archiveBlob.blobData, 'base64'))
    return archiveMessage.summarizedMessages.map(idBytes => Buffer.from(idBytes).toString('utf8'))
  })
}

/** 生成指定 o200k token 量级的中文文本 (chars/4 低估 4x 场景; 比例收敛校准) */
export function makeTokenSizedChineseText(tokens: number): string {
  let text = '汉字内容片段。'.repeat(Math.ceil(tokens / 2))
  let guard = 0
  while (guard < 200) {
    const current = countTokens(text)
    if (current <= tokens)
      break
    text = text.slice(0, Math.max(24, Math.floor(text.length * tokens / current)))
    guard += 1
  }
  return text
}

// ─── setup / teardown ───

let tmpDbPath = ''
const spillConversationIdsToClean = new Set<string>()

beforeEach(async () => {
  tmpDbPath = `/tmp/.tmp-compaction-budget-${Date.now()}-${Math.random().toString(36).slice(2)}.db`
  process.env.BYOK_AGENT_DB_PATH = tmpDbPath
  await resetAgentDatabaseForTests()
  resetBlobCacheForTests()
})

afterEach(async () => {
  await resetAgentDatabaseForTests()
  resetBlobCacheForTests()
  delete process.env.BYOK_AGENT_DB_PATH
  for (const suffix of ['', '-wal', '-shm']) {
    try {
      rmSync(`${tmpDbPath}${suffix}`)
    }
    catch {}
  }
  // 清理测试创建的真实 spill 目录 (只删本测试文件拥有的会话子目录)
  for (const conversationId of spillConversationIdsToClean) {
    try {
      rmSync(`${getSpillDir()}/${conversationId}`, { recursive: true, force: true })
    }
    catch {}
  }
  spillConversationIdsToClean.clear()
})

function trackSpillConversation(conversationId: string): string {
  spillConversationIdsToClean.add(conversationId)
  return conversationId
}

// ─── #14 / #34: Task 入口截断 ───

describe('#14/#34 Task 报告入口截断 + spill', () => {
  it('#14 150K chars 报告被截断: 标注含原始 token 数与 spill 路径, spill 文件内容为全文', () => {
    const reportText = makeVariedReportText(150_000)
    expect(countTokens(reportText)).toBeGreaterThan(TASK_ENTRY_CAP_MAX_TOKENS)
    const value = {
      conversationSteps: [
        { message: { case: 'assistantMessage', value: { text: reportText } } },
      ],
      transcriptPath: '/tmp/subagent-transcript.jsonl',
    }

    const text = buildTaskToolResultText('success', value, {
      conversationId: trackSpillConversation('conv-14'),
      contextTokenLimit: 258_400,
      toolCallId: 'call_toolu_14',
    })

    expect(text).toBeTruthy()
    // 258K 窗 → ENTRY_CAP = 25K tok
    expect(countTokens(text!)).toBeLessThan(TASK_ENTRY_CAP_MAX_TOKENS + 500)
    expect(text).toMatch(/original \d+ tokens exceeded cap 25000 tokens/)
    // spill 路径出现在截断标注中
    const spillPathMatch = text!.match(/full report saved to ([^\]]+)\]/)
    expect(spillPathMatch).toBeTruthy()
    const spillPath = spillPathMatch![1]
    expect(spillPath.startsWith(`${getSpillDir()}/conv-14/`)).toBe(true)
    // spill 文件内容 = 完整报告全文 (含 transcript 行)
    const spilled = readFileSync(spillPath, 'utf8')
    expect(spilled).toContain(reportText)
    expect(spilled).toContain('/tmp/subagent-transcript.jsonl')
    // 截断标注含 transcriptPath 找回提示
    expect(text).toContain('read the subagent transcript at /tmp/subagent-transcript.jsonl')
  })

  it('#14 未超 cap 的报告原样保留', () => {
    const reportText = 'short report'
    const value = {
      conversationSteps: [
        { message: { case: 'assistantMessage', value: { text: reportText } } },
      ],
    }
    const text = buildTaskToolResultText('success', value, {
      conversationId: 'conv-14b',
      contextTokenLimit: 258_400,
      toolCallId: 'call_14b',
    })
    expect(text).toBe(reportText)
  })

  it('#34 32K 窗 + 1e5 CJK chars: cap 缩放到 8K tok, spill 保全文 (窗口缩放)', () => {
    const cjkReport = '任务报告内容片段。'.repeat(24_000)
    const value = {
      conversationSteps: [
        { message: { case: 'assistantMessage', value: { text: cjkReport } } },
      ],
    }
    expect(countTokens(cjkReport)).toBeGreaterThan(50_000)

    const text = buildTaskToolResultText('success', value, {
      conversationId: trackSpillConversation('conv-34'),
      contextTokenLimit: 32_000,
      toolCallId: 'call_toolu_34',
    })

    // 32K 窗 → ENTRY_CAP = min(25K, 8K) = 8K tok
    expect(resolveTaskEntryCapTokens(32_000)).toBe(8_000)
    expect(countTokens(text!)).toBeLessThan(9_000)
    expect(text).toMatch(/exceeded cap 8000 tokens/)
    const spillPathMatch = text!.match(/full report saved to ([^\]]+)\]/)
    expect(spillPathMatch).toBeTruthy()
    const spilled = readFileSync(spillPathMatch![1], 'utf8')
    expect(spilled).toContain(cjkReport)
  })

  it('无 entryContext 时按固定 25K cap 处理', () => {
    const reportText = 'y'.repeat(200_000)
    const value = {
      conversationSteps: [
        { message: { case: 'assistantMessage', value: { text: reportText } } },
      ],
    }
    const text = buildTaskToolResultText('success', value)
    expect(countTokens(text!)).toBeLessThan(TASK_ENTRY_CAP_MAX_TOKENS + 500)
    expect(text).toMatch(/exceeded cap 25000 tokens/)
    // 无 conversationId → 无 spill 路径, 降级为无路径版
    expect(text).not.toMatch(/full report saved to/)
    expect(text).toContain('full text could not be saved')
  })

  it('spill 写失败不阻塞主流程 (含 NUL 的非法路径 → 降级标注)', () => {
    const reportText = 'z'.repeat(150_000)
    const value = {
      conversationSteps: [
        { message: { case: 'assistantMessage', value: { text: reportText } } },
      ],
    }
    const text = buildTaskToolResultText('success', value, {
      conversationId: 'conv\u0000fail',
      contextTokenLimit: 258_400,
      toolCallId: 'call_fail',
    })
    expect(countTokens(text!)).toBeLessThan(TASK_ENTRY_CAP_MAX_TOKENS + 500)
    expect(text).toContain('full text could not be saved')
    expect(text).not.toMatch(/full report saved to/)
  })
})

// ─── image case (阶段 1) ───

describe('formatMessageForSummary image case', () => {
  it('image block 转为 [Image] 占位 (不再静默丢弃)', () => {
    const message: LLMMessage = {
      role: 'user',
      content: [
        { type: 'text', text: 'look at this' },
        { type: 'image', mimeType: 'image/png', data: 'aGVsbG8=' },
      ],
    }
    const rendered = formatMessageForSummary(message)
    expect(rendered).toContain('[Image]')
    expect(rendered).toContain('look at this')
  })
})

// ─── #7 / #19 / #30(部分): 摘要标记双保险 (阶段 2) ───

describe('#7/#19/#30 摘要标记双保险', () => {
  it('#19 前缀检测双格式: 两种前缀均被识别为摘要消息', () => {
    expect(isSummaryBlobMessage({
      role: 'assistant',
      content: 'Previous conversation summary:\n- stuff',
    })).toBe(true)
    expect(isSummaryBlobMessage({
      role: 'assistant',
      content: '[Previous conversation summary]: older official format',
    })).toBe(true)
    // 普通 assistant 消息不以这些前缀开头 → 不误伤
    expect(isSummaryBlobMessage({
      role: 'assistant',
      content: 'Here is the fix for your bug.',
    })).toBe(false)
    // user 角色带前缀也不识别 (双条件: role + 精确前缀)
    expect(isSummaryBlobMessage({
      role: 'user',
      content: 'Previous conversation summary:\nfake',
    })).toBe(false)
    // text block 形态的 assistant 消息也能识别
    expect(isSummaryBlobMessage({
      role: 'assistant',
      content: [{ type: 'text', text: 'Previous conversation summary:\n- blocks form' }],
    })).toBe(true)
  })

  it('#7 摘要 blob 经 repairHistoryEntries 重编码后标记与前缀识别均存活', () => {
    const summaryEntry = makeBlobEntry(
      'assistant',
      'Previous conversation summary:\n- user asked X\n- assistant did Y',
      { providerOptions: { cursor: { isSummary: true } } },
    )
    const followedByToolUse = makeBlobEntry('assistant', [
      { type: 'text', text: '继续处理。' },
      { type: 'tool_use', id: 'call_adjacent', name: 'Read', input: { path: 'a.ts' } },
    ])
    const toolResult = makeBlobEntry('tool', 'read result', {
      toolCallId: 'call_adjacent',
      toolName: 'Read',
    })

    const repaired = repairHistoryEntries([summaryEntry, followedByToolUse, toolResult])

    // Σ 与相邻 assistant(tool_use) 各自完整存活, 不被合并
    expect(repaired.length).toBe(3)
    expect(repaired[0].message.role).toBe('assistant')
    expect((repaired[0].message.content as string).startsWith('Previous conversation summary:')).toBe(true)
    // 语义标记透传存活 (repair → materialize 链路)
    expect(isSummaryBlobMessage(repaired[0].raw)).toBe(true)
    expect(repaired[0].message.providerOptions).toEqual({ cursor: { isSummary: true } })
    // 前缀 fallback 亦识别 (标记丢失的存量 blob 路径)
    expect(isSummaryBlobMessage({ role: 'assistant', content: repaired[0].message.content as string })).toBe(true)
    // 邻接的 assistant(tool_use) 完整存活且未被并入 Σ
    expect(repaired[1].message.role).toBe('assistant')
    expect(
      (repaired[1].message.content as LLMContentBlock[]).some(block => block.type === 'tool_use'),
    ).toBe(true)
    expect(repaired[2].message.role).toBe('tool')
    expect(repaired[2].message.toolCallId).toBe('call_adjacent')
  })

  it('#7 createCompactionArtifacts 的 archive 过滤对旧摘要仍生效 (不再归档)', () => {
    const oldSummaryEntry = makeBlobEntry(
      'assistant',
      'Previous conversation summary:\n- older round',
      { providerOptions: { cursor: { isSummary: true } } },
    )
    const userEntry = makeBlobEntry('user', 'normal user message')
    const plan = planCompaction([userEntry, makeBlobEntry('user', 'q'), makeBlobEntry('assistant', 'a')])
    plan.leading = []
    plan.summarizeEntries = [oldSummaryEntry, userEntry]
    plan.keepTail = []

    const artifacts = createCompactionArtifacts({
      plan,
      summaryText: '- new summary',
      previousSummaryArchiveIds: [],
    })

    const archiveBlob = artifacts.archiveBlobs[0]!
    const archiveMessage = fromBinary(
      ConversationSummaryArchiveSchema,
      Buffer.from(archiveBlob.blobData, 'base64'),
    )
    const decodedBlobIds = archiveMessage.summarizedMessages.map(ids => Buffer.from(ids).toString('utf8'))
    // 旧摘要 blob 不进 archive 名单, 普通消息进
    expect(decodedBlobIds).toContain(userEntry.blobId)
    expect(decodedBlobIds).not.toContain(oldSummaryEntry.blobId)
  })
})

// ─── 阶段 3: 预算化核心 (§4 算法) ───

/** 生成指定 o200k token 量级的多样化文本 (无同字符长游程; 比例构造 + 指数收敛校准) */
export function makeVariedTokenText(tokens: number): string {
  const words = ['alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta', 'eta', 'theta', 'iota', 'kappa', 'lambda', 'mu']
  // 实测校准: 每词 (含空格) ≈ 1.25 token
  const wordCount = Math.max(4, Math.ceil((tokens * 1.03) / 1.25))
  const parts: string[] = []
  for (let index = 0; index < wordCount; index++)
    parts.push(words[index % words.length])
  let text = parts.join(' ')
  // 指数收敛: 每次裁 2% 直到达标 (~20 次编码内收敛)
  let guard = 0
  while (countTokens(text) > tokens && text.length > 24 && guard < 200) {
    text = text.slice(0, Math.floor(text.length * 0.98))
    guard += 1
  }
  return text
}

function makeLeadingEntries(): HistoryEntry[] {
  return [
    makeBlobEntry('system', 'You are a helpful assistant.'),
    makeBlobEntry('user', '<user_info>\nUser context here\n</user_info>'),
  ]
}

interface ToolGroupSpec {
  callId: string
  toolName: string
  input: Record<string, unknown>
  resultTokens: number
  resultText?: string
}

/** 构造 [assistant(tool_use) + tool(result)] 原子组 */
function makeToolGroup(spec: ToolGroupSpec): HistoryEntry[] {
  const resultText = spec.resultText ?? makeVariedTokenText(spec.resultTokens)
  return [
    makeBlobEntry('assistant', [
      { type: 'text', text: `calling ${spec.toolName}` },
      { type: 'tool_use', id: spec.callId, name: spec.toolName, input: spec.input },
    ]),
    makeBlobEntry('tool', resultText, { toolCallId: spec.callId, toolName: spec.toolName }),
  ]
}

/** 前置普通历史 (确保摘要侧非空 — 占位把尾窗压小后整体可能装入预算导致 no-op) */
function pushBulkHistory(entries: HistoryEntry[], count: number, tokensPerEntry: number): void {
  for (let index = 0; index < count; index++)
    entries.push(makeBlobEntry(index % 2 === 0 ? 'user' : 'assistant', makeVariedTokenText(tokensPerEntry)))
}

function collectToolUseIdsFromEntries(entries: HistoryEntry[]): Set<string> {
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

function collectToolResultIdsFromEntries(entries: HistoryEntry[]): Set<string> {
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

function expectNoSplitToolPairsInPlan(plan: CompactionPlan): void {
  const tailResultIds = collectToolResultIdsFromEntries(plan.keepTail)
  const summarizeUseIds = collectToolUseIdsFromEntries(plan.summarizeEntries)
  const summarizeResultIds = collectToolResultIdsFromEntries(plan.summarizeEntries)
  for (const resultId of tailResultIds)
    expect(summarizeUseIds.has(resultId), `result ${resultId} split from its use`).toBe(false)
  for (const useId of summarizeUseIds)
    expect(tailResultIds.has(useId) && !summarizeResultIds.has(useId), `use ${useId} split from its result`).toBe(false)
}

describe('#1 生产回归: 巨物尾窗序列', () => {
  it('#1 40K+40K+20K+10K+2K+1K 尾窗: 压缩后消息侧 ≤ 80K, 三条巨物被占位, 配对完整', () => {
    const entries = [
      ...makeLeadingEntries(),
      makeBlobEntry('user', makeVariedTokenText(1_000)),
    ]
    // 前置历史: 10 条 5K 消息 (模拟 70+ 轮日志中的中段)
    for (let index = 0; index < 10; index++)
      entries.push(makeBlobEntry(index % 2 === 0 ? 'user' : 'assistant', makeVariedTokenText(5_000)))
    // 生产尾窗: 40K/40K/20K/10K/2K/1K
    const tailSizes = [40_000, 40_000, 20_000, 10_000, 2_000, 1_000]
    tailSizes.forEach((resultTokens, index) => {
      entries.push(...makeToolGroup({
        callId: `call_prod_${index}`,
        toolName: 'Read',
        input: { path: `/repo/src/file-${index}.ts` },
        resultTokens,
      }))
    })

    const plan = planCompaction(entries, { contextTokenLimit: 258_400 })
    expect(plan.mode).toBe('budget')
    expect(plan.summarizeEntries.length).toBeGreaterThan(0)

    // 三条巨物 (40K/40K/20K > 巨物线 ~11K) 被占位, 10K/2K/1K 保原文
    expect(plan.diagnostics.placeholderCount).toBe(3)
    // keepTail 实占受预算约束 (leading 极小 → budget ≈ 59.5K ≤ 60K 上限)
    expect(plan.diagnostics.keepTailActualTokens).toBeLessThanOrEqual(plan.diagnostics.budgetTokens + 2_000)
    // 压缩后消息侧 = leading + Σ(按预留估计) + keepTail ≤ 80K
    const messageSideEstimate = plan.diagnostics.leadingTokens + plan.diagnostics.summaryReserveTokens + plan.diagnostics.keepTailActualTokens
    expect(messageSideEstimate).toBeLessThanOrEqual(80_000)
    // 占位符保留骨架: role/toolCallId/toolName
    const placeholderEntries = plan.keepTail.filter(entry =>
      typeof entry.message.content === 'string' && entry.message.content.includes('[tool output elided'))
    expect(placeholderEntries.length).toBe(3)
    for (const placeholderEntry of placeholderEntries) {
      expect(placeholderEntry.message.role).toBe('tool')
      expect(placeholderEntry.message.toolCallId).toMatch(/^call_prod_/)
      expect(placeholderEntry.message.toolName).toBe('Read')
    }
    // 配对完整
    expectNoSplitToolPairsInPlan(plan)
  })
})

describe('#2/#26 安全点与孤儿断言', () => {
  it('#2 工具链在头/中/尾 × OpenAI/Anthropic 双形态: 两侧均无孤立配对', () => {
    const openAiForm: HistoryEntry[] = [
      ...makeToolGroup({ callId: 'call_head', toolName: 'Read', input: { path: 'h.ts' }, resultTokens: 50 }),
      makeBlobEntry('user', makeVariedTokenText(6_000)),
      makeBlobEntry('assistant', makeVariedTokenText(6_000)),
      ...makeToolGroup({ callId: 'call_mid', toolName: 'Shell', input: { command: 'ls' }, resultTokens: 6_000 }),
      makeBlobEntry('user', makeVariedTokenText(6_000)),
      ...makeToolGroup({ callId: 'call_tail', toolName: 'Read', input: { path: 't.ts' }, resultTokens: 50 }),
    ]
    // Anthropic 形态: tool_result 为 user 消息 content block
    const anthropicForm: HistoryEntry[] = [
      makeBlobEntry('assistant', [{ type: 'tool_use', id: 'call_a1', name: 'Read', input: { path: 'a.ts' } }]),
      makeBlobEntry('user', [{ type: 'tool_result', toolUseId: 'call_a1', toolName: 'Read', content: makeVariedTokenText(6_000) }]),
      makeBlobEntry('user', makeVariedTokenText(6_000)),
      ...makeToolGroup({ callId: 'call_a2', toolName: 'Read', input: { path: 'b.ts' }, resultTokens: 6_000 }),
    ]

    for (const form of [openAiForm, anthropicForm]) {
      const plan = planCompaction([...makeLeadingEntries(), ...form], { contextTokenLimit: 258_400, budgetOverride: 8_000 })
      expectNoSplitToolPairsInPlan(plan)
      // 锚点回溯会复制一条 user 指令进尾窗 (双保险), 总数 = body + (anchorInserted ? 1 : 0)
      expect(plan.summarizeEntries.length + plan.keepTail.length).toBe(form.length + (plan.diagnostics.anchorInserted ? 1 : 0))
    }
  })

  it('#26 孤儿断言: repair 漏网的错序工具链被切分后断言捕获并回退安全边界 (不崩溃)', () => {
    // assistant(tool_use) 与其 result 之间插入 user 文本 → 分组后 result 成为独立组,
    // 预算扫描可能把切点落在 use 与 result 之间 → 断言回退直到配对同侧
    const entries = [
      ...makeLeadingEntries(),
      makeBlobEntry('user', makeVariedTokenText(500)),
      makeBlobEntry('assistant', [
        { type: 'text', text: '调用工具' },
        { type: 'tool_use', id: 'call_orphan', name: 'Read', input: { path: 'x.ts' } },
      ]),
      makeBlobEntry('user', makeVariedTokenText(200)), // 插入文本拆开 use 与 result
      makeBlobEntry('tool', makeVariedTokenText(60), { toolCallId: 'call_orphan', toolName: 'Read' }),
      makeBlobEntry('assistant', makeVariedTokenText(4_000)),
    ]
    const plan = planCompaction(entries, { contextTokenLimit: 258_400, budgetOverride: 4_000 })
    expectNoSplitToolPairsInPlan(plan)
  })
})

describe('#3/#4 预算边界与最小保留兜底', () => {
  it('#3 恰好等于/超一条/全超: chosenCut 单调且确定性', () => {
    const entries = [
      ...makeLeadingEntries(),
      makeBlobEntry('user', makeVariedTokenText(3_000)),
      makeBlobEntry('assistant', makeVariedTokenText(3_000)),
      makeBlobEntry('user', makeVariedTokenText(3_000)),
      makeBlobEntry('assistant', makeVariedTokenText(3_000)),
    ]
    // 确定性: 同输入两次规划完全一致
    const planOnce = planCompaction(entries, { contextTokenLimit: 258_400, budgetOverride: 8_000 })
    const planTwice = planCompaction(entries, { contextTokenLimit: 258_400, budgetOverride: 8_000 })
    expect(planOnce.summarizeEntries.length).toBe(planTwice.summarizeEntries.length)
    expect(planOnce.keepTail.map(e => e.blobId)).toEqual(planTwice.keepTail.map(e => e.blobId))
    // 单调: 预算减半 → keepTail 条数不增
    const planTighter = planCompaction(entries, { contextTokenLimit: 258_400, budgetOverride: 4_000 })
    expect(planTighter.keepTail.length).toBeLessThanOrEqual(planOnce.keepTail.length)
    // 全超: 每条都超预算 → 兜底保住最后一组
    const planAll = planCompaction(entries, { contextTokenLimit: 258_400, budgetOverride: 1_000 })
    expect(planAll.keepTail.length).toBeGreaterThanOrEqual(1)
    expectNoSplitToolPairsInPlan(planAll)
  })

  it('#4 最近一轮含 100K 巨物 (前沿): 强制保留该轮, 接受超支', () => {
    const entries = [
      ...makeLeadingEntries(),
      makeBlobEntry('user', makeVariedTokenText(1_000)),
      ...makeToolGroup({ callId: 'call_huge', toolName: 'Read', input: { path: 'huge.ts' }, resultTokens: 100_000 }),
    ]
    const plan = planCompaction(entries, { contextTokenLimit: 258_400, budgetOverride: 5_000 })
    // 前沿 (最后一条 assistant 及其后) 永不占位 → 该轮完整保留 (锚点副本 + 配对组)
    expect(plan.keepTail.length).toBeGreaterThanOrEqual(2)
    const toolEntry = plan.keepTail[plan.keepTail.length - 1]!
    expect(toolEntry.message.role).toBe('tool')
    expect(toolEntry.message.toolCallId).toBe('call_huge')
    expect(plan.diagnostics.placeholderCount).toBe(0)
    expect(plan.diagnostics.frontierExcessTokens).toBeGreaterThan(0)
  })
})

describe('#5/#6 图片豁免与占位符内容', () => {
  it('#5 含 image block 的 tool_result: 不占位不截断, 按 1.6K/块计价', () => {
    const imageEntry = makeBlobEntry('user', [
      { type: 'tool_result', toolUseId: 'call_img', toolName: 'Read', content: 'screenshot attached' },
      { type: 'image', mimeType: 'image/png', data: 'aGVsbG8=' },
    ])
    const entries = [
      ...makeLeadingEntries(),
      makeBlobEntry('user', makeVariedTokenText(1_000)),
      makeBlobEntry('assistant', [{ type: 'tool_use', id: 'call_img', name: 'Read', input: { path: 'shot.png' } }]),
      imageEntry,
      // 真实 user 消息紧跟其后: 尾窗自带锚点, 避免锚点回溯重扫把图片组挤出尾窗
      makeBlobEntry('user', 'please analyze this screenshot'),
      ...makeToolGroup({ callId: 'call_after', toolName: 'Read', input: { path: 'z.ts' }, resultTokens: 50 }),
      makeBlobEntry('assistant', 'done'),
    ]
    const plan = planCompaction(entries, { contextTokenLimit: 258_400, budgetOverride: 3_000 })
    expect(plan.diagnostics.placeholderCount).toBe(0)
    const keptImageEntry = plan.keepTail.find(entry => entry === imageEntry || entry.blobId === imageEntry.blobId)
    expect(keptImageEntry).toBeTruthy()
    expect(Array.isArray(keptImageEntry!.message.content)).toBe(true)
    // 计价含 1600/块
    const imageEntryMeasured = measureMessagesTokens([imageEntry.message])
    expect(imageEntryMeasured).toBeGreaterThanOrEqual(1_600)
  })

  it('#6 占位符内容: Read/Task/Shell 三类均含 ~N tokens / locator / blobId / 恢复指引 / 头尾预览', () => {
    const entries = [
      ...makeLeadingEntries(),
      makeBlobEntry('user', makeVariedTokenText(500)),
    ]
    // 前置历史确保摘要侧非空 (占位后的尾窗很小, 不加会整体装入预算 → no-op)
    pushBulkHistory(entries, 6, 8_000)
    entries.push(...makeToolGroup({ callId: 'call_r', toolName: 'Read', input: { path: '/repo/src/big.ts' }, resultTokens: 30_000 }))
    entries.push(...makeToolGroup({
      callId: 'call_t',
      toolName: 'Task',
      input: { prompt: 'explore repo' },
      resultTokens: 30_000,
      resultText: `Subagent report\n[Subagent transcript: /tmp/t.jsonl]\n${makeVariedTokenText(29_000)}`,
    }))
    entries.push(...makeToolGroup({
      callId: 'call_s',
      toolName: 'Shell',
      input: { command: 'pnpm build' },
      resultTokens: 30_000,
      resultText: `[output written to /tmp/build-overflow.log]\n${makeVariedTokenText(29_000)}`,
    }))
    entries.push(makeBlobEntry('assistant', 'analysis done')) // 使三组全部脱离前沿

    const plan = planCompaction(entries, { contextTokenLimit: 258_400, budgetOverride: 4_000 })
    expect(plan.summarizeEntries.length).toBeGreaterThan(0)
    expect(plan.diagnostics.placeholderCount).toBe(3)

    const placeholderTexts = plan.keepTail
      .filter(entry => typeof entry.message.content === 'string' && entry.message.content.includes('[tool output elided'))
      .map(entry => entry.message.content as string)
    expect(placeholderTexts.length).toBe(3)
    for (const text of placeholderTexts) {
      expect(text).toMatch(/~\d+ tokens\]/)
      expect(text).toContain('[full content archived in blob ')
      expect(text).toContain('[to recover: re-run the tool, or ask the user]')
      expect(text).toContain('--- preview (head + tail) ---')
      expect(text).toContain('…[middle elided]…')
    }
    const readPlaceholder = placeholderTexts.find(text => text.includes('/repo/src/big.ts'))
    expect(readPlaceholder).toBeTruthy()
    expect(readPlaceholder).toMatch(/totalLines=\d+/)
    const taskPlaceholder = placeholderTexts.find(text => text.includes('Task agentId'))
    expect(taskPlaceholder).toBeTruthy()
    expect(taskPlaceholder).toContain('/tmp/t.jsonl')
    const shellPlaceholder = placeholderTexts.find(text => text.includes('Shell command'))
    expect(shellPlaceholder).toBeTruthy()
    expect(shellPlaceholder).toContain('pnpm build')
    // 预览 token 封顶: 占位符整体 < 700 tok (头 175 + 尾 75 + 骨架 + locator)
    for (const text of placeholderTexts)
      expect(countTokens(text)).toBeLessThan(700)
  })
})

describe('#8/#9 多轮不退化与两路一致', () => {
  it('#8 连续 3 次压缩: 地板不单调上升, archive 不含重复条目', () => {
    let entries: HistoryEntry[] = [
      ...makeLeadingEntries(),
      makeBlobEntry('user', makeVariedTokenText(800)),
    ]
    for (let index = 0; index < 30; index++)
      entries.push(...makeToolGroup({ callId: `call_r8_${index}`, toolName: 'Read', input: { path: `f${index}.ts` }, resultTokens: 5_000 }))

    const floorHistory: number[] = []
    const archivedBlobIdsAcrossRounds = new Set<string>()
    for (let round = 0; round < 3; round++) {
      const plan = planCompaction(entries, { contextTokenLimit: 258_400 })
      expect(plan.summarizeEntries.length).toBeGreaterThan(0)
      const artifacts = createCompactionArtifacts({
        plan,
        summaryText: `- round ${round} summary of the work done`,
        previousSummaryArchiveIds: [],
      })
      floorHistory.push(plan.diagnostics.leadingTokens + plan.diagnostics.summaryReserveTokens + plan.diagnostics.keepTailActualTokens)

      if (artifacts.archiveBlobs.length > 0) {
        const archiveMessage = fromBinary(ConversationSummaryArchiveSchema, Buffer.from(artifacts.archiveBlobs[0]!.blobData, 'base64'))
        const archivedIds = archiveMessage.summarizedMessages.map(ids => Buffer.from(ids).toString('utf8'))
        for (const archivedId of archivedIds) {
          // archive 不重复收录同一条目 (旧 Σ 不再进档 → 不滚雪球)
          expect(archivedBlobIdsAcrossRounds.has(archivedId), `archive duplicate: ${archivedId}`).toBe(false)
          archivedBlobIdsAcrossRounds.add(archivedId)
        }
        // Σ blob 自身绝不入档
        expect(archivedIds).not.toContain(artifacts.summaryBlobId)
      }
      // 下一轮从压缩后的 root 重新 hydrate (Σ 带标记), 并追加新一轮工具流
      const nextEntries = hydrateHistoryEntries(artifacts.nextRootBlobIds)
      expect(nextEntries.length).toBe(plan.leading.length + 1 + plan.keepTail.length)
      for (let index = 0; index < 8; index++)
        nextEntries.push(...makeToolGroup({ callId: `call_r8_${round}_${index}`, toolName: 'Read', input: { path: `g${index}.ts` }, resultTokens: 5_000 }))
      entries = repairHistoryEntries(nextEntries)
    }
    // 地板允许小幅波动但不得单调上升 (轮 3 ≤ 轮 1 + 摘要余量)
    expect(floorHistory[2]).toBeLessThanOrEqual(floorHistory[0] + 2_000)
  })

  it('#9 两路一致: 同一 entries 相同 options 两次规划完全相同', () => {
    const entries = [
      ...makeLeadingEntries(),
      makeBlobEntry('user', makeVariedTokenText(600)),
      ...makeToolGroup({ callId: 'call_c9', toolName: 'Read', input: { path: 'c9.ts' }, resultTokens: 20_000 }),
      makeBlobEntry('assistant', 'summary of findings'),
    ]
    const inlinePlan = planCompaction(entries, { contextTokenLimit: 258_400 })
    const actionPlan = planCompaction(entries, { contextTokenLimit: 258_400 })
    expect(inlinePlan.mode).toBe(actionPlan.mode)
    expect(inlinePlan.summarizeEntries.map(e => e.blobId)).toEqual(actionPlan.summarizeEntries.map(e => e.blobId))
    expect(inlinePlan.keepTail.map(e => e.blobId)).toEqual(actionPlan.keepTail.map(e => e.blobId))
    expect(inlinePlan.elidedOriginals).toEqual(actionPlan.elidedOriginals)
  })
})

describe('#10/#24 小窗: clamp / B 模式 / 禁用', () => {
  it('#10/#24-1 32K 窗 + 15K leading: 走 B 模式 (全量摘要 + 锚点单条)', () => {
    const bigLeading = [
      makeBlobEntry('system', makeVariedTokenText(14_000)),
      makeBlobEntry('user', `<user_info>\n${makeVariedTokenText(1_000)}\n</user_info>`),
    ]
    const entries = [
      ...bigLeading,
      makeBlobEntry('user', 'fix the build error'),
      ...makeToolGroup({ callId: 'call_s32', toolName: 'Read', input: { path: 's.ts' }, resultTokens: 2_000 }),
    ]
    const plan = planCompaction(entries, { contextTokenLimit: 32_000 })
    expect(plan.mode).toBe('b-mode')
    // 双保险 (§3.5, 验收修正 F1): 锚点既留在摘要源 (摘要器对齐任务) 又原文保留于尾窗
    expect(plan.summarizeEntries.map(entry => entry.message.role)).toEqual(['user', 'assistant', 'tool'])
    expect(plan.keepTail.length).toBe(1)
    expect(plan.keepTail[0]!.message.role).toBe('user')
    expect(plan.keepTail[0]!.message.content).toContain('fix the build error')
    expect(plan.anchorBlobId).toBe(plan.keepTail[0]!.blobId)
    expect(plan.summarizeEntries.some(entry => entry.blobId === plan.anchorBlobId)).toBe(true)
    // archive 不因双保险重复归档锚点 (createCompactionArtifacts 按 anchorBlobId 排除)
    const artifacts = createCompactionArtifacts({ plan, summaryText: 'b-mode summary', previousSummaryArchiveIds: [] })
    const archivedBlobIds = decodeArchivedBlobIds(artifacts)
    expect(archivedBlobIds).not.toContain(plan.anchorBlobId)
    expect(artifacts.nextRootBlobIds).toContain(plan.anchorBlobId)
  })

  it('#24-2 leading 过大 (~28K on 32K 窗): 禁用自动压缩 (拒动为合格终态)', () => {
    const hugeLeading = [
      makeBlobEntry('system', makeVariedTokenText(27_500)),
      makeBlobEntry('user', `<user_info>\n${makeVariedTokenText(1_000)}\n</user_info>`),
    ]
    const entries = [
      ...hugeLeading,
      makeBlobEntry('user', 'do something'),
      ...makeToolGroup({ callId: 'call_d32', toolName: 'Read', input: { path: 'd.ts' }, resultTokens: 500 }),
    ]
    const plan = planCompaction(entries, { contextTokenLimit: 32_000 })
    expect(plan.mode).toBe('disabled')
    expect(plan.summarizeEntries.length).toBe(0)
  })

  it('#10-3 32K 窗正常 leading: budget 模式可用 (targetFloor − leading − reserve 路径)', () => {
    const entries = [
      ...makeLeadingEntries(),
      makeBlobEntry('user', makeVariedTokenText(400)),
      ...makeToolGroup({ callId: 'call_n32', toolName: 'Read', input: { path: 'n.ts' }, resultTokens: 6_000 }),
      ...makeToolGroup({ callId: 'call_n32b', toolName: 'Read', input: { path: 'n2.ts' }, resultTokens: 6_000 }),
    ]
    const plan = planCompaction(entries, { contextTokenLimit: 32_000 })
    expect(plan.mode).toBe('budget')
    // targetFloor 8K − leading − reserve ≈ 7.3K (未触底 min(8K, 5%×32K)=1.6K)
    expect(plan.diagnostics.budgetTokens).toBeGreaterThan(1_600)
    expect(plan.diagnostics.budgetTokens).toBeLessThanOrEqual(8_000)
  })
})

describe('#12/#13/#29 中文预算 / 性能冒烟 / 组边界', () => {
  it('#12 中文预算: 等 token 量中文工具流在 o200k 计数下不超预算', () => {
    const entries = [
      ...makeLeadingEntries(),
      makeBlobEntry('user', '请重构这个模块'),
    ]
    for (let index = 0; index < 30; index++) {
      entries.push(...makeToolGroup({
        callId: `call_cjk_${index}`,
        toolName: 'Read',
        input: { path: `文件${index}.ts` },
        resultTokens: 4_000,
        resultText: makeTokenSizedChineseText(4_000),
      }))
    }
    const plan = planCompaction(entries, { contextTokenLimit: 258_400 })
    expect(plan.summarizeEntries.length).toBeGreaterThan(0)
    // o200k 计价下 keepTail 受预算约束 (chars/4 会低估 4x 导致超支 — 修正后不超)
    expect(plan.diagnostics.keepTailActualTokens).toBeLessThanOrEqual(plan.diagnostics.budgetTokens + 2_000)
    expectNoSplitToolPairsInPlan(plan)
  })

  it('#13 性能冒烟: 200 条 (100 组) planCompaction < 1s', () => {
    const entries = [...makeLeadingEntries()]
    for (let index = 0; index < 100; index++)
      entries.push(...makeToolGroup({ callId: `call_perf_${index}`, toolName: 'Read', input: { path: `p${index}.ts` }, resultTokens: 1_200 }))
    expect(entries.length).toBe(202)

    const startedAt = Date.now()
    const plan = planCompaction(entries, { contextTokenLimit: 258_400 })
    const elapsedMs = Date.now() - startedAt
    expect(elapsedMs).toBeLessThan(1_000)
    expect(plan.mode).toBe('budget')
  }, 15_000)

  it('#29 纯工具流 70 组 (无任何 v1-safe 消息): 切点落在预算允许的最深组边界, 不钉死于 Σ/锚点', () => {
    const entries = [
      ...makeLeadingEntries(),
      makeBlobEntry('user', 'run the migration'),
    ]
    for (let index = 0; index < 70; index++)
      entries.push(...makeToolGroup({ callId: `call_g29_${index}`, toolName: 'Read', input: { path: `m${index}.ts` }, resultTokens: 3_000 }))

    const plan = planCompaction(entries, { contextTokenLimit: 258_400 })
    expect(plan.mode).toBe('budget')
    // 真实切分发生: 摘要侧与尾窗均非空 (v1 谓词在此场景会钉死切点 → 尾窗无界累积)
    expect(plan.summarizeEntries.length).toBeGreaterThan(0)
    expect(plan.keepTail.length).toBeGreaterThan(2)
    expect(plan.diagnostics.keepTailActualTokens).toBeLessThanOrEqual(plan.diagnostics.budgetTokens + 2_000)
    expectNoSplitToolPairsInPlan(plan)
    // 锚点: 尾窗无真 user → 指令锚点被回溯插入头部 (双保险)
    expect(plan.diagnostics.anchorInserted).toBe(true)
    expect(plan.keepTail[0]!.message.role).toBe('user')
  })
})

describe('#20/#21/#22/#23/#27/#28/#32 锚点 / 前沿 / 输入侧 / 升级链', () => {
  it('#20 指令锚点保底 + canonical 地板: 40×3K 对抗序列, 地板 ≤ 承诺', () => {
    const instructionText = `<user_query>\nrefactor the auth module and add tests\n</user_query>\n${makeVariedTokenText(700)}`
    const entries = [
      ...makeLeadingEntries(),
      makeBlobEntry('user', instructionText),
    ]
    // 70 轮工具流, 其中 40 组为 3K 中等消息 (对抗序列: 不触发巨物占位)
    for (let index = 0; index < 70; index++)
      entries.push(...makeToolGroup({ callId: `call_c20_${index}`, toolName: 'Read', input: { path: `a${index}.ts` }, resultTokens: index < 40 ? 3_000 : 2_000 }))

    const plan = planCompaction(entries, { contextTokenLimit: 258_400 })
    expect(plan.mode).toBe('budget')
    // 指令原文存活于尾窗头部 (锚点, 不截断)
    expect(plan.keepTail[0]!.message.role).toBe('user')
    expect(plan.keepTail[0]!.message.content).toBe(instructionText)
    // 双保险: 指令同时存在于摘要侧 (不移除)
    expect(plan.summarizeEntries.some(entry => entry.message.content === instructionText)).toBe(true)
    // archive 不含锚点 blobId
    expect(plan.anchorBlobId).toBeTruthy()
    expect(plan.elidedOriginals).not.toContain(plan.anchorBlobId)
    // tool_result 载体不被误选为锚点
    expect(plan.keepTail[0]!.message.toolCallId).toBeUndefined()
    // canonical 地板: 消息侧 ≤ 承诺地板 (targetFloor × 1.2)
    const occupancy = plan.diagnostics.leadingTokens + plan.diagnostics.summaryReserveTokens + plan.diagnostics.keepTailActualTokens
    expect(occupancy).toBeLessThanOrEqual(Math.floor(0.25 * 258_400 * 1.2))
  })

  it('#21 因果前沿豁免: 40K 结果刚落地不被占位; 消费一轮后再压缩则被占位 (单轮自愈)', () => {
    const baseEntries = (): HistoryEntry[] => {
      const entries = [
        ...makeLeadingEntries(),
        makeBlobEntry('user', makeVariedTokenText(500)),
      ]
      pushBulkHistory(entries, 6, 8_000)
      entries.push(...makeToolGroup({ callId: 'call_f21', toolName: 'Read', input: { path: 'fresh.ts' }, resultTokens: 40_000 }))
      return entries
    }
    // 场景 A: 结果是最后一条 (未消费) → 前沿豁免, 原文留尾窗
    const planFresh = planCompaction(baseEntries(), { contextTokenLimit: 258_400, budgetOverride: 5_000 })
    expect(planFresh.summarizeEntries.length).toBeGreaterThan(0)
    expect(planFresh.diagnostics.placeholderCount).toBe(0)
    expect(planFresh.keepTail.some(entry => entry.message.toolCallId === 'call_f21'
      && typeof entry.message.content === 'string' && !entry.message.content.includes('[tool output elided'))).toBe(true)
    expect(planFresh.diagnostics.frontierExcessTokens).toBeGreaterThan(0)

    // 场景 B: 追加 assistant (模拟消费) → 脱离前沿 → 可占位
    const consumedEntries = [...baseEntries(), makeBlobEntry('assistant', 'I have read the file, continuing')]
    const planConsumed = planCompaction(consumedEntries, { contextTokenLimit: 258_400, budgetOverride: 5_000 })
    expect(planConsumed.summarizeEntries.length).toBeGreaterThan(0)
    expect(planConsumed.diagnostics.placeholderCount).toBe(1)
    // 首次消费损失恒 0 (前沿豁免回归指标)
    expect(planConsumed.diagnostics.firstConsumptionLossCount).toBe(0)
    expect(planFresh.diagnostics.firstConsumptionLossCount).toBe(0)
  })

  it('#22 锚点计价: 20K 长指令触发锚点回溯, 以扣减预算重扫, 指令不截断', () => {
    const longInstruction = `Please carefully refactor the entire authentication subsystem ${makeVariedTokenText(19_000)}`
    const entries = [
      ...makeLeadingEntries(),
      makeBlobEntry('user', longInstruction),
    ]
    for (let index = 0; index < 40; index++)
      entries.push(...makeToolGroup({ callId: `call_c22_${index}`, toolName: 'Read', input: { path: `b${index}.ts` }, resultTokens: 3_000 }))

    const plan = planCompaction(entries, { contextTokenLimit: 258_400 })
    expect(plan.diagnostics.anchorInserted).toBe(true)
    // 指令原文完整 (不截断)
    expect(plan.keepTail[0]!.message.content).toBe(longInstruction)
    // 锚点计入预算: keepTail (含 20K 锚点) 受预算 + 锚点超额约束
    expect(plan.diagnostics.keepTailActualTokens).toBeLessThanOrEqual(plan.diagnostics.budgetTokens + 21_000)
  })

  it('#23 渲染后计价: 中文巨物 + 超长路径 — 占位符实测 token = 计价值 (账面=实际)', () => {
    const longPath = `/very/long/nested/directory/structure/that/keeps/going/${'segment/'.repeat(60)}file.ts`
    const entries = [
      ...makeLeadingEntries(),
      makeBlobEntry('user', makeVariedTokenText(300)),
    ]
    pushBulkHistory(entries, 6, 8_000)
    entries.push(...makeToolGroup({
      callId: 'call_c23',
      toolName: 'Read',
      input: { path: longPath },
      resultTokens: 30_000,
      resultText: makeTokenSizedChineseText(30_000),
    }))
    entries.push(makeBlobEntry('assistant', 'done reading'))

    const plan = planCompaction(entries, { contextTokenLimit: 258_400, budgetOverride: 3_000 })
    expect(plan.summarizeEntries.length).toBeGreaterThan(0)
    expect(plan.diagnostics.placeholderCount).toBe(1)
    // keepTail 实测 = 诊断里的 keepTailActualTokens (同一把尺子, 账面=实际)
    const remeasured = measureMessagesTokens(plan.keepTail.map(entry => entry.message))
    expect(remeasured).toBe(plan.diagnostics.keepTailActualTokens)
    // CJK 预览 token 封顶: 占位符 < 900 tok (头175+尾75+骨架+超长路径)
    const placeholderEntry = plan.keepTail.find(entry =>
      typeof entry.message.content === 'string' && entry.message.content.includes('[tool output elided'))!
    expect(countTokens(placeholderEntry.message.content as string)).toBeLessThan(900)
    expect(placeholderEntry.message.content).toContain(longPath)
  })

  it('#27 输入侧占位: assistant 含 30K Write.contents → 字段省略标记替换, id/name/path 原样', () => {
    const entries = [
      ...makeLeadingEntries(),
      makeBlobEntry('user', makeVariedTokenText(300)),
    ]
    pushBulkHistory(entries, 6, 8_000)
    entries.push(makeBlobEntry('assistant', [
      { type: 'text', text: 'writing the file now' },
      { type: 'tool_use', id: 'call_w27', name: 'Write', input: { path: '/repo/new-file.ts', contents: makeVariedTokenText(30_000) } },
    ]))
    entries.push(makeBlobEntry('tool', 'File written successfully', { toolCallId: 'call_w27', toolName: 'Write' }))
    entries.push(makeBlobEntry('assistant', 'file created'))

    const plan = planCompaction(entries, { contextTokenLimit: 258_400, budgetOverride: 3_000 })
    expect(plan.summarizeEntries.length).toBeGreaterThan(0)
    expect(plan.diagnostics.inputElidedCount).toBe(1)
    const elidedAssistant = plan.keepTail.find(entry =>
      Array.isArray(entry.message.content)
      && entry.message.content.some(block => typeof block === 'object' && 'input' in block
        && typeof (block as { input: Record<string, unknown> }).input.contents === 'string'
        && ((block as { input: Record<string, unknown> }).input.contents as string).includes('elided during context compaction')))
    expect(elidedAssistant).toBeTruthy()
    const toolUseBlock = (elidedAssistant!.message.content as LLMContentBlock[]).find(block => block.type === 'tool_use') as Extract<LLMContentBlock, { type: 'tool_use' }>
    // id / name / path 原样保留
    expect(toolUseBlock.id).toBe('call_w27')
    expect(toolUseBlock.name).toBe('Write')
    expect(toolUseBlock.input.path).toBe('/repo/new-file.ts')
    // locator 指向磁盘文件
    expect(toolUseBlock.input.contents).toMatch(/recover from the file at \/repo\/new-file\.ts/)
    // 配对完整
    expectNoSplitToolPairsInPlan(plan)
  })

  it('#28 输入侧前沿豁免: 最后一条 assistant 含大 tool_use 不省略; 消费后可省略', () => {
    const baseEntries = (): HistoryEntry[] => {
      const entries = [
        ...makeLeadingEntries(),
        makeBlobEntry('user', makeVariedTokenText(300)),
      ]
      pushBulkHistory(entries, 6, 8_000)
      entries.push(makeBlobEntry('assistant', [
        { type: 'tool_use', id: 'call_w28', name: 'Write', input: { path: '/repo/frontier.ts', contents: makeVariedTokenText(30_000) } },
      ]))
      return entries
    }
    // 场景 A: 该 assistant 是最后一条 → 前沿, 不省略
    const planFrontier = planCompaction(baseEntries(), { contextTokenLimit: 258_400, budgetOverride: 3_000 })
    expect(planFrontier.summarizeEntries.length).toBeGreaterThan(0)
    expect(planFrontier.diagnostics.inputElidedCount).toBe(0)
    // 场景 B: 追加消费 (tool result + assistant) → 可省略
    const consumedEntries = [
      ...baseEntries(),
      makeBlobEntry('tool', 'File written', { toolCallId: 'call_w28', toolName: 'Write' }),
      makeBlobEntry('assistant', 'written'),
    ]
    const planConsumed = planCompaction(consumedEntries, { contextTokenLimit: 258_400, budgetOverride: 3_000 })
    expect(planConsumed.summarizeEntries.length).toBeGreaterThan(0)
    expect(planConsumed.diagnostics.inputElidedCount).toBe(1)
  })

  it('#32 违约就地升级链: 实占 > 承诺×1.2 → 依次升级至 B 模式, 每级确定性终止', () => {
    // leading 80K (超 targetFloor 64.6K) + 图片巨物尾窗 (不可占位) → 违约穿透升级链 → B 模式终态
    const hugeLeading = [
      makeBlobEntry('system', makeVariedTokenText(79_000)),
      makeBlobEntry('user', `<user_info>\n${makeVariedTokenText(1_000)}\n</user_info>`),
    ]
    const entries = [
      ...hugeLeading,
      makeBlobEntry('user', 'keep working on the images'),
    ]
    for (let index = 0; index < 8; index++) {
      entries.push(makeBlobEntry('assistant', [{ type: 'tool_use', id: `call_i32_${index}`, name: 'Read', input: { path: `img${index}.png` } }]))
      entries.push(makeBlobEntry('user', [
        { type: 'tool_result', toolUseId: `call_i32_${index}`, toolName: 'Read', content: 'screenshot' },
        { type: 'image', mimeType: 'image/png', data: 'aGVsbG8=' },
      ]))
    }
    const plan = planCompaction(entries, { contextTokenLimit: 258_400 })
    // 违约无法通过占位线/预算减半消除 (图片原子豁免) → B 模式终态
    expect(plan.mode).toBe('b-mode')
    expect(plan.diagnostics.escalationLevel).toBe('b-mode')
    // B 模式地板 = 全量摘要 + 锚点单条
    expect(plan.keepTail.length).toBe(1)
    expect(plan.keepTail[0]!.message.role).toBe('user')
  })
})

describe('#11/#16/#17/#33 摘要源治理与三级兜底 (阶段 4)', () => {
  it('#11 摘要源封顶: 巨物进摘要侧 → 总长 ≤ 预算, 路径清单与错误行保留', () => {
    // 258,400 窗 → 摘要源预算 min(0.6×258400×4, 3.2e6) = 620,160 chars
    const entries: HistoryEntry[] = []
    const toolResultWithPaths = [
      'Analyzing module dependencies...',
      'Read /repo/src/auth/login.ts',
      'Read /repo/src/auth/session.ts',
      'ERROR: cannot resolve module /repo/src/missing.ts',
      makeVariedTokenText(400_000),
      'Conclusion: the auth module requires session refactor',
    ].join('\n')
    entries.push(makeBlobEntry('user', 'investigate auth'))
    entries.push(makeBlobEntry('assistant', [{ type: 'tool_use', id: 'call_s11', name: 'Read', input: { path: '/repo/src/auth.ts' } }]))
    entries.push(makeBlobEntry('tool', toolResultWithPaths, { toolCallId: 'call_s11', toolName: 'Read' }))
    entries.push(makeBlobEntry('assistant', 'done'))

    // 用小窗口把预算压到 ~48K chars (0.6×20_000×4), 强制水位分配截断
    const source = buildSummarySource(entries, { contextTokenLimit: 20_000 })
    const totalBudget = computeSummarySourceBudgetChars(20_000)
    expect(totalBudget).toBe(48_000)
    expect(source.length).toBeLessThanOrEqual(totalBudget + 2_000)
    // 路径清单与错误行强制保留
    expect(source).toContain('/repo/src/auth/login.ts')
    expect(source).toContain('ERROR')
  })

  it('#16 摘要三级兜底: LLM 三次失败 → 确定性降级 (水位分配产物 + 注入防御声明)', async () => {
    const failingProvider = {
      async* stream() {
        throw new Error('provider unavailable')
      },
    }
    const sourceText = makeVariedTokenText(2_000)
    const summaryText = await generateSummaryWithFallback({
      provider: failingProvider,
      model: 'test-model',
      sourceText,
      contextTokenLimit: 258_400,
    })

    // 确定性降级: 不经模型, 含注入防御声明与转录内容
    expect(summaryText).toContain('not instructions from the user')
    expect(summaryText).toContain(sourceText.slice(0, 100))
    // 降级预算 = clamp(258400×2%×4, 50K, 3.2M) = 50K chars — 全量保留
    expect(summaryText.length).toBeGreaterThan(2_000)
  })

  it('#35 挂死网关: 流永不产出事件 → 限时超时驱动兜底梯子, 有界时间内出确定性降级 (F3)', async () => {
    const hangingProvider = {
      stream: () => ({
        [Symbol.asyncIterator]: () => ({
          // next() 永不 resolve — 复刻实弹事故里挂死 ~4 分钟的网关
          next: () => new Promise<IteratorResult<{ type: string, text?: string }>>(() => {}),
        }),
      }),
    }
    const sourceText = `hanging-gateway-test ${makeVariedTokenText(500)}`
    const ladderStartTime = Date.now()
    const summaryText = await generateSummaryWithFallback({
      provider: hangingProvider,
      model: 'test-model',
      sourceText,
      contextTokenLimit: 120_000,
      timeoutsOverride: { firstEventMs: 60, stallMs: 60, totalMs: 200 },
    })
    const ladderElapsedMs = Date.now() - ladderStartTime

    // 三次限时尝试 (≤200ms each) + 确定性降级 — 全程远低于旧实现的无限期挂死
    expect(ladderElapsedMs).toBeLessThan(5_000)
    expect(summaryText).toContain('not instructions from the user')
    expect(summaryText).toContain('hanging-gateway-test')
  })

  it('#36 摘要请求显式输出上限: maxTokens = clamp(2×hardCap, 4096, 16384) (F4)', async () => {
    const capturedRequests: Array<{ maxTokens?: number }> = []
    const recordingProvider = {
      async* stream(request: { model: string, maxTokens?: number }) {
        capturedRequests.push({ maxTokens: request.maxTokens })
        yield { type: 'text_delta', text: '- summary line' }
      },
    }
    await generateSummaryWithFallback({
      provider: recordingProvider,
      model: 'test-model',
      sourceText: makeVariedTokenText(300),
      contextTokenLimit: 120_000,
    })

    // 120K 窗: hardCap = 2×min(5000, 2%×120000=2400) = 4800 → maxTokens = 9600
    expect(capturedRequests).toHaveLength(1)
    expect(capturedRequests[0]!.maxTokens).toBe(computeSummaryMaxOutputTokens(120_000))
    expect(capturedRequests[0]!.maxTokens).toBe(9_600)
    // 巨窗上界封顶 16384: 1M 窗 hardCap = 2×5000 = 10000 → 2×10000 = 20000 → clamp 16384
    expect(computeSummaryMaxOutputTokens(1_000_000)).toBe(16_384)
  })

  it('#37 摘要推理档位: 仅 openai-responses × thinking 模型传 low, 其余不传 (F6)', async () => {
    // 决策矩阵
    expect(resolveSummaryThinkingLevel({ provider: { name: 'openai-responses' }, thinking: true })).toBe('low')
    expect(resolveSummaryThinkingLevel({ provider: { name: 'openai-responses' }, thinking: false })).toBeUndefined()
    expect(resolveSummaryThinkingLevel({ provider: { name: 'anthropic' }, thinking: true })).toBeUndefined()
    expect(resolveSummaryThinkingLevel({ provider: { name: 'openai-chat' }, thinking: true })).toBeUndefined()
    expect(resolveSummaryThinkingLevel({ provider: { name: 'gemini' }, thinking: true })).toBeUndefined()

    // 端到端透传: thinkingLevel 出现在 provider.stream 请求上
    const capturedRequests: Array<{ thinkingLevel?: string }> = []
    const recordingProvider = {
      async* stream(request: { model: string, thinkingLevel?: string }) {
        capturedRequests.push({ thinkingLevel: request.thinkingLevel })
        yield { type: 'text_delta', text: '- summary line' }
      },
    }
    await generateSummaryWithFallback({
      provider: recordingProvider,
      model: 'test-model',
      sourceText: makeVariedTokenText(300),
      contextTokenLimit: 120_000,
      thinkingLevel: 'low',
    })
    expect(capturedRequests[0]!.thinkingLevel).toBe('low')
  })

  it('#17 水位分配器: 200 条不等长消息 — min-quota 丢弃占位, <user_query> 块存续', () => {
    const entries: HistoryEntry[] = []
    // 1 条含 <user_query> 的 user 消息 (会被截断但保 query 块) + 199 条不等长消息
    const longUserQuery = `<user_query>\nrefactor the entire auth subsystem carefully\n</user_query>\n${makeVariedTokenText(30_000)}`
    entries.push(makeBlobEntry('user', longUserQuery))
    for (let index = 0; index < 199; index++) {
      const sizeClass = index % 3
      entries.push(makeBlobEntry(
        index % 2 === 0 ? 'user' : 'assistant',
        makeVariedTokenText(sizeClass === 0 ? 50 : sizeClass === 1 ? 1_200 : 8_000),
      ))
    }

    // 小窗口: 预算 0.6×20_000×4 = 48K chars, 199 条总量 ≈ 89K+ → 强制分配
    const source = buildSummarySource(entries, { contextTokenLimit: 20_000 })
    expect(source.length).toBeLessThanOrEqual(computeSummarySourceBudgetChars(20_000) + 3_000)
    // 截断的 user 消息保留 <user_query> 块
    expect(source).toContain('<user_query>')
    expect(source).toContain('refactor the entire auth subsystem carefully')
    // 短消息 (50 tok) 整条保留 / 长消息被截断标注
    expect(source).toMatch(/\[\.\.\. truncated, \d+ chars\]/)
  })

  it('#33 摘要输出硬上界: mock 摘要器输出 20K → shorter-output 重试 → 仍超则裁剪至 SUMMARY_HARD_CAP', async () => {
    // 恒定输出 20K tok 的"劣质"摘要器 (忽略指令)
    const alwaysVerboseProvider = {
      async* stream() {
        yield { type: 'text_delta', text: makeVariedTokenText(20_000) }
      },
    }
    const summaryText = await generateSummaryWithFallback({
      provider: alwaysVerboseProvider,
      model: 'test-model',
      sourceText: makeVariedTokenText(1_000),
      contextTokenLimit: 258_400,
    })

    // 258,400 窗 → SUMMARY_HARD_CAP = 2 × 5,000 = 10,000 tok
    expect(computeSummaryHardCapTokens(258_400)).toBe(10_000)
    expect(countTokens(summaryText)).toBeLessThanOrEqual(10_000)
  })
})

describe('#25 并发互斥 (compactionLock)', () => {
  it('#25 inline 与 summarizeAction 并发: 后到 inline 跳过, 释放后 summarizeAction 可进入', async () => {
    const conversationId = 'conv-mutex-25'
    // inline 先拿到锁
    expect(tryAcquireCompactionLock(conversationId)).toBe(true)
    // 第二个尝试 (inline 语义) → 跳过并计数
    expect(tryAcquireCompactionLock(conversationId)).toBe(false)
    expect(tryAcquireCompactionLock(conversationId)).toBe(false)
    expect(getCompactionContentionCount(conversationId)).toBe(2)
    // summarizeAction 语义: 等待释放
    let resumed = false
    const waitPromise = waitForCompactionLockRelease(conversationId).then(() => {
      resumed = true
    })
    await new Promise(resolve => setTimeout(resolve, 20))
    expect(resumed).toBe(false)
    releaseCompactionLock(conversationId)
    await waitPromise
    expect(resumed).toBe(true)
    // 释放后可重新获取
    expect(tryAcquireCompactionLock(conversationId)).toBe(true)
    releaseCompactionLock(conversationId)
  })
})

describe('#15/#18 错误驱动重试与触发公式 (阶段 5)', () => {
  it('#15 context-length 错误分类: 白名单文案命中, 非 window 类错误不命中', async () => {
    expect(isContextLengthLimitError(new Error('This model\'s maximum context length is 128000 tokens. However, you requested 150000 tokens.'))).toBe(true)
    expect(isContextLengthLimitError(new Error('Error code: 400 - prompt is too long: 210000 tokens > 200000 maximum'))).toBe(true)
    expect(isContextLengthLimitError(new Error('input token count exceeds the maximum number of input tokens'))).toBe(true)
    expect(isContextLengthLimitError(new Error('context_length_exceeded'))).toBe(true)
    // 非白名单: 普通 provider 故障
    expect(isContextLengthLimitError(new Error('rate limit exceeded'))).toBe(false)
    expect(isContextLengthLimitError(new Error('connection timeout'))).toBe(false)
    expect(isContextLengthLimitError(new Error('invalid api key'))).toBe(false)
    expect(isContextLengthLimitError(null)).toBe(false)
  })

  it('#15 错误驱动 aggressive 压缩: budget/2^retry 逐轮减半, ≤3 轮', () => {
    // 模拟 provider 前两轮报 context-length 错误 → 第三轮成功的 ladder:
    // planCompaction 以 budgetOverride (基准/2^retry) 规划, 摘要侧逐轮扩大
    const baseBudget = 59_580 // 258,400 窗小 leading 下的典型基准
    const aggressiveBudgets = [1, 2, 3].map(retry => Math.max(1, Math.floor(baseBudget / 2 ** retry)))
    expect(aggressiveBudgets[0]).toBe(29_790)
    expect(aggressiveBudgets[1]).toBe(14_895)
    expect(aggressiveBudgets[2]).toBe(7_447)
    // 重试上限 3 (constants)
    expect(CONTEXT_LENGTH_RETRY_MAX).toBe(3)
    // aggressive 档下 keepTail 单调不增 (预算减半 → 尾窗更小)
    const entries: HistoryEntry[] = [...makeLeadingEntries(), makeBlobEntry('user', makeVariedTokenText(500))]
    for (let index = 0; index < 40; index++)
      entries.push(...makeToolGroup({ callId: `call_c15_${index}`, toolName: 'Read', input: { path: `p${index}.ts` }, resultTokens: 3_000 }))
    const planBase = planCompaction(entries, { contextTokenLimit: 258_400 })
    const planAggressive = planCompaction(entries, { contextTokenLimit: 258_400, budgetOverride: aggressiveBudgets[1] })
    expect(planAggressive.diagnostics.budgetTokens).toBe(aggressiveBudgets[1])
    expect(planAggressive.diagnostics.keepTailActualTokens).toBeLessThanOrEqual(planBase.diagnostics.keepTailActualTokens)
    expect(planAggressive.summarizeEntries.length).toBeGreaterThanOrEqual(planBase.summarizeEntries.length)
  })

  it('#18 触发线公式: 六档逐值断言 (窗口 − min(40K, 15%×窗口))', () => {
    expect(getAutoCompactThreshold(32_000)).toBe(27_200)
    expect(getAutoCompactThreshold(64_000)).toBe(54_400)
    expect(getAutoCompactThreshold(96_000)).toBe(81_600)
    expect(getAutoCompactThreshold(128_000)).toBe(108_800)
    expect(getAutoCompactThreshold(258_400)).toBe(219_640)
    expect(getAutoCompactThreshold(1_000_000)).toBe(960_000)
    // 32K/64K 死带消除: 旧公式分别取 −8K / 24K
    expect(getAutoCompactThreshold(32_000)).toBeGreaterThan(0)
    expect(getAutoCompactThreshold(64_000)).toBeGreaterThan(50_000)
  })
})

describe('sanity', () => {
  it('estimateMessagesTokens 可调用', () => {
    const messages: LLMMessage[] = [
      { role: 'user', content: 'hello world' },
      { role: 'assistant', content: 'hi' },
    ]
    expect(estimateMessagesTokens(messages)).toBeGreaterThan(0)
  })
})

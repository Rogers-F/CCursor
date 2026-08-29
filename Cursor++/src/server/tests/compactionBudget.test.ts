/**
 * compactionBudget.test.ts — 第二阶段自动压缩修复 (keepTail 预算化) 测试
 *
 * 用例编号对应设计文档 §8 测试计划表 (#1-#34), 表为权威清单。
 * 分阶段落地: 阶段 1 (入口截断/image) → 阶段 5 (触发公式/重试) 逐批补齐。
 */
import type { HistoryEntry } from '../handlers/agent/historyManager'
import type { LLMContentBlock, LLMMessage } from '../handlers/llm/types'
import { readFileSync, rmSync } from 'node:fs'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'
import { getSpillDir } from '../config/paths'
import { resetAgentDatabaseForTests } from '../database/sqlite'
import { encodeBlob } from '../handlers/agent/blob'
import { cacheBlob, resetBlobCacheForTests } from '../handlers/agent/blobStore'
import { estimateMessagesTokens, formatMessageForSummary } from '../handlers/agent/compactionStrategy'
import { countTokens } from '../handlers/agent/tokenCounter'
import {
  buildTaskToolResultText,
  resolveTaskEntryCapTokens,
  TASK_ENTRY_CAP_MAX_TOKENS,
} from '../handlers/agent/toolkit/results/taskToolResults'

// ─── helpers ───

export function makeBlobEntry(
  role: LLMMessage['role'],
  content: string | LLMContentBlock[],
  extra?: Record<string, unknown>,
): HistoryEntry {
  const raw: Record<string, unknown> = { role, content, ...extra }
  const blob = encodeBlob(raw)
  cacheBlob(blob.blobId, blob.blobData)
  return {
    blobId: blob.blobId,
    raw,
    message: { role, content },
  }
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

/** 生成指定 o200k token 量级的中文文本 (chars/4 低估 4x 场景) */
export function makeTokenSizedChineseText(tokens: number): string {
  let text = '汉字内容片段'.repeat(Math.ceil(tokens / 3))
  while (countTokens(text) > tokens) {
    text = text.slice(0, Math.max(0, text.length - 2))
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

// ─── 占位符: 后续阶段测试占位, 保证文件始终可运行 ───

describe('sanity', () => {
  it('estimateMessagesTokens 可调用', () => {
    const messages: LLMMessage[] = [
      { role: 'user', content: 'hello world' },
      { role: 'assistant', content: 'hi' },
    ]
    expect(estimateMessagesTokens(messages)).toBeGreaterThan(0)
  })
})

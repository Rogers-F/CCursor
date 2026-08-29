/**
 * Token 计数工具 — 基于 gpt-tokenizer (o200k_base)
 *
 * 用于 Context Window breakdown 估算。跨 provider 误差 ~10-15%，
 * 足以驱动 UI 进度条显示。不用于计费。
 */
import { decode, encode } from 'gpt-tokenizer/encoding/o200k_base'

/**
 * 同字符长游程阈值。
 *
 * gpt-tokenizer 的 BPE 在"同一字符连续重复"的长游程上会退化到平方级耗时
 * (实测 200K 个连续 'y' 编码需 13s; 无空格混合内容仅 5ms —— 退化只发生在
 * 等值 token 反复合并使 token 字符串不断增长的场景)。base64 编码的二进制零段
 * ("AAAA...") 等真实数据也能触发。
 *
 * 处理: 计数时把超长游程按"512 字符样本实测比率线性外推"估算 —— 保留量级
 * (游程 token 效率随长度略增, 线性外推偏保守/高估, 对预算制安全),
 * 非游程段精确编码。自然文本不含 >256 的同字符游程, 不走该分支。
 */
const SAME_CHAR_RUN_LIMIT = 256
const OVERSIZED_SAME_CHAR_RUN_RE = /(.)\1{255,}/
const RUN_ESTIMATE_SAMPLE_LENGTH = 512

function hasOversizedSameCharRuns(text: string): boolean {
  return OVERSIZED_SAME_CHAR_RUN_RE.test(text)
}

/** 游程 token 数: 512 字符样本实测比率线性外推 (样本内精确编码, 无退化) */
function estimateTokensForSameCharRun(runChar: string, runLength: number): number {
  const sampleLength = Math.min(runLength, RUN_ESTIMATE_SAMPLE_LENGTH)
  const sampleTokens = encode(runChar.repeat(sampleLength), { allowedSpecial: 'all' }).length
  const tokensPerChar = sampleTokens / sampleLength
  return Math.ceil(runLength * tokensPerChar)
}

/** 有界样本的诚实 token/字符比率 (不走折叠, 样本长度有界故无退化风险) */
function measureTokensPerChar(text: string): number {
  const sample = text.slice(0, RUN_ESTIMATE_SAMPLE_LENGTH)
  const sampleTokens = encode(sample, { allowedSpecial: 'all' }).length
  return Math.max(1 / 16, sampleTokens / Math.max(1, sample.length))
}

function countTokensWithRunEstimates(text: string): number {
  let total = 0
  let cursor = 0
  const pattern = new RegExp(`(.)\\1{${SAME_CHAR_RUN_LIMIT - 1},}`, 'g')
  let match: RegExpExecArray | null
  while ((match = pattern.exec(text)) !== null) {
    if (match.index > cursor)
      total += encode(text.slice(cursor, match.index), { allowedSpecial: 'all' }).length
    total += estimateTokensForSameCharRun(match[1]!, match[0].length)
    cursor = match.index + match[0].length
  }
  if (cursor < text.length)
    total += encode(text.slice(cursor), { allowedSpecial: 'all' }).length
  return total
}

export function countTokens(text: string): number {
  if (!text) return 0
  if (!hasOversizedSameCharRuns(text))
    return encode(text, { allowedSpecial: 'all' }).length
  return countTokensWithRunEstimates(text)
}

/**
 * token 级头尾切片: 头 headTokens + 尾 tailTokens, 中段丢弃。
 *
 * 用于入口截断与压缩占位符预览 — token 封顶 (而非字符封顶) 保证
 * CJK 内容下截断产物有构造上界 (中文 1000 字符 ≈ 1000 tok, 字符封顶会被击穿)。
 *
 * 病态输入 (同字符长游程) 精确编码会退化到秒级, 按有界样本比率折算字符切点。
 */
export function sliceTextHeadTailTokens(text: string, headTokens: number, tailTokens: number): { head: string, tail: string } {
  if (!text)
    return { head: '', tail: '' }

  if (hasOversizedSameCharRuns(text)) {
    const tokensPerChar = measureTokensPerChar(text)
    const headChars = Math.min(text.length, Math.floor(headTokens / tokensPerChar))
    const tailChars = Math.min(text.length - headChars, Math.floor(tailTokens / tokensPerChar))
    return {
      head: text.slice(0, headChars),
      tail: text.slice(Math.max(headChars, text.length - tailChars)),
    }
  }

  const tokens = encode(text, { allowedSpecial: 'all' })
  if (tokens.length <= headTokens + tailTokens)
    return { head: text, tail: '' }
  const head = decode(tokens.slice(0, Math.max(0, headTokens)))
  const tail = decode(tokens.slice(Math.max(0, tokens.length - Math.max(0, tailTokens))))
  return { head, tail }
}

/** token 级头部截取: 保留前 maxTokens 个 token */
export function takeTextByTokens(text: string, maxTokens: number): string {
  if (!text || maxTokens <= 0)
    return ''

  if (hasOversizedSameCharRuns(text))
    return text.slice(0, Math.min(text.length, Math.floor(maxTokens / measureTokensPerChar(text))))

  const tokens = encode(text, { allowedSpecial: 'all' })
  if (tokens.length <= maxTokens)
    return text
  return decode(tokens.slice(0, maxTokens))
}

export type ContextCategory =
  | 'system_prompt'
  | 'tools'
  | 'rules'
  | 'skills'
  | 'mcp'
  | 'subagents'
  | 'conversation'
  | 'summarized_conversation'

const CATEGORY_LABELS: Record<ContextCategory, string> = {
  system_prompt: 'System prompt',
  tools: 'Tool definitions',
  rules: 'Rules',
  skills: 'Skills',
  mcp: 'MCP & dynamic tools',
  subagents: 'Subagent definitions',
  conversation: 'Conversation',
  summarized_conversation: 'Summarized conversation',
}

export class ContextTokenTracker {
  private counts = new Map<ContextCategory, number>()

  add(category: ContextCategory, tokens: number): void {
    this.counts.set(category, (this.counts.get(category) ?? 0) + tokens)
  }

  addText(category: ContextCategory, text: string): void {
    if (text) this.add(category, countTokens(text))
  }

  get(category: ContextCategory): number {
    return this.counts.get(category) ?? 0
  }

  get total(): number {
    let sum = 0
    for (const v of this.counts.values()) sum += v
    return sum
  }

  toBreakdownCategories(): Array<{ id: string, label: string, estimatedTokens: number }> {
    const out: Array<{ id: string, label: string, estimatedTokens: number }> = []
    for (const [id, tokens] of this.counts) {
      if (tokens > 0)
        out.push({ id, label: CATEGORY_LABELS[id] ?? id, estimatedTokens: tokens })
    }
    return out
  }
}

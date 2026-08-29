import type { HistoryEntry } from '../handlers/agent/historyManager'
import type { LLMMessage } from '../handlers/llm/types'
import { unlinkSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, beforeEach, expect, it } from 'vitest'
import { resetAgentDatabaseForTests } from '../database/sqlite'
import { encodeBlob } from '../handlers/agent/blob'
import { cacheBlob, resetBlobCacheForTests } from '../handlers/agent/blobStore'
import { createCompactionArtifacts, estimateMessagesTokens, formatMessageForSummary, planCompaction } from '../handlers/agent/compactionStrategy'
import { hydrateHistoryEntries, isSummaryBlobMessage } from '../handlers/agent/historyManager'
import { clampTokenDetails, computeContextUsagePercent, getAutoCompactThreshold, shouldTriggerCompaction } from '../handlers/agent/usage'

// ─── helpers ───

function makeBlobEntry(role: string, content: string, extra?: Record<string, unknown>): { blobId: string, raw: Record<string, unknown>, message: LLMMessage } {
  const raw: Record<string, unknown> = { role, content, ...extra }
  const blob = encodeBlob(raw)
  cacheBlob(blob.blobId, blob.blobData)
  return {
    blobId: blob.blobId,
    raw,
    message: { role: role as 'system' | 'user' | 'assistant', content },
  }
}

function makeHistoryEntries(count: number, opts?: { withSystem?: boolean, withPreamble?: boolean }): HistoryEntry[] {
  const entries: HistoryEntry[] = []
  if (opts?.withSystem) {
    entries.push(makeBlobEntry('system', 'You are a helpful assistant.'))
  }
  if (opts?.withPreamble) {
    entries.push(makeBlobEntry('user', '<user_info>\nUser context here\n</user_info>'))
  }
  for (let i = 0; i < count; i++) {
    const role = i % 2 === 0 ? 'user' : 'assistant'
    const content = role === 'user'
      ? `User message ${Math.floor(i / 2) + 1}: ${'x'.repeat(200)}`
      : `Assistant response ${Math.floor(i / 2) + 1}: ${'y'.repeat(300)}`
    entries.push(makeBlobEntry(role, content))
  }
  return entries
}

// ─── setup / teardown ───

let tmpDbPath = ''

beforeEach(async () => {
  tmpDbPath = join(tmpdir(), `.tmp-auto-summarize-${Date.now()}-${Math.random().toString(36).slice(2)}.db`)
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
      unlinkSync(`${tmpDbPath}${suffix}`)
    }
    catch {}
  }
})

// ─── shouldTriggerCompaction tests ───

it('shouldTriggerCompaction returns false below threshold', () => {
  expect(shouldTriggerCompaction(80000, 100000, 85)).toBe(false)
  expect(shouldTriggerCompaction(84000, 100000, 85)).toBe(false)
})

it('shouldTriggerCompaction returns true at/above threshold', () => {
  expect(shouldTriggerCompaction(85000, 100000, 85)).toBe(true)
  expect(shouldTriggerCompaction(90000, 100000, 85)).toBe(true)
  expect(shouldTriggerCompaction(100000, 100000, 85)).toBe(true)
})

it('shouldTriggerCompaction default uses absolute buffer threshold', () => {
  // 绝对 buffer 模式 (对齐 Claude Code):
  //   threshold = maxTokens - min(maxOutputTokens, 20K outputReserve) - 20K buffer
  // 默认 maxOutputTokens=8192 → 100K 模型 threshold = 100000 - 8192 - 20000 = 71808
  expect(getAutoCompactThreshold(100000)).toBe(71808)
  expect(shouldTriggerCompaction(71807, 100000)).toBe(false)
  expect(shouldTriggerCompaction(71808, 100000)).toBe(true)
})

it('getAutoCompactThreshold caps output reserve at 20K', () => {
  // maxOutputTokens 超过 20K 时按 20K 计 — 200K 模型 threshold=160K (~80%, 基准)
  expect(getAutoCompactThreshold(200000, 64000)).toBe(160000)
  // 1M 模型 threshold=960K (~96%)
  expect(getAutoCompactThreshold(1000000, 32000)).toBe(960000)
})

// ─── clampTokenDetails tests ───

it('clampTokenDetails clamps to valid range', () => {
  const result = clampTokenDetails(150000, 100000)
  expect(result.usedTokens).toBe(100000)
  expect(result.maxTokens).toBe(100000)
})

it('clampTokenDetails handles zero maxTokens', () => {
  const result = clampTokenDetails(5000, 0)
  expect(result.maxTokens).toBe(1)
  expect(result.usedTokens).toBe(1)
})

it('clampTokenDetails passes through valid values', () => {
  const result = clampTokenDetails(50000, 200000)
  expect(result.usedTokens).toBe(50000)
  expect(result.maxTokens).toBe(200000)
})

// ─── computeContextUsagePercent tests ───

it('computeContextUsagePercent computes correctly', () => {
  expect(computeContextUsagePercent(85000, 100000)).toBe(85)
  expect(computeContextUsagePercent(0, 100000)).toBe(0)
  expect(computeContextUsagePercent(100000, 100000)).toBe(100)
})

// ─── planCompaction tests (第二阶段: token 预算制, 原"按条数"断言改为预算断言) ───

function makeVariedFiller(chars: number): string {
  const words = ['alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta', 'eta', 'theta', 'iota', 'kappa']
  let text = ''
  let wordIndex = 0
  while (text.length < chars)
    text += `${words[wordIndex++ % words.length]} `
  return text
}

function makeLargeHistoryEntries(count: number, opts?: { withSystem?: boolean, withPreamble?: boolean, tokensPerEntry?: number }): HistoryEntry[] {
  const tokensPerEntry = opts?.tokensPerEntry ?? 6_000
  const entries: HistoryEntry[] = []
  if (opts?.withSystem) {
    entries.push(makeBlobEntry('system', 'You are a helpful assistant.'))
  }
  if (opts?.withPreamble) {
    entries.push(makeBlobEntry('user', '<user_info>\nUser context here\n</user_info>'))
  }
  for (let i = 0; i < count; i++) {
    const role = i % 2 === 0 ? 'user' : 'assistant'
    const content = `${role === 'user' ? 'User message' : 'Assistant response'} ${Math.floor(i / 2) + 1}: ${makeVariedFiller(tokensPerEntry * 5)}`
    entries.push(makeBlobEntry(role, content))
  }
  return entries
}

it('planCompaction preserves system and preamble in leading', () => {
  // 258,400 窗 + 10×5K tok: budget ≈ 44K, 扫描后只保留尾部若干组, 但 leading 恒完整保留
  const entries = makeLargeHistoryEntries(10, { withSystem: true, withPreamble: true })
  const plan = planCompaction(entries, { contextTokenLimit: 258_400 })

  expect(plan.leading.length).toBe(2)
  expect(plan.leading[0].message.role).toBe('system')
  expect((plan.leading[1].message.content as string).includes('<user_info>')).toBe(true)
  expect(plan.summarizeEntries.length).toBeGreaterThan(0)
  expect(plan.keepTail.length).toBeGreaterThan(0)
  // 预算断言: keepTail 实占 (o200k) 不超过预算 + 安全边际余量
  expect(plan.diagnostics.keepTailActualTokens).toBeLessThanOrEqual(plan.diagnostics.budgetTokens + 1_000)
})

it('planCompaction keeps system+preamble in leading, compacts body', () => {
  // 预算制下 1 条小 body 全部落在预算内 → 无需压缩 (旧"条数定额"会把单条也标记摘要)
  const entries = makeHistoryEntries(1, { withSystem: true, withPreamble: true })
  const plan = planCompaction(entries, { contextTokenLimit: 258_400 })

  expect(plan.leading.length).toBe(2) // system + preamble
  expect(plan.summarizeEntries.length).toBe(0)
  expect(plan.keepTail.length).toBe(1)
  expect(plan.mode).toBe('budget')
})

it('planCompaction splits medium conversations correctly (预算断言)', () => {
  // 6 条 × 5K tok ≈ 30K, budget ≈ 44K → 全部装得下 → 不压缩;
  // 加大单条体积到 15K tok (6×15K=90K > 44K) → 必须切分
  const fitsEntirely = makeLargeHistoryEntries(6, { withSystem: true, tokensPerEntry: 5_000 })
  const planNoop = planCompaction(fitsEntirely, { contextTokenLimit: 258_400 })
  expect(planNoop.summarizeEntries.length).toBe(0)

  const oversized = makeLargeHistoryEntries(6, { withSystem: true, tokensPerEntry: 15_000 })
  const plan = planCompaction(oversized, { contextTokenLimit: 258_400 })
  expect(plan.leading.length).toBe(1) // system
  expect(plan.summarizeEntries.length).toBeGreaterThan(0)
  expect(plan.keepTail.length).toBeGreaterThan(0)
  expect(plan.summarizeEntries.length + plan.keepTail.length).toBe(6)
  // keepTail 实占受预算约束 (中等工作集: 15K/条, 不足以触发巨物占位)
  expect(plan.diagnostics.keepTailActualTokens).toBeLessThanOrEqual(plan.diagnostics.budgetTokens + 2_000)
})

// ─── createCompactionArtifacts tests ───

it('createCompactionArtifacts produces valid summary blob and archive', () => {
  // 20×5K tok 超出 44K 预算 → 产生真实的摘要侧 + archive
  const entries = makeLargeHistoryEntries(20, { withSystem: true, withPreamble: true })
  const plan = planCompaction(entries, { contextTokenLimit: 258_400 })
  expect(plan.summarizeEntries.length).toBeGreaterThan(0)

  const artifacts = createCompactionArtifacts({
    plan,
    summaryText: '- User asked about X.\n- Assistant explained Y.',
    previousSummaryArchiveIds: [],
  })

  expect(artifacts.summaryBlobId).toBeTruthy()
  expect(artifacts.summaryBlobData).toBeTruthy()
  expect(artifacts.archiveBlobs.length > 0).toBeTruthy()
  expect(artifacts.nextRootBlobIds.length > 0).toBeTruthy()
  expect(artifacts.nextSummaryArchiveIds.length > 0).toBeTruthy()

  // nextRootBlobIds = leading + summary + keepTail
  expect(
    artifacts.nextRootBlobIds.length,
  ).toBe(
    plan.leading.length + 1 + plan.keepTail.length,
  )

  // summary blob should be in root IDs
  expect(artifacts.nextRootBlobIds.includes(artifacts.summaryBlobId)).toBeTruthy()
})

it('createCompactionArtifacts preserves previous summary archive IDs', () => {
  const entries = makeLargeHistoryEntries(20, { withSystem: true })
  const plan = planCompaction(entries, { contextTokenLimit: 258_400 })
  expect(plan.summarizeEntries.length).toBeGreaterThan(0)

  const artifacts = createCompactionArtifacts({
    plan,
    summaryText: '- Previous summary context.',
    previousSummaryArchiveIds: ['old-archive-1', 'old-archive-2'],
  })

  expect(artifacts.nextSummaryArchiveIds.includes('old-archive-1')).toBeTruthy()
  expect(artifacts.nextSummaryArchiveIds.includes('old-archive-2')).toBeTruthy()
  expect(artifacts.nextSummaryArchiveIds.length >= 3).toBeTruthy() // old + new
})

// ─── hydrateHistoryEntries tests ───

it('hydrateHistoryEntries recovers cached blobs', () => {
  const entries = makeHistoryEntries(4)
  const blobIds = entries.map(e => e.blobId)

  const hydrated = hydrateHistoryEntries(blobIds)

  expect(hydrated.length).toBe(4)
  expect(hydrated[0].message.role).toBe('user')
  expect(hydrated[1].message.role).toBe('assistant')
})

it('hydrateHistoryEntries skips missing blobs', () => {
  const entries = makeHistoryEntries(2)
  const blobIds = [entries[0].blobId, 'nonexistent-blob-id', entries[1].blobId]

  const hydrated = hydrateHistoryEntries(blobIds)

  expect(hydrated.length).toBe(2)
})

// ─── isSummaryBlobMessage tests ───

it('isSummaryBlobMessage detects summary blobs', () => {
  expect(isSummaryBlobMessage({
    role: 'assistant',
    content: 'Previous conversation summary:\n- stuff',
    providerOptions: { cursor: { isSummary: true } },
  })).toBe(true)

  expect(isSummaryBlobMessage({
    role: 'assistant',
    content: 'Normal response',
  })).toBe(false)
})

// ─── formatMessageForSummary tests ───

it('formatMessageForSummary formats string content', () => {
  const result = formatMessageForSummary({ role: 'user', content: 'Hello world' })
  expect(result).toMatch(/user:\nHello world/)
})

it('formatMessageForSummary formats content blocks', () => {
  const msg: LLMMessage = {
    role: 'assistant',
    content: [
      { type: 'thinking', text: 'I should help' },
      { type: 'text', text: 'Here is the answer' },
      { type: 'tool_use', id: 't1', name: 'readFile', input: { path: '/foo' } },
    ],
  }
  const result = formatMessageForSummary(msg)
  expect(result).toMatch(/\[thinking\] I should help/)
  expect(result).toMatch(/Here is the answer/)
  expect(result).toMatch(/\[tool call\] readFile/)
})

// ─── estimateMessagesTokens tests ───

it('estimateMessagesTokens provides reasonable estimates', () => {
  const messages: LLMMessage[] = [
    { role: 'user', content: 'a'.repeat(400) }, // ~100 tokens
    { role: 'assistant', content: 'b'.repeat(800) }, // ~200 tokens
  ]
  const estimate = estimateMessagesTokens(messages)
  expect(estimate >= 200, `expected >= 200, got ${estimate}`).toBeTruthy()
  expect(estimate <= 400, `expected <= 400, got ${estimate}`).toBeTruthy()
})

// ─── End-to-end compaction flow test ───

it('end-to-end: compaction reduces blob count and token estimate', () => {
  // 20×5K tok ≈ 100K var-tokens > 44K 预算 → 真实压缩
  const entries = makeLargeHistoryEntries(20, { withSystem: true, withPreamble: true })
  const originalBlobIds = entries.map(e => e.blobId)
  const originalTokenEstimate = estimateMessagesTokens(entries.map(e => e.message))

  const plan = planCompaction(entries, { contextTokenLimit: 258_400 })
  expect(plan.summarizeEntries.length > 0, 'should have entries to summarize').toBeTruthy()

  const artifacts = createCompactionArtifacts({
    plan,
    summaryText: '- User discussed topics A, B, C.\n- Assistant provided solutions.',
    previousSummaryArchiveIds: [],
  })

  const compactedTokenEstimate = estimateMessagesTokens([
    ...plan.leading.map(e => e.message),
    { role: 'assistant', content: `Previous conversation summary:\n${artifacts.summaryText}` },
    ...plan.keepTail.map(e => e.message),
  ])

  expect(artifacts.nextRootBlobIds.length < originalBlobIds.length, `expected fewer blobs: ${artifacts.nextRootBlobIds.length} < ${originalBlobIds.length}`).toBeTruthy()
  expect(compactedTokenEstimate < originalTokenEstimate, `expected fewer tokens: ${compactedTokenEstimate} < ${originalTokenEstimate}`).toBeTruthy()

  // Verify compacted blobs can be hydrated
  const hydrated = hydrateHistoryEntries(artifacts.nextRootBlobIds)
  expect(hydrated.length > 0, 'compacted blobs should be hydratable').toBeTruthy()

  // Summary blob should be among them
  const summaryEntry = hydrated.find(e => isSummaryBlobMessage(e.raw))
  expect(summaryEntry, 'should have a summary blob in compacted history').toBeTruthy()
  expect(summaryEntry!.message.content as string).toMatch(/Previous conversation summary/)
})

it('end-to-end: shouldTriggerCompaction integrates with token estimation', () => {
  // Simulate a conversation that's approaching context limit
  const contextLimit = 8000
  const entries = makeHistoryEntries(30, { withSystem: true })
  const estimate = estimateMessagesTokens(entries.map(e => e.message))

  // With a 30-message conversation and ~125+ chars each, estimate should be substantial
  expect(estimate > 0, `token estimate should be positive: ${estimate}`).toBeTruthy()

  // Test with tight limit to simulate pressure
  const shouldCompact = shouldTriggerCompaction(estimate, contextLimit, 85)
  if (estimate >= contextLimit * 0.85) {
    expect(shouldCompact, 'should trigger compaction when over 85%').toBe(true)
  }
})

import { randomUUID } from 'crypto';
import type { AgentServerMessage } from '../../gen/agent_v1_pb';
import { workspaceUris, type ParsedRunRequest } from './protocol';
import type { AgentSession } from './session';
import { heartbeat, checkpoint, kvMessage, summary, summaryCompleted, summaryStarted } from './stream';
import { clampTokenDetails, computeContextUsagePercent } from './usage';
import { resolveProviderRuntime } from '../llm';
import { hydrateHistoryEntries, repairHistoryEntries } from './historyManager';
import { buildSummarySource, createCompactionArtifacts, measureMessagesTokens, planCompaction, resolveSummaryThinkingLevel, streamSummaryWithFallback } from './compactionStrategy';
import { releaseCompactionLock, tryAcquireCompactionLock, waitForCompactionLockRelease } from './compactionLock';
import { executePreCompactHook } from './hookRuntime';
import { persistConversationCheckpoint } from '../../database/checkpoints';
import { logger } from '../../logger';

export async function* handleSummarizeAction(
    parsed: ParsedRunRequest,
    session: AgentSession | null,
): AsyncIterable<AgentServerMessage> {
    const route = resolveProviderRuntime(parsed.modelId);
    // 并发互斥 (设计文档 §7#7): 等待 inline 压缩释放后再重新评估是否仍需压缩
    await waitForCompactionLockRelease(parsed.conversationId);
    if (!tryAcquireCompactionLock(parsed.conversationId))
        logger.warn({ conversationId: parsed.conversationId }, '[AUTOCOMPACT] summarizeAction lock contention — proceeding after wait');
    try {
        yield* handleSummarizeActionLocked(parsed, session, route);
    }
    finally {
        releaseCompactionLock(parsed.conversationId);
    }
}

async function* handleSummarizeActionLocked(
    parsed: ParsedRunRequest,
    session: AgentSession | null,
    route: ReturnType<typeof resolveProviderRuntime>,
): AsyncIterable<AgentServerMessage> {
    const hydratedHistoryEntries = hydrateHistoryEntries(parsed.historyBlobIds);
    const missingHistoryBlobs = Math.max(0, parsed.historyBlobIds.length - hydratedHistoryEntries.length);
    const historyEntries = repairHistoryEntries(hydratedHistoryEntries);
    const contextTokenLimit = parsed.historyTokenDetails?.maxTokens ?? parsed.contextTokenLimit ?? route.contextTokenLimit;
    const compactionPlan = planCompaction(historyEntries, { contextTokenLimit });
    const currentTokenDetails = clampTokenDetails(
        parsed.historyTokenDetails?.usedTokens ?? measureMessagesTokens(historyEntries.map(entry => entry.message)),
        contextTokenLimit,
    );
    const contextUsagePercent = computeContextUsagePercent(currentTokenDetails.usedTokens, currentTokenDetails.maxTokens);
    const generationId = randomUUID();

    logger.info({
        conversationId: parsed.conversationId,
        model: route.model,
        historyBlobIds: parsed.historyBlobIds.length,
        hydratedEntries: hydratedHistoryEntries.length,
        missingBlobs: missingHistoryBlobs,
        summarizeEntries: compactionPlan.summarizeEntries.length,
        keepTail: compactionPlan.keepTail.length,
        contextUsagePercent: contextUsagePercent.toFixed(1),
        usedTokens: currentTokenDetails.usedTokens,
        maxTokens: currentTokenDetails.maxTokens,
    }, '[SUMMARIZE] action started');

    yield heartbeat();

    const hookMessage = yield* executePreCompactHook({
        session,
        conversationId: parsed.conversationId,
        generationId,
        modelId: parsed.modelId,
        contextUsagePercent,
        contextTokens: currentTokenDetails.usedTokens,
        contextWindowSize: currentTokenDetails.maxTokens,
        messageCount: historyEntries.length,
        messagesToCompact: compactionPlan.summarizeEntries.length,
        isFirstCompaction: parsed.historySummaryArchiveIds.length === 0,
        execMessageId: 1,
    });

    yield summaryStarted();

    if (missingHistoryBlobs > 0) {
        logger.warn({
            conversationId: parsed.conversationId,
            requestedBlobs: parsed.historyBlobIds.length,
            resolvedBlobs: hydratedHistoryEntries.length,
            missingHistoryBlobs,
        }, '[AGENT] summarizeAction skipped due to incomplete history');

        logger.info({
            conversationId: parsed.conversationId,
            origin: 'client_summarize',
            kind: 'committed',
            usedTokens: currentTokenDetails.usedTokens,
            maxTokens: currentTokenDetails.maxTokens,
            rootBlobCount: parsed.historyBlobIds.length,
            summaryArchiveCount: parsed.historySummaryArchiveIds.length,
        }, '[AUTOCOMPACT] checkpoint write');
        persistConversationCheckpoint({
            kind: 'committed',
            conversationId: parsed.conversationId,
            rootBlobIds: parsed.historyBlobIds,
            turnBlobIds: parsed.historyTurnBlobIds,
            summaryArchiveIds: parsed.historySummaryArchiveIds,
            tokenDetails: currentTokenDetails,
            mode: parsed.mode,
            updatedAt: Date.now(),
        });

        yield checkpoint(
            parsed.historyBlobIds,
            currentTokenDetails.usedTokens,
            currentTokenDetails.maxTokens,
            parsed.mode,
            undefined,
            {
                turnBlobIds: parsed.historyTurnBlobIds,
                summaryArchiveIds: parsed.historySummaryArchiveIds,
                workspaceUris: workspaceUris(parsed),
                readPaths: parsed.readPaths,
                modelName: route.model,
                gitRepos: parsed.gitRepos?.map(r => ({ path: r.path, branchName: r.branchName })),
            },
        );
        yield summaryCompleted(hookMessage ?? 'Compaction deferred: conversation history is incomplete.');
        return;
    }

    if (compactionPlan.summarizeEntries.length === 0) {
        // F2: mode==='disabled' 时 plan 同样返回空 summarizeEntries, 但语义是
        // "压缩结构性不可行" (leading 过大/窗口过小), 不是"已经够紧凑" — 文案须区分
        if (compactionPlan.mode === 'disabled') {
            logger.warn({
                conversationId: parsed.conversationId,
                contextTokenLimit,
                leadingTokens: compactionPlan.diagnostics.leadingTokens,
            }, '[AUTOCOMPACT] summarizeAction skipped — compaction structurally infeasible for this window');
        }
        logger.info({
            conversationId: parsed.conversationId,
            origin: 'client_summarize',
            kind: 'committed',
            usedTokens: currentTokenDetails.usedTokens,
            maxTokens: currentTokenDetails.maxTokens,
            rootBlobCount: parsed.historyBlobIds.length,
            summaryArchiveCount: parsed.historySummaryArchiveIds.length,
        }, '[AUTOCOMPACT] checkpoint write');
        persistConversationCheckpoint({
            kind: 'committed',
            conversationId: parsed.conversationId,
            rootBlobIds: parsed.historyBlobIds,
            turnBlobIds: parsed.historyTurnBlobIds,
            summaryArchiveIds: parsed.historySummaryArchiveIds,
            tokenDetails: currentTokenDetails,
            mode: parsed.mode,
            updatedAt: Date.now(),
        });

        yield summaryCompleted(hookMessage ?? (compactionPlan.mode === 'disabled'
            ? 'Compaction unavailable: system prompt plus reserves exceed this model\'s usable context window. Consider a larger-context model.'
            : 'Conversation already compact enough.'));
        yield checkpoint(
            parsed.historyBlobIds,
            currentTokenDetails.usedTokens,
            currentTokenDetails.maxTokens,
            parsed.mode,
            undefined,
            {
                turnBlobIds: parsed.historyTurnBlobIds,
                summaryArchiveIds: parsed.historySummaryArchiveIds,
                workspaceUris: workspaceUris(parsed),
                readPaths: parsed.readPaths,
                modelName: route.model,
                gitRepos: parsed.gitRepos?.map(r => ({ path: r.path, branchName: r.branchName })),
            },
        );
        return;
    }

    // 摘要源构造 (阶段 4): 总预算 min(0.6×窗口×4, 3.2e6) chars, 超限走 max-min 水位分配
    const summarySourceText = buildSummarySource(compactionPlan.summarizeEntries, { contextTokenLimit });

    let summaryText = '';
    const llmStartTime = Date.now();
    logger.info({
        conversationId: parsed.conversationId,
        model: route.model,
        sourceTextLen: summarySourceText.length,
        summarizeEntries: compactionPlan.summarizeEntries.length,
        keepTail: compactionPlan.keepTail.length,
    }, '[SUMMARIZE] LLM summary starting');

    // 三级兜底 (流式, 与 inline 路径同一实现 — 两路行为一致)
    let lastHeartbeatTime = Date.now();
    for await (const summaryEvent of streamSummaryWithFallback({
        provider: route.provider,
        model: route.model,
        sourceText: summarySourceText,
        contextTokenLimit,
        thinkingLevel: resolveSummaryThinkingLevel(route),
    })) {
        if (summaryEvent.type === 'delta') {
            summaryText += summaryEvent.text;
            yield summary(summaryEvent.text);
        }
        // LLM 生成期间持续 yield heartbeat, 防止客户端 stall detector 误判
        if (Date.now() - lastHeartbeatTime >= 4000) {
            yield heartbeat();
            lastHeartbeatTime = Date.now();
        }
        if (summaryEvent.type === 'done')
            summaryText = summaryEvent.text;
    }

    logger.info({
        conversationId: parsed.conversationId,
        summaryLen: summaryText.length,
        durationMs: Date.now() - llmStartTime,
    }, '[SUMMARIZE] LLM summary done');

    const artifacts = createCompactionArtifacts({
        plan: compactionPlan,
        summaryText,
        previousSummaryArchiveIds: parsed.historySummaryArchiveIds,
    });

    yield kvMessage(1, artifacts.summaryBlobId, artifacts.summaryBlobData);
    for (const [index, archiveBlob] of artifacts.archiveBlobs.entries()) {
        yield kvMessage(2 + index, archiveBlob.blobId, archiveBlob.blobData, archiveBlob.blobDataRaw);
    }

    const compactedUsedTokens = clampTokenDetails(
        // o200k 实测重置 (与 inline 路径同口径, 两路行为一致由单一实现保证)
        measureMessagesTokens([
            ...compactionPlan.leading.map(entry => entry.message),
            { role: 'assistant', content: `Previous conversation summary:\n${artifacts.summaryText}` },
            ...compactionPlan.keepTail.map(entry => entry.message),
        ]),
        currentTokenDetails.maxTokens,
    );

    logger.info({
        conversationId: parsed.conversationId,
        origin: 'client_summarize',
        kind: 'committed',
        usedTokens: compactedUsedTokens.usedTokens,
        maxTokens: compactedUsedTokens.maxTokens,
        rootBlobCount: artifacts.nextRootBlobIds.length,
        summaryArchiveCount: artifacts.nextSummaryArchiveIds.length,
    }, '[AUTOCOMPACT] checkpoint write');
    persistConversationCheckpoint({
        kind: 'committed',
        conversationId: parsed.conversationId,
        rootBlobIds: artifacts.nextRootBlobIds,
        turnBlobIds: parsed.historyTurnBlobIds,
        summaryArchiveIds: artifacts.nextSummaryArchiveIds,
        tokenDetails: compactedUsedTokens,
        mode: parsed.mode,
        updatedAt: Date.now(),
    });

    yield checkpoint(
        artifacts.nextRootBlobIds,
        compactedUsedTokens.usedTokens,
        compactedUsedTokens.maxTokens,
        parsed.mode,
        undefined,
        {
            turnBlobIds: parsed.historyTurnBlobIds,
            summaryArchiveIds: artifacts.nextSummaryArchiveIds,
            workspaceUris: workspaceUris(parsed),
            readPaths: parsed.readPaths,
            modelName: route.model,
            gitRepos: parsed.gitRepos?.map(r => ({ path: r.path, branchName: r.branchName })),
        },
    );
    yield summaryCompleted(hookMessage ?? 'Chat context summarized.');
}

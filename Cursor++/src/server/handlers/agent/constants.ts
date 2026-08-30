// Compatibility and heuristic constants shared across the agent runtime.
//
// Categorization:
// - compatibility default: chosen to match stable Cursor client behavior and avoid idle stalls
// - heuristic: local compaction policy, not protocol-defined by Cursor
// - UX timeout: conservative defaults for interactive tool execution

// Keep the agent stream active under common ~5s idle UI/proxy thresholds.
// This is a compatibility/UX interval, not a Cursor protocol requirement.
export const AGENT_HEARTBEAT_INTERVAL_MS = 4_000;

// Foreground shell commands should finish promptly during an agent turn.
// Longer operations should explicitly opt into background-style execution.
export const SHELL_DEFAULT_TIMEOUT_MS = 30_000;

// Cursor shell exec args support a much larger hard timeout for long-running commands.
// Keep aligned across started/exec arg builders.
export const SHELL_HARD_TIMEOUT_MS = 86_400_000;

// Forwarded to Cursor shell exec args to avoid overly large inline file output payloads.
// Compatibility-oriented default in the current implementation.
export const SHELL_FILE_OUTPUT_THRESHOLD_BYTES = 40_000n;

// Cursor shell timeout behavior enum value.
//
// IMPORTANT: 对齐 3.0.16 proto 的 agent.v1.TimeoutBehavior enum:
//   UNSPECIFIED = 0
//   CANCEL = 1
//   BACKGROUND = 2   ← 这里
//
// Bug 历史: 此常量原为 1, 对应 CANCEL, 导致 Cursor 客户端把 shell tool call
// 误识别为 "cancelled" 状态, UI 触发 fallback 显示 "Command failed to generate".
// 修正为 2 (BACKGROUND) 之后, shell tool call 在 UI 里显示为正常的后台执行态.
// 详见 analysis/checkpoint-revert-protocol.md 的 Round D 记录.
export const SHELL_TIMEOUT_BEHAVIOR_BACKGROUND = 2;

// 空窗期 idle hint: 模型输出文本后长时间无新事件时,
// 注入 thinkingCompleted 让客户端从 streaming_text 转到 waiting_server_next,
// 下一个 heartbeat 将触发 "Generating response" 状态显示。
// 避免兼容性较差的模型提供商 (如 GLM 不流式 tool_use) 导致 UI 看起来卡死。
export const IDLE_HINT_AFTER_MS = 3_000;

// Heuristic compaction policy (legacy, 条数定额):
// preserve more recent turns uncompressed so continuation quality keeps short-term state.
// These values are local policy choices, not protocol-defined by Cursor.
//
// @deprecated 第二阶段已切换为 token 预算制 (设计文档 §4/§5),
// 保留仅作备选方案 (§3.3 灰度前置) 与回滚开关用, 勿在新代码引用。
export const COMPACTION_MEDIUM_BODY_THRESHOLD = 2;
export const COMPACTION_MEDIUM_BODY_KEEP_TAIL = 2;
export const COMPACTION_LONG_BODY_THRESHOLD = 8;
export const COMPACTION_LONG_BODY_KEEP_TAIL = 6;

// ═══════════════════════════════════════════════════════════════════
// 第二阶段: keepTail 预算化参数 (设计文档 §5 参数表, 唯一权威为该表)
// 公式集中于此, planCompaction / usage / 摘要侧共享, 禁止散落内联数字。
// ═══════════════════════════════════════════════════════════════════

/** 触发线 = 窗口 − min(该值, 15% × 窗口)。审计三修正: 双轨 min(max−40K, 0.85max) 在小窗死带, 改为单一预留式 */
export const AUTOCOMPACT_TRIGGER_RESERVE_MAX_TOKENS = 40_000;
export const AUTOCOMPACT_TRIGGER_RESERVE_RATIO = 0.15;

/** 压缩后地板目标 = 25% × 窗口 (258,400 窗 → 64,600) */
export const TARGET_FLOOR_RATIO = 0.25;

/** 摘要预留 = min(5K, 2% × 窗口) — 按窗口比例缩放 (审计四小窗修正) */
export const SUMMARY_RESERVE_MAX_TOKENS = 5_000;
export const SUMMARY_RESERVE_RATIO = 0.02;

/** keepTail 预算 clamp 下限 = min(8K, 5% × 窗口); 上限固定 60K (1M 窗 25%=250K 失去压缩意义) */
export const KEEP_TAIL_BUDGET_MIN_TOKENS = 8_000;
export const KEEP_TAIL_BUDGET_MIN_RATIO = 0.05;
export const KEEP_TAIL_BUDGET_MAX_TOKENS = 60_000;

/** 巨物线 = max(4K, min(12K, 25% × budget)); 超线的 tool_result/大字段参与占位计价 */
export const LARGE_ENTRY_MIN_TOKENS = 4_000;
export const LARGE_ENTRY_MAX_TOKENS = 12_000;
export const LARGE_ENTRY_BUDGET_RATIO = 0.25;

/** 摘要输出硬上界 = 2 × summaryReserve (审计四: 官方 prompt 第 6 节随会话年龄单调增长) */
export const SUMMARY_HARD_CAP_RESERVE_MULTIPLE = 2;

/** 占位预览 = 头 175 tok + 尾 75 tok (token 封顶, CJK 不击穿; 尾部信息密度高) */
export const PLACEHOLDER_PREVIEW_HEAD_TOKENS = 175;
export const PLACEHOLDER_PREVIEW_TAIL_TOKENS = 75;

/**
 * 图片计价 = 1,600 tok/块。
 * 校准清单项: 非跨 provider 普适常数 (审计二 Gemini 小图实测 ~258 tok/图),
 * 由 o200k vs provider usage 观测差校准; 仅用于预算计价, 不用于计费。
 */
export const IMAGE_BILLED_TOKENS = 1_600;

/** 违约判定 = 实占 > 承诺地板 × 1.2 (扣除因果前沿超额后) */
export const FLOOR_VIOLATION_RATIO = 1.2;

/** 预算安全边际: o200k 是校准估计器 (非 OpenAI 系偏差 10-15%), 计价乘 1.15 由观测校准 */
export const BUDGET_SAFETY_MARGIN = 1.15;

/** 可行性检查的输出预留 (对齐 usage.MAX_OUTPUT_RESERVE 量级, planCompaction 不感知 maxOutputTokens) */
export const FEASIBILITY_OUTPUT_RESERVE_TOKENS = 20_000;

/** 摘要源总预算 = min(0.6 × 窗口 × 4, 3.2e6) chars; min-quota 200 chars (官方 CC-012) */
export const SUMMARY_SOURCE_WINDOW_RATIO = 0.6;
export const SUMMARY_SOURCE_MAX_CHARS = 3_200_000;
export const SUMMARY_SOURCE_MIN_QUOTA_CHARS = 200;

/** 摘要三级兜底 (官方 CC-011 库参数全量接线, 官方生产只接线两级) */
export const SUMMARY_RETRY_MAX_ATTEMPTS = 3;
export const SUMMARY_RETRY_MIN_BUDGET_CHARS = 50_000;
export const SUMMARY_RETRY_MAX_INPUT_RATIO = 0.75;
/** 确定性降级预算 = clamp(窗口 × 2% × 4, 50K, 3.2e6) chars */
export const SUMMARY_FALLBACK_WINDOW_RATIO = 0.02;
export const SUMMARY_FALLBACK_MIN_CHARS = 50_000;
export const SUMMARY_FALLBACK_MAX_CHARS = 3_200_000;

/** 错误驱动压缩重试上限 (官方 5 轮, BYOK 单轮成本更高取保守值) */
export const CONTEXT_LENGTH_RETRY_MAX = 3;

/**
 * 摘要执行可靠性 (2026-08-29 实弹验证修正 F3/F4):
 * 慢/挂死网关下摘要调用曾无限期占锁 (~4 分钟无产出), 期间并发 run 反复
 * 撞锁跳过并误增熔断计数, 上下文静默膨胀 (120K 窗实测冲到 158K)。
 *
 * 每次尝试三重限时: 首事件 / 事件间停顿 / 总时长; 超时抛错驱动兜底梯子
 * (递减源重试 → 确定性降级), 保证压缩在有界时间内必然完成。
 * 死等静默网关的最坏路径: 30s + 20s + 20s ≈ 70s 到确定性降级。
 *
 * 对照 (2026-08-29 三方调研): Codex 全流统一 idle-only 300s (无总时长上限,
 * DEFAULT_STREAM_IDLE_TIMEOUT_MS) — 宽松是因为主回合也要容纳推理黑箱期;
 * Cursor 官方摘要请求客户端侧未见任何超时 (托管通道 + 服务端专用摘要模型池)。
 * 我们停顿限时敢收紧到 20s 的前提是 F6 已把推理期变为可见流 (thinking_delta
 * 心跳); 总时长取 180s 而非两家的"无上限": 锁持有时长必须有界 (F5 依赖),
 * 但要容纳健康慢流 — 180s × ~20 tok/s ≈ 3600 tok ≥ 典型摘要长度。
 */
export const SUMMARY_ATTEMPT_FIRST_EVENT_TIMEOUT_MS = 30_000;
export const SUMMARY_ATTEMPT_STALL_TIMEOUT_MS = 20_000;
export const SUMMARY_ATTEMPT_TOTAL_TIMEOUT_MS = 180_000;
export const SUMMARY_RETRY_FIRST_EVENT_TIMEOUT_MS = 20_000;
export const SUMMARY_RETRY_TOTAL_TIMEOUT_MS = 60_000;

/**
 * 摘要请求输出上限 = clamp(2 × SUMMARY_HARD_CAP, 4096, 16384)。
 * 不传时请求继承 provider 默认 (曾观测挂到主模型的 128K 配置), 推理型模型
 * 会把预算烧在 reasoning 上拖慢生成; 显式封顶让生成时长有界。
 * 上界 2×hardCap 而非 hardCap: reasoning token 计入 max_output_tokens 的
 * provider (openai-responses) 需要余量, 超出部分由 hard-cap 裁剪兜底。
 */
export const SUMMARY_MAX_OUTPUT_TOKENS_MIN = 4_096;
export const SUMMARY_MAX_OUTPUT_TOKENS_MAX = 16_384;

/** 错误驱动重试撞压缩锁时的最长等待 (等持锁压缩完成后自行重压) */
export const CONTEXT_RETRY_LOCK_WAIT_MAX_MS = 90_000;

/**
 * 摘要请求的推理档位 (F6, 2026-08-29 实弹诊断):
 * openai-responses 推理模型不带 reasoning 参数时按模型默认 effort 闷头思考,
 * 且黑箱期 API 不推送任何可映射事件 (reasoning summary 需 summary:'auto' 才有)
 * — 消费端与挂死网关不可区分, 实弹观测 4 分钟零事件。
 * 显式压到 low: 首字快出 + summary:'auto' 提供推理期心跳; 摘要任务不需要深推理。
 * 仅对 route.thinking===true 且 provider 为 openai-responses 的组合传递
 * (Claude 不带 thinking 参数即不思考, 已被本体会话压缩成功实证)。
 */
export const SUMMARY_THINKING_LEVEL = 'low' as const;

/** 入口截断 (阶段 1): ENTRY_CAP = min(25K tok, 25% × 窗口) */
export const TASK_ENTRY_CAP_RATIO = 0.25;

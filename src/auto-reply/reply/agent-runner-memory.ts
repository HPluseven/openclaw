import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import type { AgentMessage } from "@mariozechner/pi-agent-core";
import { resolveBootstrapWarningSignaturesSeen } from "../../agents/bootstrap-budget.js";
import { estimateMessagesTokens } from "../../agents/compaction.js";
import { runWithModelFallback } from "../../agents/model-fallback.js";
import { isCliProvider } from "../../agents/model-selection.js";
import { runEmbeddedPiAgent } from "../../agents/pi-embedded.js";
import { resolveSandboxConfigForAgent, resolveSandboxRuntimeStatus } from "../../agents/sandbox.js";
import {
  derivePromptTokens,
  hasNonzeroUsage,
  normalizeUsage,
  type UsageLike,
} from "../../agents/usage.js";
import type { OpenClawConfig } from "../../config/config.js";
import {
  resolveAgentIdFromSessionKey,
  resolveSessionFilePath,
  resolveSessionFilePathOptions,
  type SessionEntry,
  updateSessionStoreEntry,
} from "../../config/sessions.js";
import { logVerbose } from "../../globals.js";
import { registerAgentRunContext } from "../../infra/agent-events.js";
import { appendFileWithinRoot } from "../../infra/fs-safe.js";
import type { TemplateContext } from "../templating.js";
import type { VerboseLevel } from "../thinking.js";
import type { GetReplyOptions, ReplyPayload } from "../types.js";
import {
  buildEmbeddedRunExecutionParams,
  resolveModelFallbackOptions,
} from "./agent-runner-utils.js";
import {
  hasAlreadyFlushedForCurrentCompaction,
  MEMORY_FLUSH_APPEND_BLOCK_PREFIX,
  resolveMemoryFlushContextWindowTokens,
  resolveMemoryFlushRelativePathForRun,
  resolveMemoryFlushPromptForRun,
  resolveMemoryFlushSettings,
  shouldRunMemoryFlush,
} from "./memory-flush.js";
import { normalizeReplyPayload } from "./normalize-reply.js";
import { refreshQueuedFollowupSession, type FollowupRun } from "./queue.js";
import { incrementCompactionCount } from "./session-updates.js";

export function estimatePromptTokensForMemoryFlush(prompt?: string): number | undefined {
  const trimmed = prompt?.trim();
  if (!trimmed) {
    return undefined;
  }
  const message: AgentMessage = { role: "user", content: trimmed, timestamp: Date.now() };
  const tokens = estimateMessagesTokens([message]);
  if (!Number.isFinite(tokens) || tokens <= 0) {
    return undefined;
  }
  return Math.ceil(tokens);
}

export function resolveEffectivePromptTokens(
  basePromptTokens?: number,
  lastOutputTokens?: number,
  promptTokenEstimate?: number,
): number {
  const base = Math.max(0, basePromptTokens ?? 0);
  const output = Math.max(0, lastOutputTokens ?? 0);
  const estimate = Math.max(0, promptTokenEstimate ?? 0);
  // Flush gating projects the next input context by adding the previous
  // completion and the current user prompt estimate.
  return base + output + estimate;
}

export type SessionTranscriptUsageSnapshot = {
  promptTokens?: number;
  outputTokens?: number;
};

// Keep a generous near-threshold window so large assistant outputs still trigger
// transcript reads in time to flip memory-flush gating when needed.
const TRANSCRIPT_OUTPUT_READ_BUFFER_TOKENS = 8192;
const TRANSCRIPT_TAIL_CHUNK_BYTES = 64 * 1024;

type MemoryFlushFileSnapshot = {
  exists: boolean;
  size?: number;
  mtimeMs?: number;
};

async function getMemoryFlushFileSnapshot(params: {
  workspaceDir: string;
  relativePath: string;
}): Promise<MemoryFlushFileSnapshot> {
  try {
    const stat = await fs.promises.stat(path.join(params.workspaceDir, params.relativePath));
    return { exists: true, size: stat.size, mtimeMs: stat.mtimeMs };
  } catch {
    return { exists: false };
  }
}

function didMemoryFlushFileChange(
  before: MemoryFlushFileSnapshot,
  after: MemoryFlushFileSnapshot,
): boolean {
  if (before.exists !== after.exists) {
    return true;
  }
  if (!before.exists && !after.exists) {
    return false;
  }
  return before.size !== after.size || before.mtimeMs !== after.mtimeMs;
}

type MemoryFlushPayloadResolution =
  | { kind: "no_reply" }
  | { kind: "append_text"; text: string }
  | { kind: "invalid" };

function extractMemoryFlushAppendText(text: string): string | null {
  const normalized = normalizeReplyPayload({ text });
  const cleaned = normalized?.text?.trim() ?? "";
  if (!cleaned) {
    return null;
  }
  if (!cleaned.startsWith(MEMORY_FLUSH_APPEND_BLOCK_PREFIX)) {
    return null;
  }
  const body = cleaned.slice(MEMORY_FLUSH_APPEND_BLOCK_PREFIX.length).trim();
  return body || null;
}

function resolveMemoryFlushPayload(
  payloads: ReplyPayload[] | undefined,
): MemoryFlushPayloadResolution {
  if (!Array.isArray(payloads) || payloads.length === 0) {
    return { kind: "no_reply" };
  }

  if (payloads.some((payload) => payload?.isError)) {
    return { kind: "invalid" };
  }

  const visibleTexts = payloads
    .filter((payload) => !payload?.isReasoning)
    .map((payload) => payload?.text?.trim() ?? "")
    .filter((value) => value.length > 0);

  if (visibleTexts.length === 0) {
    return { kind: "no_reply" };
  }

  const appendTexts = visibleTexts
    .map((text) => extractMemoryFlushAppendText(text))
    .filter((value): value is string => Boolean(value));

  if (appendTexts.length === 0) {
    return { kind: "no_reply" };
  }

  return { kind: "append_text", text: appendTexts.join("\n\n") };
}

function parseUsageFromTranscriptLine(line: string): ReturnType<typeof normalizeUsage> | undefined {
  const trimmed = line.trim();
  if (!trimmed) {
    return undefined;
  }
  try {
    const parsed = JSON.parse(trimmed) as {
      message?: { usage?: UsageLike };
      usage?: UsageLike;
    };
    const usageRaw = parsed.message?.usage ?? parsed.usage;
    const usage = normalizeUsage(usageRaw);
    if (usage && hasNonzeroUsage(usage)) {
      return usage;
    }
  } catch {
    // ignore bad lines
  }
  return undefined;
}

function resolveSessionLogPath(
  sessionId?: string,
  sessionEntry?: SessionEntry,
  sessionKey?: string,
  opts?: { storePath?: string },
): string | undefined {
  if (!sessionId) {
    return undefined;
  }

  try {
    const transcriptPath = (
      sessionEntry as (SessionEntry & { transcriptPath?: string }) | undefined
    )?.transcriptPath?.trim();
    const sessionFile = sessionEntry?.sessionFile?.trim() || transcriptPath;
    const agentId = resolveAgentIdFromSessionKey(sessionKey);
    const pathOpts = resolveSessionFilePathOptions({
      agentId,
      storePath: opts?.storePath,
    });
    // Normalize sessionFile through resolveSessionFilePath so relative entries
    // are resolved against the sessions dir/store layout, not process.cwd().
    return resolveSessionFilePath(
      sessionId,
      sessionFile ? { sessionFile } : sessionEntry,
      pathOpts,
    );
  } catch {
    return undefined;
  }
}

function deriveTranscriptUsageSnapshot(
  usage: ReturnType<typeof normalizeUsage> | undefined,
): SessionTranscriptUsageSnapshot | undefined {
  if (!usage) {
    return undefined;
  }
  const promptTokens = derivePromptTokens(usage);
  const outputRaw = usage.output;
  const outputTokens =
    typeof outputRaw === "number" && Number.isFinite(outputRaw) && outputRaw > 0
      ? outputRaw
      : undefined;
  if (!(typeof promptTokens === "number") && !(typeof outputTokens === "number")) {
    return undefined;
  }
  return {
    promptTokens,
    outputTokens,
  };
}

type SessionLogSnapshot = {
  byteSize?: number;
  usage?: SessionTranscriptUsageSnapshot;
};

async function readSessionLogSnapshot(params: {
  sessionId?: string;
  sessionEntry?: SessionEntry;
  sessionKey?: string;
  opts?: { storePath?: string };
  includeByteSize: boolean;
  includeUsage: boolean;
}): Promise<SessionLogSnapshot> {
  const logPath = resolveSessionLogPath(
    params.sessionId,
    params.sessionEntry,
    params.sessionKey,
    params.opts,
  );
  if (!logPath) {
    return {};
  }

  const snapshot: SessionLogSnapshot = {};

  if (params.includeByteSize) {
    try {
      const stat = await fs.promises.stat(logPath);
      const size = Math.floor(stat.size);
      snapshot.byteSize = Number.isFinite(size) && size >= 0 ? size : undefined;
    } catch {
      snapshot.byteSize = undefined;
    }
  }

  if (params.includeUsage) {
    try {
      const lastUsage = await readLastNonzeroUsageFromSessionLog(logPath);
      snapshot.usage = deriveTranscriptUsageSnapshot(lastUsage);
    } catch {
      snapshot.usage = undefined;
    }
  }

  return snapshot;
}

async function readLastNonzeroUsageFromSessionLog(logPath: string) {
  const handle = await fs.promises.open(logPath, "r");
  try {
    const stat = await handle.stat();
    let position = stat.size;
    let leadingPartial = "";
    while (position > 0) {
      const chunkSize = Math.min(TRANSCRIPT_TAIL_CHUNK_BYTES, position);
      const start = position - chunkSize;
      const buffer = Buffer.allocUnsafe(chunkSize);
      const { bytesRead } = await handle.read(buffer, 0, chunkSize, start);
      if (bytesRead <= 0) {
        break;
      }
      const chunk = buffer.toString("utf-8", 0, bytesRead);
      const combined = `${chunk}${leadingPartial}`;
      const lines = combined.split(/\n+/);
      leadingPartial = lines.shift() ?? "";
      for (let i = lines.length - 1; i >= 0; i -= 1) {
        const usage = parseUsageFromTranscriptLine(lines[i] ?? "");
        if (usage) {
          return usage;
        }
      }
      position = start;
    }
    return parseUsageFromTranscriptLine(leadingPartial);
  } finally {
    await handle.close();
  }
}

export async function readPromptTokensFromSessionLog(
  sessionId?: string,
  sessionEntry?: SessionEntry,
  sessionKey?: string,
  opts?: { storePath?: string },
): Promise<SessionTranscriptUsageSnapshot | undefined> {
  const snapshot = await readSessionLogSnapshot({
    sessionId,
    sessionEntry,
    sessionKey,
    opts,
    includeByteSize: false,
    includeUsage: true,
  });
  return snapshot.usage;
}

export async function runMemoryFlushIfNeeded(params: {
  cfg: OpenClawConfig;
  followupRun: FollowupRun;
  promptForEstimate?: string;
  sessionCtx: TemplateContext;
  opts?: GetReplyOptions;
  defaultModel: string;
  agentCfgContextTokens?: number;
  resolvedVerboseLevel: VerboseLevel;
  sessionEntry?: SessionEntry;
  sessionStore?: Record<string, SessionEntry>;
  sessionKey?: string;
  storePath?: string;
  isHeartbeat: boolean;
}): Promise<SessionEntry | undefined> {
  const memoryFlushSettings = resolveMemoryFlushSettings(params.cfg);
  if (!memoryFlushSettings) {
    return params.sessionEntry;
  }

  const memoryFlushWritable = (() => {
    if (!params.sessionKey) {
      return true;
    }
    const runtime = resolveSandboxRuntimeStatus({
      cfg: params.cfg,
      sessionKey: params.sessionKey,
    });
    if (!runtime.sandboxed) {
      return true;
    }
    const sandboxCfg = resolveSandboxConfigForAgent(params.cfg, runtime.agentId);
    return sandboxCfg.workspaceAccess === "rw";
  })();

  const isCli = isCliProvider(params.followupRun.run.provider, params.cfg);
  const canAttemptFlush = memoryFlushWritable && !params.isHeartbeat && !isCli;
  let entry =
    params.sessionEntry ??
    (params.sessionKey ? params.sessionStore?.[params.sessionKey] : undefined);
  const contextWindowTokens = resolveMemoryFlushContextWindowTokens({
    modelId: params.followupRun.run.model ?? params.defaultModel,
    agentCfgContextTokens: params.agentCfgContextTokens,
  });

  const promptTokenEstimate = estimatePromptTokensForMemoryFlush(
    params.promptForEstimate ?? params.followupRun.prompt,
  );
  const persistedPromptTokensRaw = entry?.totalTokens;
  const persistedPromptTokens =
    typeof persistedPromptTokensRaw === "number" &&
    Number.isFinite(persistedPromptTokensRaw) &&
    persistedPromptTokensRaw > 0
      ? persistedPromptTokensRaw
      : undefined;
  const hasFreshPersistedPromptTokens =
    typeof persistedPromptTokens === "number" && entry?.totalTokensFresh === true;

  const flushThreshold =
    contextWindowTokens -
    memoryFlushSettings.reserveTokensFloor -
    memoryFlushSettings.softThresholdTokens;

  // When totals are stale/unknown, derive prompt + last output from transcript so memory
  // flush can still be evaluated against projected next-input size.
  //
  // When totals are fresh, only read the transcript when we're close enough to the
  // threshold that missing the last output tokens could flip the decision.
  const shouldReadTranscriptForOutput =
    canAttemptFlush &&
    entry &&
    hasFreshPersistedPromptTokens &&
    typeof promptTokenEstimate === "number" &&
    Number.isFinite(promptTokenEstimate) &&
    flushThreshold > 0 &&
    (persistedPromptTokens ?? 0) + promptTokenEstimate >=
      flushThreshold - TRANSCRIPT_OUTPUT_READ_BUFFER_TOKENS;

  const shouldReadTranscript = Boolean(
    canAttemptFlush && entry && (!hasFreshPersistedPromptTokens || shouldReadTranscriptForOutput),
  );

  const forceFlushTranscriptBytes = memoryFlushSettings.forceFlushTranscriptBytes;
  const shouldCheckTranscriptSizeForForcedFlush = Boolean(
    canAttemptFlush &&
    entry &&
    Number.isFinite(forceFlushTranscriptBytes) &&
    forceFlushTranscriptBytes > 0,
  );
  const shouldReadSessionLog = shouldReadTranscript || shouldCheckTranscriptSizeForForcedFlush;
  const sessionLogSnapshot = shouldReadSessionLog
    ? await readSessionLogSnapshot({
        sessionId: params.followupRun.run.sessionId,
        sessionEntry: entry,
        sessionKey: params.sessionKey ?? params.followupRun.run.sessionKey,
        opts: { storePath: params.storePath },
        includeByteSize: shouldCheckTranscriptSizeForForcedFlush,
        includeUsage: shouldReadTranscript,
      })
    : undefined;
  const transcriptByteSize = sessionLogSnapshot?.byteSize;
  const shouldForceFlushByTranscriptSize =
    typeof transcriptByteSize === "number" && transcriptByteSize >= forceFlushTranscriptBytes;

  const transcriptUsageSnapshot = sessionLogSnapshot?.usage;
  const transcriptPromptTokens = transcriptUsageSnapshot?.promptTokens;
  const transcriptOutputTokens = transcriptUsageSnapshot?.outputTokens;
  const hasReliableTranscriptPromptTokens =
    typeof transcriptPromptTokens === "number" &&
    Number.isFinite(transcriptPromptTokens) &&
    transcriptPromptTokens > 0;
  const shouldPersistTranscriptPromptTokens =
    hasReliableTranscriptPromptTokens &&
    (!hasFreshPersistedPromptTokens ||
      (transcriptPromptTokens ?? 0) > (persistedPromptTokens ?? 0));

  if (entry && shouldPersistTranscriptPromptTokens) {
    const nextEntry = {
      ...entry,
      totalTokens: transcriptPromptTokens,
      totalTokensFresh: true,
    };
    entry = nextEntry;
    if (params.sessionKey && params.sessionStore) {
      params.sessionStore[params.sessionKey] = nextEntry;
    }
    if (params.storePath && params.sessionKey) {
      try {
        const updatedEntry = await updateSessionStoreEntry({
          storePath: params.storePath,
          sessionKey: params.sessionKey,
          update: async () => ({ totalTokens: transcriptPromptTokens, totalTokensFresh: true }),
        });
        if (updatedEntry) {
          entry = updatedEntry;
          if (params.sessionStore) {
            params.sessionStore[params.sessionKey] = updatedEntry;
          }
        }
      } catch (err) {
        logVerbose(`failed to persist derived prompt totalTokens: ${String(err)}`);
      }
    }
  }

  const promptTokensSnapshot = Math.max(
    hasFreshPersistedPromptTokens ? (persistedPromptTokens ?? 0) : 0,
    hasReliableTranscriptPromptTokens ? (transcriptPromptTokens ?? 0) : 0,
  );
  const hasFreshPromptTokensSnapshot =
    promptTokensSnapshot > 0 &&
    (hasFreshPersistedPromptTokens || hasReliableTranscriptPromptTokens);

  const projectedTokenCount = hasFreshPromptTokensSnapshot
    ? resolveEffectivePromptTokens(
        promptTokensSnapshot,
        transcriptOutputTokens,
        promptTokenEstimate,
      )
    : undefined;
  const tokenCountForFlush =
    typeof projectedTokenCount === "number" &&
    Number.isFinite(projectedTokenCount) &&
    projectedTokenCount > 0
      ? projectedTokenCount
      : undefined;

  // Diagnostic logging to understand why memory flush may not trigger.
  logVerbose(
    `memoryFlush check: sessionKey=${params.sessionKey} ` +
      `tokenCount=${tokenCountForFlush ?? "undefined"} ` +
      `contextWindow=${contextWindowTokens} threshold=${flushThreshold} ` +
      `isHeartbeat=${params.isHeartbeat} isCli=${isCli} memoryFlushWritable=${memoryFlushWritable} ` +
      `compactionCount=${entry?.compactionCount ?? 0} memoryFlushCompactionCount=${entry?.memoryFlushCompactionCount ?? "undefined"} ` +
      `persistedPromptTokens=${persistedPromptTokens ?? "undefined"} persistedFresh=${entry?.totalTokensFresh === true} ` +
      `promptTokensEst=${promptTokenEstimate ?? "undefined"} transcriptPromptTokens=${transcriptPromptTokens ?? "undefined"} transcriptOutputTokens=${transcriptOutputTokens ?? "undefined"} ` +
      `projectedTokenCount=${projectedTokenCount ?? "undefined"} transcriptBytes=${transcriptByteSize ?? "undefined"} ` +
      `forceFlushTranscriptBytes=${forceFlushTranscriptBytes} forceFlushByTranscriptSize=${shouldForceFlushByTranscriptSize}`,
  );

  const shouldFlushMemory =
    (memoryFlushSettings &&
      memoryFlushWritable &&
      !params.isHeartbeat &&
      !isCli &&
      shouldRunMemoryFlush({
        entry,
        tokenCount: tokenCountForFlush,
        contextWindowTokens,
        reserveTokensFloor: memoryFlushSettings.reserveTokensFloor,
        softThresholdTokens: memoryFlushSettings.softThresholdTokens,
      })) ||
    (shouldForceFlushByTranscriptSize &&
      entry != null &&
      !hasAlreadyFlushedForCurrentCompaction(entry));

  if (!shouldFlushMemory) {
    return entry ?? params.sessionEntry;
  }

  logVerbose(
    `memoryFlush triggered: sessionKey=${params.sessionKey} tokenCount=${tokenCountForFlush ?? "undefined"} threshold=${flushThreshold}`,
  );

  let activeSessionEntry = entry ?? params.sessionEntry;
  const activeSessionStore = params.sessionStore;
  let bootstrapPromptWarningSignaturesSeen = resolveBootstrapWarningSignaturesSeen(
    activeSessionEntry?.systemPromptReport ??
      (params.sessionKey ? activeSessionStore?.[params.sessionKey]?.systemPromptReport : undefined),
  );
  const flushRunId = crypto.randomUUID();
  if (params.sessionKey) {
    registerAgentRunContext(flushRunId, {
      sessionKey: params.sessionKey,
      verboseLevel: params.resolvedVerboseLevel,
    });
  }
  let memoryCompactionCompleted = false;
  const memoryFlushNowMs = Date.now();
  const memoryFlushWritePath = resolveMemoryFlushRelativePathForRun({
    cfg: params.cfg,
    nowMs: memoryFlushNowMs,
  });
  const flushSystemPrompt = [
    params.followupRun.run.extraSystemPrompt,
    memoryFlushSettings.systemPrompt,
  ]
    .filter(Boolean)
    .join("\n\n");
  let postCompactionSessionId: string | undefined;
  try {
    let memoryFlushCompleted = false;
    await runWithModelFallback({
      ...resolveModelFallbackOptions(params.followupRun.run),
      runId: flushRunId,
      run: async (provider, model, runOptions) => {
        const flushRun = {
          ...params.followupRun.run,
          verboseLevel: "off" as const,
          reasoningLevel: "off" as const,
        };
        const { embeddedContext, senderContext, runBaseParams } = buildEmbeddedRunExecutionParams({
          run: flushRun,
          sessionCtx: params.sessionCtx,
          hasRepliedRef: params.opts?.hasRepliedRef,
          provider,
          model,
          runId: flushRunId,
          allowTransientCooldownProbe: runOptions?.allowTransientCooldownProbe,
        });
        const beforeSnapshot = await getMemoryFlushFileSnapshot({
          workspaceDir: params.followupRun.run.workspaceDir,
          relativePath: memoryFlushWritePath,
        });
        const result = await runEmbeddedPiAgent({
          ...embeddedContext,
          ...senderContext,
          ...runBaseParams,
          allowGatewaySubagentBinding: true,
          trigger: "memory",
          memoryFlushWritePath,
          prompt: resolveMemoryFlushPromptForRun({
            prompt: memoryFlushSettings.prompt,
            cfg: params.cfg,
            nowMs: memoryFlushNowMs,
          }),
          extraSystemPrompt: flushSystemPrompt,
          bootstrapPromptWarningSignaturesSeen,
          bootstrapPromptWarningSignature:
            bootstrapPromptWarningSignaturesSeen[bootstrapPromptWarningSignaturesSeen.length - 1],
          onAgentEvent: (evt) => {
            if (evt.stream === "compaction") {
              const phase = typeof evt.data.phase === "string" ? evt.data.phase : "";
              if (phase === "end") {
                memoryCompactionCompleted = true;
              }
            }
          },
        });
        const afterSnapshot = await getMemoryFlushFileSnapshot({
          workspaceDir: params.followupRun.run.workspaceDir,
          relativePath: memoryFlushWritePath,
        });
        const persistedViaLegacyWrite = didMemoryFlushFileChange(beforeSnapshot, afterSnapshot);
        const payload = resolveMemoryFlushPayload(result.payloads);
        if (persistedViaLegacyWrite) {
          memoryFlushCompleted = true;
        } else if (payload.kind === "no_reply") {
          memoryFlushCompleted = true;
        } else if (payload.kind === "append_text") {
          try {
            await appendFileWithinRoot({
              rootDir: params.followupRun.run.workspaceDir,
              relativePath: memoryFlushWritePath,
              data: payload.text,
              mkdir: true,
              prependNewlineIfNeeded: true,
            });
            memoryFlushCompleted = true;
          } catch (err) {
            logVerbose(`memory flush append skipped after model output: ${String(err)}`);
          }
        } else {
          logVerbose("memory flush produced no appendable payload; leaving flush state retryable");
        }
        if (result.meta?.agentMeta?.sessionId) {
          postCompactionSessionId = result.meta.agentMeta.sessionId;
        }
        bootstrapPromptWarningSignaturesSeen = resolveBootstrapWarningSignaturesSeen(
          result.meta?.systemPromptReport,
        );
        return result;
      },
    });
    let memoryFlushCompactionCount =
      activeSessionEntry?.compactionCount ??
      (params.sessionKey ? activeSessionStore?.[params.sessionKey]?.compactionCount : 0) ??
      0;
    if (memoryCompactionCompleted) {
      const previousSessionId = activeSessionEntry?.sessionId ?? params.followupRun.run.sessionId;
      const nextCount = await incrementCompactionCount({
        sessionEntry: activeSessionEntry,
        sessionStore: activeSessionStore,
        sessionKey: params.sessionKey,
        storePath: params.storePath,
        newSessionId: postCompactionSessionId,
      });
      const updatedEntry = params.sessionKey ? activeSessionStore?.[params.sessionKey] : undefined;
      if (updatedEntry) {
        activeSessionEntry = updatedEntry;
        params.followupRun.run.sessionId = updatedEntry.sessionId;
        if (updatedEntry.sessionFile) {
          params.followupRun.run.sessionFile = updatedEntry.sessionFile;
        }
        const queueKey = params.followupRun.run.sessionKey ?? params.sessionKey;
        if (queueKey) {
          refreshQueuedFollowupSession({
            key: queueKey,
            previousSessionId,
            nextSessionId: updatedEntry.sessionId,
            nextSessionFile: updatedEntry.sessionFile,
          });
        }
      }
      if (typeof nextCount === "number") {
        memoryFlushCompactionCount = nextCount;
      }
    }
    if (memoryFlushCompleted && params.storePath && params.sessionKey) {
      try {
        const updatedEntry = await updateSessionStoreEntry({
          storePath: params.storePath,
          sessionKey: params.sessionKey,
          update: async () => ({
            memoryFlushAt: Date.now(),
            memoryFlushCompactionCount,
          }),
        });
        if (updatedEntry) {
          activeSessionEntry = updatedEntry;
          params.followupRun.run.sessionId = updatedEntry.sessionId;
          if (updatedEntry.sessionFile) {
            params.followupRun.run.sessionFile = updatedEntry.sessionFile;
          }
        }
      } catch (err) {
        logVerbose(`failed to persist memory flush metadata: ${String(err)}`);
      }
    }
  } catch (err) {
    logVerbose(`memory flush run failed: ${String(err)}`);
  }

  return activeSessionEntry;
}

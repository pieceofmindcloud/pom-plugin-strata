import { useCallback, useEffect, useRef, useState } from "react";
import { rpc, ui, usePluginI18n } from "../host/runtime";

export type State = "idle" | "installing" | "preparing" | "loading" | "ready" | "stopping" | "error";

export type StrataStatus = {
  state: State;
  detail: string | null;
  model: string | null;
  modelPath: string | null;
  gpus: number[] | null;
  context: number | null;
  busy: boolean;
  runtime: { ready: boolean; version: string | null; error: string | null };
  log: string[] | null;
};

const STATES: State[] = ["idle", "installing", "preparing", "loading", "ready", "stopping", "error"];
const TONES = {
  idle: "neutral",
  installing: "sky",
  preparing: "sky",
  loading: "sky",
  ready: "aqua",
  stopping: "amber",
  error: "rose",
} as const;

/** Anything that is not the status contract is treated as "no answer". */
export function normalizeStatus(value: unknown): StrataStatus | null {
  if (typeof value !== "object" || value === null) return null;
  const raw = value as Record<string, unknown>;
  if (!STATES.includes(raw.state as State)) return null;
  const text = (field: unknown) => (typeof field === "string" && field ? field : null);
  const runtime = (raw.runtime ?? {}) as Record<string, unknown>;
  return {
    state: raw.state as State,
    detail: text(raw.detail),
    model: text(raw.model),
    modelPath: text(raw.model_path),
    gpus: Array.isArray(raw.gpus) ? raw.gpus.filter((gpu): gpu is number => typeof gpu === "number") : null,
    context: typeof raw.n_ctx === "number" ? raw.n_ctx : null,
    busy: raw.busy === true,
    runtime: { ready: runtime.ready === true, version: text(runtime.version), error: text(runtime.error) },
    log: Array.isArray(raw.log) ? raw.log.filter((line): line is string => typeof line === "string") : null,
  };
}

export function useStatus() {
  const [status, setStatus] = useState<StrataStatus | null>(null);
  const [failed, setFailed] = useState(false);

  const refresh = useCallback(async () => {
    try {
      const next = normalizeStatus(await rpc<unknown>("status"));
      setStatus(next);
      setFailed(next === null);
    } catch {
      setFailed(true);
    }
  }, []);

  useEffect(() => {
    void refresh();
    const timer = window.setInterval(() => void refresh(), 2000);
    return () => window.clearInterval(timer);
  }, [refresh]);

  return { status, failed };
}

/** The engine's state, model, cards and log, from one `useStatus` poll. */
export function StatusView({ status, failed }: { status: StrataStatus | null; failed: boolean }) {
  const { t } = usePluginI18n();
  const { PageHeader, Panel, PanelHeader, Badge, KeyValue } = ui();

  return (
    <div className="pb-page pb-page-pad pb-status">
      <PageHeader title={t("title")} subtitle={t("subtitle")} />

      {failed && !status && <p className="pb-note pb-note-error" role="alert">{t("unreachable")}</p>}

      {status && (
        <>
          <Panel className="pb-panel">
            <PanelHeader
              title={t("engine")}
              actions={<Badge tone={TONES[status.state]}>{t(`state.${status.state}`)}</Badge>}
            />
            <div className="pb-body">
              {status.state === "error" && status.detail && (
                <p className="pb-note pb-note-error" role="alert">{status.detail}</p>
              )}
              {status.state !== "error" && status.detail && <p className="pb-note" role="status">{status.detail}</p>}
              {status.state === "idle" && <p className="pb-note">{t("idleHint")}</p>}
              {!status.runtime.ready && status.runtime.error && (
                <p className="pb-note pb-note-error" role="alert">{status.runtime.error}</p>
              )}
              <dl className="pb-facts">
                <KeyValue label={t("model")}>
                  <span className="pb-mono">{status.model ?? t("none")}</span>
                </KeyValue>
                <KeyValue label={t("gpus")}>
                  <span className="pb-mono">
                    {status.gpus && status.gpus.length > 0 ? status.gpus.join(", ") : status.model ? "0" : t("none")}
                  </span>
                </KeyValue>
                <KeyValue label={t("context")}>
                  <span className="pb-mono">{status.context ? status.context.toLocaleString() : t("none")}</span>
                </KeyValue>
                <KeyValue label={t("activity")}>
                  <span className="pb-mono">{status.state === "ready" ? t(status.busy ? "busy" : "free") : t("none")}</span>
                </KeyValue>
                <KeyValue label={t("runtime")}>
                  <span className="pb-mono">{status.runtime.version ?? t("none")}</span>
                </KeyValue>
                <KeyValue label={t("modelPath")}>
                  <span className="pb-mono">{status.modelPath ?? t("none")}</span>
                </KeyValue>
              </dl>
            </div>
          </Panel>

          {status.log && <LogTerminal lines={status.log} />}
        </>
      )}
    </div>
  );
}

/** Copies text with the async API, which a page served over plain HTTP does not have, falling back to a hidden
 * textarea and `execCommand("copy")` (as pom-plugin-logs). */
async function copyText(text: string): Promise<boolean> {
  try {
    if (navigator.clipboard?.writeText) {
      await navigator.clipboard.writeText(text);
      return true;
    }
    const helper = document.createElement("textarea");
    helper.value = text;
    helper.style.position = "fixed";
    helper.style.opacity = "0";
    document.body.appendChild(helper);
    helper.select();
    const ok = document.execCommand("copy");
    helper.remove();
    return ok;
  } catch {
    return false;
  }
}

/** Within this many pixels of the end the log counts as followed (the POM's log console uses the same). */
const NEAR_BOTTOM = 32;
/** Strata's own tag at the start of a line: `[pom]`, `[strata]`, `strata serve:`. */
const TAG = /^(\[[a-z]+\]|strata [a-z]+:)/;

/** Strata's log as a terminal filling the rest of the screen, as the POM's deployment log console: it follows
 * the end while it is at the end, stops when the reader scrolls up, and copies what it shows. */
function LogTerminal({ lines }: { lines: string[] }) {
  const { t } = usePluginI18n();
  const { Panel, PanelHeader, Badge } = ui();
  const terminal = useRef<HTMLDivElement>(null);
  const atBottom = useRef(true);
  const [follow, setFollow] = useState(true);
  const [copied, setCopied] = useState<"idle" | "copied" | "failed">("idle");
  const last = lines[lines.length - 1] ?? "";

  useEffect(() => {
    const box = terminal.current;
    if (box && follow && atBottom.current) box.scrollTo({ top: box.scrollHeight, behavior: "auto" });
  }, [follow, lines.length, last]);

  const onScroll = () => {
    const box = terminal.current;
    if (!box) return;
    atBottom.current = box.scrollHeight - (box.scrollTop + box.clientHeight) <= NEAR_BOTTOM;
    if (!atBottom.current) setFollow(false);
  };
  const toggle = () => {
    const next = !follow;
    setFollow(next);
    if (next) {
      atBottom.current = true;
      terminal.current?.scrollTo({ top: terminal.current.scrollHeight, behavior: "auto" });
    }
  };
  const copy = async () => setCopied((await copyText(lines.join("\n"))) ? "copied" : "failed");

  return (
    <Panel className="pb-panel pb-log-panel">
      <PanelHeader title={t("log")} actions={<Badge tone="aqua">{t("live")}</Badge>} />
      <div className="pb-term-wrap">
        <div className="pb-term-actions">
          <button
            type="button"
            className="pb-term-button"
            aria-pressed={follow}
            title={t(follow ? "autoScrollDisable" : "autoScrollEnable")}
            onClick={toggle}
          >
            <span aria-hidden="true">⤓</span>
            {t(follow ? "autoScrollOn" : "autoScrollOff")}
          </button>
          <button type="button" className="pb-term-button" onClick={() => void copy()}>
            <span aria-hidden="true">{copied === "copied" ? "✓" : "⧉"}</span>
            {t(copied === "copied" ? "logCopied" : copied === "failed" ? "logCopyFailed" : "copyLog")}
          </button>
        </div>
        <div ref={terminal} className="pb-term" role="log" aria-live="polite" onScroll={onScroll}>
          {lines.length === 0 && <div className="pb-term-line pb-term-empty">{t("none")}</div>}
          {lines.map((line, index) => {
            const tag = TAG.exec(line)?.[0];
            return (
              <div key={index} className="pb-term-line">
                {tag && <span className="pb-term-tag">{tag}</span>}
                {tag ? line.slice(tag.length) : line}
              </div>
            );
          })}
        </div>
      </div>
    </Panel>
  );
}

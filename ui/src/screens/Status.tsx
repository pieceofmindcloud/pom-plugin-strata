import { useCallback, useEffect, useState } from "react";
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

export function Status() {
  const { t } = usePluginI18n();
  const { PageHeader, Panel, PanelHeader, Badge, KeyValue } = ui();
  const { status, failed } = useStatus();

  return (
    <div className="pb-page">
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

          {status.log && (
            <Panel className="pb-panel">
              <PanelHeader title={t("log")} />
              <pre className="pb-log">{status.log.length > 0 ? status.log.join("\n") : t("none")}</pre>
            </Panel>
          )}
        </>
      )}
    </div>
  );
}

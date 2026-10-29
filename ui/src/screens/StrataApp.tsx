import { useState } from "react";
import { usePluginI18n } from "../host/runtime";
import { useScreenReady } from "../host/ready";
import { StatusView, useStatus } from "./Status";

declare const __POM_PLUGIN_CODE__: string;

/** Strata's own web app (chat, live GPU, speed and cache monitor), served by
 * the plugin through the POM plugin proxy. */
const APP = `/api/ui/plugins/${encodeURIComponent(__POM_PLUGIN_CODE__)}/proxy/`;

type Tab = "app" | "strata" | "status";
const STORAGE_KEY = "pom-strata.tab";
const ICONS: Record<Tab, string> = { app: "✦", strata: "◆", status: "≡" };

function savedTab(): Tab | null {
  try {
    const saved = localStorage.getItem(STORAGE_KEY);
    return saved === "app" || saved === "strata" || saved === "status" ? saved : null;
  } catch {
    return null;
  }
}

/** The single Strata screen: Strata's app (chats, tasks and the agent, whose
 * requests go through the POM like any client's) while it runs, Strata's own
 * web page while this machine serves a model, and the engine's status. With
 * neither, it is the status alone. */
export function StrataApp() {
  const { t } = usePluginI18n();
  const { status, failed } = useStatus();
  const [chosen, setTab] = useState<Tab | null>(savedTab);
  const choose = (next: Tab) => {
    setTab(next);
    try {
      localStorage.setItem(STORAGE_KEY, next);
    } catch {
      // Remembering the tab is a convenience only.
    }
  };

  // The engine's status is what this screen is for: the host opens it when the
  // first poll answered (or failed), not over the empty page before it.
  useScreenReady(status !== null || failed);

  // Strata's web page runs on the machine that serves the requests; an earlier
  // stage of a split model has only the engine. The app needs no local engine.
  const serving = status?.state === "ready" && !status.earlierStage;
  const appReady = status?.app?.state === "ready";
  const tabs: Tab[] = [...(appReady ? (["app"] as Tab[]) : []), ...(serving ? (["strata"] as Tab[]) : []), "status"];
  const tab: Tab = chosen && tabs.includes(chosen) ? chosen : tabs[0];
  if (tabs.length === 1) {
    return (
      <div className="pb-scroll">
        <StatusView status={status} failed={failed} />
      </div>
    );
  }
  return (
    <div className="pb-shell">
      <div className="pb-shell-tabs" role="tablist" aria-label="Strata">
        {tabs.map((value) => (
          <button
            key={value}
            type="button"
            role="tab"
            id={`pb-tab-${value}`}
            aria-selected={tab === value}
            aria-controls="pb-shell-panel"
            className={tab === value ? "pb-shell-tab pb-shell-tab-on" : "pb-shell-tab"}
            onClick={() => choose(value)}
          >
            <span aria-hidden="true">{ICONS[value]}</span>
            {t(`tab.${value}`)}
          </button>
        ))}
      </div>
      <div className="pb-shell-panel" id="pb-shell-panel" role="tabpanel" aria-labelledby={`pb-tab-${tab}`}>
        {/* The app stays mounted while the model is served: switching tabs keeps its page and its chat. */}
        {appReady && <iframe className="pb-frame" title={t("appTitle")} src={`${APP}app/`} hidden={tab !== "app"} />}
        {serving && <iframe className="pb-frame" title={t("monitorTitle")} src={APP} hidden={tab !== "strata"} />}
        {tab === "status" && (
          <div className="pb-scroll">
            <StatusView status={status} failed={failed} />
          </div>
        )}
      </div>
    </div>
  );
}

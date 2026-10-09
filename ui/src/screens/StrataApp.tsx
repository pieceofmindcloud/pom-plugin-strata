import { useState } from "react";
import { usePluginI18n } from "../host/runtime";
import { StatusView, useStatus } from "./Status";

declare const __POM_PLUGIN_CODE__: string;

/** Strata's own web app (chat, live GPU, speed and cache monitor), served by
 * the plugin through the POM plugin proxy. */
const APP = `/api/ui/plugins/${encodeURIComponent(__POM_PLUGIN_CODE__)}/proxy/`;

type Tab = "strata" | "status";
const TABS: Tab[] = ["strata", "status"];
const STORAGE_KEY = "pom-strata.tab";

function initialTab(): Tab {
  try {
    return localStorage.getItem(STORAGE_KEY) === "status" ? "status" : "strata";
  } catch {
    return "strata";
  }
}

/** The single Strata screen. With no model served, or on an earlier stage of a
 * split model, it is the engine's status; while this machine serves a model,
 * Strata's own app and the status are tabs. */
export function StrataApp() {
  const { t } = usePluginI18n();
  const { status, failed } = useStatus();
  const [tab, setTab] = useState<Tab>(initialTab);
  const choose = (next: Tab) => {
    setTab(next);
    try {
      localStorage.setItem(STORAGE_KEY, next);
    } catch {
      // Remembering the tab is a convenience only.
    }
  };

  // Strata's app runs on the machine that serves the requests; an earlier
  // stage of a split model has only the engine, so it shows the status alone.
  if (status?.state !== "ready" || status.earlierStage) {
    return (
      <div className="pb-scroll">
        <StatusView status={status} failed={failed} />
      </div>
    );
  }
  return (
    <div className="pb-shell">
      <div className="pb-shell-tabs" role="tablist" aria-label="Strata">
        {TABS.map((value) => (
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
            <span aria-hidden="true">{value === "strata" ? "◆" : "≡"}</span>
            {t(`tab.${value}`)}
          </button>
        ))}
      </div>
      <div className="pb-shell-panel" id="pb-shell-panel" role="tabpanel" aria-labelledby={`pb-tab-${tab}`}>
        {/* The app stays mounted while the model is served: switching tabs keeps its page and its chat. */}
        <iframe className="pb-frame" title={t("monitorTitle")} src={APP} hidden={tab !== "strata"} />
        {tab === "status" && (
          <div className="pb-scroll">
            <StatusView status={status} failed={failed} />
          </div>
        )}
      </div>
    </div>
  );
}

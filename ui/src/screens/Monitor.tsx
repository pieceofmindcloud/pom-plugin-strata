import { usePluginI18n } from "../host/runtime";
import { useStatus } from "./Status";

declare const __POM_PLUGIN_CODE__: string;

/** Strata's own web app (chat, live GPU, speed and cache monitor), served by
 * the plugin through the POM plugin proxy. */
const APP = `/api/ui/plugins/${encodeURIComponent(__POM_PLUGIN_CODE__)}/proxy/`;

export function Monitor() {
  const { t } = usePluginI18n();
  const { status } = useStatus();

  // The frame is mounted once Strata answers and stays mounted while it does:
  // remounting on every status poll would reload the page.
  if (status?.state === "ready") {
    return <iframe className="pb-frame" title={t("monitorTitle")} src={APP} />;
  }
  return (
    <div className="pb-page pb-frame-wait">
      <p className={status?.state === "error" ? "pb-note pb-note-error" : "pb-note"} role="status">
        {status?.state === "error" && status.detail
          ? t("monitorUnavailable", { error: status.detail })
          : status && status.state !== "idle"
            ? t("monitorStarting")
            : t("monitorIdle")}
      </p>
    </div>
  );
}

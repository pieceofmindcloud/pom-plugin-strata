import type { ComponentType, ReactNode } from "react";

type I18nHook = () => { t: (key: string, values?: Record<string, unknown>) => string; locale?: string };

declare const __POM_PLUGIN_CODE__: string;

type Tone = "neutral" | "iris" | "aqua" | "amber" | "rose" | "sky" | "lime";

/** The part of the POM design kit this plugin draws with. */
type HostUi = {
  PageHeader: ComponentType<{ title: string; subtitle?: ReactNode; actions?: ReactNode }>;
  Panel: ComponentType<{ className?: string; children?: ReactNode }>;
  PanelHeader: ComponentType<{ title: ReactNode; subtitle?: ReactNode; actions?: ReactNode }>;
  Button: ComponentType<{
    variant?: "primary" | "outline" | "ghost" | "danger";
    size?: "sm" | "md";
    loading?: boolean;
    disabled?: boolean;
    onClick?: () => void;
    children?: ReactNode;
  }>;
  Badge: ComponentType<{ tone?: Tone; mono?: boolean; children: ReactNode }>;
  Progress: ComponentType<{ value: number | null; tone?: Tone }>;
  KeyValue: ComponentType<{ label: string; children: ReactNode }>;
};

type ScopedApi = {
  rpc: <T = unknown>(op: string, params?: unknown) => Promise<T>;
};

type PomHost = {
  version?: number;
  ui: HostUi;
  format: { bytes: (value: number | null | undefined) => string };
  hooks: { useI18n: I18nHook };
  plugin?: (code: string) => ScopedApi;
};

function host(): PomHost {
  const current = (globalThis as typeof globalThis & { __POM_HOST__?: PomHost }).__POM_HOST__;
  if (!current) throw new Error("POM host SDK is not installed");
  return current;
}

/** The POM's own components, so the screen looks like the rest of the panel. */
export function ui(): HostUi {
  return host().ui;
}

export function formatBytes(value: number): string {
  return host().format.bytes(value);
}

/** Translations of this plugin; the build prefixes every key with the plugin code. */
export function usePluginI18n(): ReturnType<I18nHook> {
  const { t, locale } = host().hooks.useI18n();
  const pluginCode = __POM_PLUGIN_CODE__;
  return { t: (key: string, values?: Record<string, unknown>) => t(`${pluginCode}.${key}`, values), locale };
}

/** Calls one of the plugin's `rpc.<op>` operations (POM host version 4). */
export function rpc<T>(op: string, params?: unknown): Promise<T> {
  const scoped = host().plugin;
  if (typeof scoped !== "function") {
    return Promise.reject(new Error("this POM version has no plugin RPC"));
  }
  const api = scoped(__POM_PLUGIN_CODE__);
  if (typeof api.rpc !== "function") {
    return Promise.reject(new Error("this POM version has no plugin RPC"));
  }
  return api.rpc<T>(op, params);
}

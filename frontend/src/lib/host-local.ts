import { API_BASE_URL } from "./api-client";

/** True when the UI is served from this PC (loopback), where model packs can be managed. */
export function isHostLocalUi(): boolean {
  if (typeof window === "undefined") return false;
  const isLoopback = (hostname: string) => {
    const host = hostname.replace(/^\[|\]$/g, "").toLowerCase();
    if (host === "localhost" || host === "::1") return true;
    const parts = host.split(".").map(Number);
    return parts.length === 4 && parts[0] === 127 && parts.every((part) => Number.isInteger(part) && part >= 0 && part <= 255);
  };
  try {
    const localWindow = Boolean(window.electronAPI) || isLoopback(window.location.hostname);
    return localWindow && isLoopback(new URL(API_BASE_URL).hostname);
  } catch {
    return false;
  }
}

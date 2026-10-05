/**
 * Theme state for the whole app: one provider, one localStorage key,
 * one place that sets the `.dark` class on <html>. Components take
 * their colours from the token blocks in index.css and never carry a
 * hex or a `dark:` colour variant; this file only decides which block
 * applies.
 *
 * index.html applies the class once before the bundle loads (the FOUC
 * guard); it mirrors readPref/resolve here and the two must stay in
 * agreement.
 *
 * Unlike the species-name mode, switching the theme never reloads the
 * page: the provider re-renders, and chart code re-reads its colours
 * through useChartColors below.
 *
 * The provider component lives in ThemeProvider.tsx; this module holds
 * the state and hooks, because a component file must export only
 * components for fast refresh to work (same split as viewer-tools.ts).
 */

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
} from "react";

export type ThemePref = "system" | "light" | "dark";
type ResolvedTheme = "light" | "dark";

const LS_KEY = "addaxai:theme";
const DARK_QUERY = "(prefers-color-scheme: dark)";

function readPref(): ThemePref {
  try {
    const v = localStorage.getItem(LS_KEY);
    if (v === "light" || v === "dark" || v === "system") return v;
  } catch {
    /* blocked storage means system */
  }
  return "system";
}

function resolve(pref: ThemePref): ResolvedTheme {
  if (pref === "system") {
    return window.matchMedia(DARK_QUERY).matches ? "dark" : "light";
  }
  return pref;
}

/** Synchronous on purpose: the class must be on <html> before React
 *  re-renders, so getComputedStyle in useChartColors reads the new
 *  token values in the same pass. */
function applyClass(resolved: ResolvedTheme) {
  document.documentElement.classList.toggle("dark", resolved === "dark");
}

interface ThemeContextValue {
  /** The stored preference, what the menu radio shows. */
  theme: ThemePref;
  /** What is actually on screen ("system" resolved). */
  resolvedTheme: ResolvedTheme;
  setTheme: (pref: ThemePref) => void;
}

const ThemeContext = createContext<ThemeContextValue | null>(null);

/** Everything ThemeProvider.tsx needs; internal to the theme pair. */
export function useThemeState(): ThemeContextValue {
  const [pref, setPref] = useState<ThemePref>(readPref);
  const [resolved, setResolved] = useState<ResolvedTheme>(() =>
    resolve(readPref())
  );

  const setTheme = useCallback((next: ThemePref) => {
    try {
      localStorage.setItem(LS_KEY, next);
    } catch {
      /* the session still gets the theme, it just won't persist */
    }
    const r = resolve(next);
    applyClass(r);
    setPref(next);
    setResolved(r);
    window.electronAPI?.setThemeMenuMode?.(next);
  }, []);

  // Follow the OS while the preference is "system".
  useEffect(() => {
    if (pref !== "system") return;
    const mq = window.matchMedia(DARK_QUERY);
    const onChange = () => {
      const r = resolve("system");
      applyClass(r);
      setResolved(r);
    };
    mq.addEventListener("change", onChange);
    return () => mq.removeEventListener("change", onChange);
  }, [pref]);

  // Reconcile with whatever the index.html guard applied, and tell
  // Electron the stored preference so the menu radio, nativeTheme and
  // the pre-paint window background match after a fresh launch
  // (setTheme covers every later change).
  useEffect(() => {
    applyClass(resolved);
    window.electronAPI?.setThemeMenuMode?.(pref);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return useMemo(
    () => ({ theme: pref, resolvedTheme: resolved, setTheme }),
    [pref, resolved, setTheme]
  );
}

export { ThemeContext };

export function useTheme(): ThemeContextValue {
  const ctx = useContext(ThemeContext);
  if (!ctx) {
    throw new Error("useTheme must be used inside ThemeProvider");
  }
  return ctx;
}

/** `#rgb` or `#rrggbb` to `rgba(...)`. Chart tints must go through
 *  this instead of hex-concatenation (`color + "20"`), which breaks
 *  the moment a colour is not a 6-digit hex. */
export function withAlpha(hex: string, alpha: number): string {
  const h = hex.replace("#", "");
  const full =
    h.length === 3 ? h.split("").map((c) => c + c).join("") : h;
  const r = parseInt(full.slice(0, 2), 16);
  const g = parseInt(full.slice(2, 4), 16);
  const b = parseInt(full.slice(4, 6), 16);
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

export interface ChartColors {
  /** Tick and axis label text. */
  axis: string;
  /** Primary text, for titles drawn inside a chart. */
  foreground: string;
  /** Grid lines. */
  grid: string;
  /** Card surface behind a chart (matrix ramps start here in dark). */
  card: string;
  primaryInk: string;
  goodInk: string;
  badInk: string;
  middle: string;
  destructiveInk: string;
  warningInk: string;
  infoInk: string;
  successInk: string;
  withAlpha: typeof withAlpha;
}

/**
 * Resolved colour strings for canvas/chart.js, which cannot read CSS
 * variables themselves. Memoised per resolved theme; every chart that
 * draws with these must include them (or resolvedTheme) in its own
 * memo deps, or it silently keeps the old theme's colours.
 */
export function useChartColors(): ChartColors {
  const { resolvedTheme } = useTheme();
  return useMemo(() => {
    const cs = getComputedStyle(document.documentElement);
    const triplet = (name: string) =>
      `hsl(${cs.getPropertyValue(name).trim()})`;
    const raw = (name: string) => cs.getPropertyValue(name).trim();
    return {
      axis: triplet("--muted-foreground"),
      foreground: triplet("--foreground"),
      grid: raw("--chart-grid"),
      card: triplet("--card"),
      primaryInk: raw("--primary-ink"),
      goodInk: raw("--good-ink"),
      badInk: raw("--bad-ink"),
      middle: raw("--middle"),
      destructiveInk: raw("--destructive-ink"),
      warningInk: raw("--warning-ink"),
      infoInk: raw("--info-ink"),
      successInk: raw("--success-ink"),
      withAlpha,
    };
    // getComputedStyle is the dependency that matters; resolvedTheme is
    // its proxy (the class flip happens before this render, see
    // applyClass).
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [resolvedTheme]);
}

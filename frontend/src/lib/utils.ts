/**
 * Utility functions for shadcn/ui components.
 *
 * Following DEVELOPERS.md principles:
 * - Simple, clear purpose
 * - Type hints everywhere
 */

import { type ClassValue, clsx } from "clsx";
import { twMerge } from "tailwind-merge";

/**
 * Merges Tailwind CSS classes with proper precedence.
 * Used by shadcn/ui components for conditional styling.
 */
export function cn(...inputs: ClassValue[]): string {
  return twMerge(clsx(inputs));
}

/**
 * Format a number in compact form: 0, 1, 999, 1.2k, 148k, 2.3M.
 */
export function formatCompact(n: number): string {
  if (n < 1000) return String(n);
  if (n < 1_000_000) {
    const k = n / 1000;
    return k >= 10 ? `${Math.round(k)}k` : `${k.toFixed(1).replace(/\.0$/, "")}k`;
  }
  const m = n / 1_000_000;
  return m >= 10 ? `${Math.round(m)}M` : `${m.toFixed(1).replace(/\.0$/, "")}M`;
}

/**
 * Format a datetime offset in seconds as a human-readable string.
 * Shows every non-zero component (days, hours, minutes, seconds).
 *
 * Examples:
 *   0      -> "no offset"
 *   30     -> "+30 seconds"
 *   3665   -> "+1 hour, 1 minute, 5 seconds"
 *   -86400 -> "-1 day"
 */
export function formatOffset(seconds: number): string {
  if (seconds === 0) return "no offset";
  const sign = seconds > 0 ? "+" : "-";
  const abs = Math.abs(seconds);
  const days = Math.floor(abs / 86400);
  const hours = Math.floor((abs % 86400) / 3600);
  const minutes = Math.floor((abs % 3600) / 60);
  const secs = abs % 60;
  const parts: string[] = [];
  if (days) parts.push(`${days} ${days === 1 ? "day" : "days"}`);
  if (hours) parts.push(`${hours} ${hours === 1 ? "hour" : "hours"}`);
  if (minutes) parts.push(`${minutes} ${minutes === 1 ? "minute" : "minutes"}`);
  if (secs) parts.push(`${secs} ${secs === 1 ? "second" : "seconds"}`);
  return `${sign}${parts.join(", ")}`;
}

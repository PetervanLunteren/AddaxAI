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

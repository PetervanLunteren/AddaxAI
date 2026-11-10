/**
 * Callout - the single block-level advisory box for the whole app.
 *
 * One component for every info / warning / success / error message that sits
 * inline in a flow (dialog notes, step warnings, run advisories, section
 * banners). Replaces the old shadcn `Alert` and the hand-rolled colored
 * `<div>`s so every advisory looks and behaves the same.
 *
 * NOT for: field validation (use FormMessage), transient toasts (use sonner),
 * status badges/cards (color-by-state, not a message), or full-screen error
 * states.
 *
 * Density:
 *   - "default": dialog notes and standalone advisories (p-4, text-sm).
 *   - "compact": tight inline warnings inside busy panels (px-3 py-2, text-xs).
 *
 * Banners are just a full-width Callout with an `action` (and optionally
 * `onDismiss`); there is no separate Banner component.
 */

import type { LucideIcon } from "lucide-react";
import { AlertCircle, AlertTriangle, CheckCircle2, Info, X } from "lucide-react";
import { cn } from "@/lib/utils";

export type CalloutVariant = "info" | "warning" | "success" | "error";
type CalloutSize = "default" | "compact";

interface CalloutProps {
  children: React.ReactNode;
  variant?: CalloutVariant;
  size?: CalloutSize;
  title?: string;
  /** Right-aligned action area, e.g. a button. Makes a "banner". */
  action?: React.ReactNode;
  /** When set, renders a dismiss "×" in the top-right corner. */
  onDismiss?: () => void;
  /** Hide the leading variant icon (rare; the icon is the default). */
  hideIcon?: boolean;
  className?: string;
}

// Boxes and icons use the intent trios from index.css (identical to
// the old tailwind tints in light by construction). Body text is the
// one deliberate exception to the no-dark:-variants rule: the light
// -900 shades are darker than the inks and keeping them keeps light
// mode pixel-identical, so the dark override lives here, in the one
// component, instead of as a fourth token.
const VARIANTS: Record<
  CalloutVariant,
  { box: string; text: string; icon: LucideIcon; iconColor: string }
> = {
  info: {
    box: "bg-info-subtle border-info-border",
    text: "text-blue-900 dark:text-info-ink",
    icon: Info,
    iconColor: "text-info-ink",
  },
  warning: {
    box: "bg-warning-subtle border-warning-border",
    text: "text-amber-900 dark:text-warning-ink",
    icon: AlertTriangle,
    iconColor: "text-warning-ink",
  },
  success: {
    box: "bg-success-subtle border-success-border",
    text: "text-green-900 dark:text-success-ink",
    icon: CheckCircle2,
    iconColor: "text-success-ink",
  },
  error: {
    box: "bg-destructive-subtle border-destructive-border",
    text: "text-red-900 dark:text-destructive-ink",
    icon: AlertCircle,
    iconColor: "text-destructive-ink",
  },
};

const SIZES: Record<
  CalloutSize,
  { pad: string; gap: string; icon: string; text: string }
> = {
  default: { pad: "p-4", gap: "gap-3", icon: "h-5 w-5", text: "text-sm" },
  compact: { pad: "px-3 py-2", gap: "gap-2.5", icon: "h-4 w-4", text: "text-xs" },
};

export function Callout({
  children,
  variant = "info",
  size = "default",
  title,
  action,
  onDismiss,
  hideIcon = false,
  className,
}: CalloutProps) {
  const v = VARIANTS[variant];
  const s = SIZES[size];
  const Icon = v.icon;

  return (
    <div
      className={cn(
        "flex items-start rounded-lg border",
        s.pad,
        s.gap,
        v.box,
        v.text,
        className,
      )}
      role={variant === "error" ? "alert" : "status"}
    >
      {!hideIcon && (
        <Icon className={cn("shrink-0 mt-0.5", s.icon, v.iconColor)} />
      )}
      <div className={cn("flex-1 min-w-0", s.text)}>
        {title && <p className="font-semibold mb-0.5">{title}</p>}
        <div className={cn(!title && "leading-snug")}>{children}</div>
      </div>
      {action && <div className="shrink-0 self-center">{action}</div>}
      {onDismiss && (
        <button
          type="button"
          onClick={onDismiss}
          aria-label="Dismiss"
          className={cn(
            "shrink-0 rounded p-0.5 opacity-60 transition-opacity hover:opacity-100",
            v.iconColor,
          )}
        >
          <X className={s.icon} />
        </button>
      )}
    </div>
  );
}

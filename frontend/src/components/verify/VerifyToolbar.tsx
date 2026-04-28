/**
 * Verify toolbar — slim row below the filter bar.
 *
 * Hosts the inline utility icons (help / keyboard / settings / refresh,
 * tab-specific) on the left and the verification progress pill on the
 * right. Each tab composes the contents from `VerifyToolbarIcon` and
 * `VerifyProgressPill`.
 */

import type { ComponentType, ReactNode, SVGProps } from "react";

import { cn } from "../../lib/utils";

interface VerifyToolbarProps {
  children: ReactNode;
  className?: string;
}

export function VerifyToolbar({ children, className }: VerifyToolbarProps) {
  return (
    <div
      className={cn(
        "flex flex-wrap items-center gap-3 min-h-12 py-2 px-3 bg-white rounded-lg border shadow-sm",
        className,
      )}
    >
      {children}
    </div>
  );
}

interface VerifyToolbarIconProps {
  icon: ComponentType<SVGProps<SVGSVGElement>>;
  title: string;
  onClick: () => void;
  disabled?: boolean;
  /** When true the icon spins (used for in-flight refresh). */
  spinning?: boolean;
}

export function VerifyToolbarIcon({
  icon: Icon,
  title,
  onClick,
  disabled = false,
  spinning = false,
}: VerifyToolbarIconProps) {
  return (
    <button
      type="button"
      title={title}
      aria-label={title}
      onClick={onClick}
      disabled={disabled}
      className="text-muted-foreground hover:text-foreground disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
    >
      <Icon className={cn("h-4 w-4", spinning && "animate-spin")} />
    </button>
  );
}

interface VerifyProgressPillProps {
  /** 0 to 100. Bar width and the percentage label render from this. */
  pct: number;
  /** Trailing text, e.g. "events verified" / "files verified". */
  label: string;
}

export function VerifyProgressPill({ pct, label }: VerifyProgressPillProps) {
  const clamped = Math.max(0, Math.min(100, pct));
  return (
    <div className="ml-auto flex items-center gap-1.5 text-xs text-muted-foreground">
      <div className="relative h-2 w-20 overflow-hidden rounded-full bg-muted">
        <div
          className="h-full transition-all duration-500 ease-out rounded-full"
          style={{ width: `${clamped}%`, backgroundColor: "#0f6064" }}
        />
      </div>
      {Math.round(clamped)}% {label}
    </div>
  );
}

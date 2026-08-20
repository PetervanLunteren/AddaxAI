/**
 * Verify toolbar — slim row below the filter bar.
 *
 * Hosts the inline utility icons (help / keyboard / settings,
 * tab-specific) on the left and the verification progress pill on the
 * right. Each tab composes the contents from `VerifyToolbarIcon` and
 * `VerifyProgressPill`.
 */

import type { ComponentType, ReactNode, SVGProps } from "react";
import { CircleHelp } from "lucide-react";

import { cn } from "../../lib/utils";

const GUIDE_URLS = {
  labels: "https://docs.addaxai.com/docs/guides/check-labels/",
  counts: "https://docs.addaxai.com/docs/guides/confirm-counts/",
} as const;

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

/** Shared chrome for the toolbar's icon buttons: a 32px ghost button so
 * the icons sit on the same visual line as the labeled buttons and
 * dropdowns in the row. Also used by the popover triggers (keyboard
 * shortcuts, view options) so every icon in the row matches. */
export const VERIFY_TOOLBAR_ICON_CLASS =
  "inline-flex h-8 w-8 items-center justify-center rounded-md " +
  "text-muted-foreground transition-colors hover:bg-muted " +
  "hover:text-foreground disabled:opacity-50 disabled:cursor-not-allowed";

/** The (?) icon: opens the step's guide on docs.addaxai.com, where the
 * workflow explanation and the video tutorial live. Same chrome as the
 * icon buttons so it sits on the same line. The keys stay in the app,
 * in the keyboard popover next to it. */
export function VerifyGuideLink({ step }: { step: keyof typeof GUIDE_URLS }) {
  return (
    <a
      href={GUIDE_URLS[step]}
      target="_blank"
      rel="noopener noreferrer"
      title="Open the guide"
      aria-label="Open the guide"
      className={VERIFY_TOOLBAR_ICON_CLASS}
    >
      <CircleHelp className="h-4 w-4" />
    </a>
  );
}

interface VerifyToolbarIconProps {
  icon: ComponentType<SVGProps<SVGSVGElement>>;
  title: string;
  onClick: () => void;
  disabled?: boolean;
  /** Toggle icons pass this so an enabled state reads as "on" (teal),
   *  not just a swapped glyph the user has to hunt for. */
  active?: boolean;
}

export function VerifyToolbarIcon({
  icon: Icon,
  title,
  onClick,
  disabled = false,
  active = false,
}: VerifyToolbarIconProps) {
  return (
    <button
      type="button"
      title={title}
      aria-label={title}
      aria-pressed={active}
      onClick={onClick}
      disabled={disabled}
      className={cn(
        VERIFY_TOOLBAR_ICON_CLASS,
        active &&
          "bg-primary text-primary-foreground hover:bg-primary/90 hover:text-primary-foreground",
      )}
    >
      <Icon className="h-4 w-4" />
    </button>
  );
}

interface VerifyProgressPillProps {
  /** 0 to 100. Bar width and the percentage label render from this. */
  pct: number;
  /** Trailing text, e.g. "events verified" / "files verified". */
  label: string;
  /** Optional native tooltip explaining what population the % covers. */
  title?: string;
}

export function VerifyProgressPill({
  pct,
  label,
  title,
}: VerifyProgressPillProps) {
  const clamped = Math.max(0, Math.min(100, pct));
  return (
    <div
      title={title}
      className="flex items-center gap-1.5 text-xs text-muted-foreground"
    >
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

import type { ReactNode } from "react";

/**
 * The keyboard hint on a button: "Verify ⏎", "Mark false X".
 *
 * One style for the whole app. `onPrimary` is for a key on a filled
 * (primary) button, where the muted border would vanish into the fill.
 */
export function Kbd({ children, onPrimary }: { children: ReactNode; onPrimary?: boolean }) {
  return (
    <kbd
      className={
        onPrimary
          ? "ml-1.5 text-[10px] font-sans text-primary-foreground/60 border border-primary-foreground/30 rounded px-1 py-0.5 shadow-[0_1px_0_0_rgba(255,255,255,0.1)] leading-none"
          : "ml-1.5 text-[10px] font-sans text-muted-foreground/60 border border-border/60 rounded px-1 py-0.5 shadow-[0_1px_0_0_rgba(0,0,0,0.08)] leading-none"
      }
    >
      {children}
    </kbd>
  );
}

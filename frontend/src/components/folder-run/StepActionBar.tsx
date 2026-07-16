/**
 * The sticky bottom bar carrying a step's Back / primary action.
 *
 * Steps are taller than a laptop viewport: the Save step needs ~975px to show
 * its button, so at 1080p/150% scaling (720px) it sits 210px below the fold.
 * A bar at the natural end of the form is therefore off-screen for everyone
 * not on a desktop monitor, which reads as "there is nothing left to do".
 * Sticking it to the bottom keeps the action in view without moving it above
 * the options that decide what it does.
 *
 * Layout contract: the negative margins bleed the bar to the page edges, so
 * this must stay a DIRECT child of the step root, where FolderRunLayout's
 * padding (px-4 / sm:px-6 / lg:px-8) applies. Nested inside a column or a
 * card, the bleed misaligns. FolderRunLayout deliberately avoids transforms
 * and full-bleed for the same reason: they would break sticky positioning.
 */

import type { ReactNode } from "react";

export function StepActionBar({ children }: { children: ReactNode }) {
  return (
    <div className="sticky bottom-0 z-30 -mx-4 border-t bg-white/80 px-4 py-3 backdrop-blur-sm sm:-mx-6 sm:px-6 lg:-mx-8 lg:px-8">
      <div className="mx-auto flex max-w-7xl items-center justify-between gap-3">
        {children}
      </div>
    </div>
  );
}

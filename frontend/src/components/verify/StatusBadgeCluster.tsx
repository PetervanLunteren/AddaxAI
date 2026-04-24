/**
 * StatusBadgeCluster - top-right corner cluster of status badges for
 * Files / Events cards.
 *
 * Three booleans drive up to three circular badges: verified (teal),
 * favorited (dark red), flagged (light teal). Stacked right-to-left with
 * an overlapping ring-2 ring-background treatment so the badges protrude
 * from the card's corner. Matches AddaxAI-Connect's ImagesPage pattern
 * (services/frontend/src/pages/ImagesPage.tsx:306-334) and WebUI's status
 * colour palette (#882000 / #71b7ba / #0f6064).
 *
 * The parent Card must be `overflow-visible` for the -top-2 -right-2
 * offset to protrude. Keep any inner image wrapper with its own
 * overflow-hidden so detection overlays still clip correctly.
 */

import { Check, Flag, Heart } from "lucide-react";

interface StatusBadgeClusterProps {
  verified?: boolean;
  favorited?: boolean;
  flagged?: boolean;
}

export function StatusBadgeCluster({
  verified = false,
  favorited = false,
  flagged = false,
}: StatusBadgeClusterProps) {
  if (!verified && !favorited && !flagged) return null;

  return (
    <div className="absolute -top-2 -right-2 z-10 flex flex-row-reverse -space-x-1.5 space-x-reverse">
      {verified && (
        <div
          className="relative z-30 w-6 h-6 rounded-full flex items-center justify-center ring-2 ring-background"
          style={{ backgroundColor: "#0f6064" }}
          title="Verified"
        >
          <Check className="h-3.5 w-3.5 text-white" strokeWidth={3} />
        </div>
      )}
      {favorited && (
        <div
          className="relative z-20 w-6 h-6 rounded-full flex items-center justify-center ring-2 ring-background"
          style={{ backgroundColor: "#882000" }}
          title="Favorited"
        >
          <Heart
            className="h-3.5 w-3.5 text-white fill-current"
            strokeWidth={2.5}
          />
        </div>
      )}
      {flagged && (
        <div
          className="relative z-10 w-6 h-6 rounded-full flex items-center justify-center ring-2 ring-background"
          style={{ backgroundColor: "#71b7ba" }}
          title="Flagged"
        >
          <Flag
            className="h-3.5 w-3.5 text-white fill-current"
            strokeWidth={2.5}
          />
        </div>
      )}
    </div>
  );
}

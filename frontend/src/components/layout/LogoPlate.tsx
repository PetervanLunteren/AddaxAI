/**
 * The AddaxAI wordmark on a frosted plate.
 *
 * The wordmark is teal on a transparent background, so over a photo it
 * sinks into the picture. The plate is what keeps it readable. Shared by
 * the home screen and the setup screen, which are the two screens that
 * put the logo on a photo, so the two cannot drift apart.
 *
 * backdrop-filter is set inline so the look does not depend on the
 * Tailwind backdrop-blur utilities being enabled.
 */

import { cn } from "../../lib/utils";

const GLASS: React.CSSProperties = {
  backgroundColor: "rgba(255, 255, 255, 0.72)",
  backdropFilter: "blur(22px) saturate(150%)",
  WebkitBackdropFilter: "blur(22px) saturate(150%)",
};

interface LogoPlateProps {
  /** Positioning and margins. The plate itself brings its own padding. */
  className?: string;
  /** Height of the wordmark, e.g. "h-14". Width always follows. */
  logoClassName?: string;
}

export function LogoPlate({ className, logoClassName }: LogoPlateProps) {
  return (
    <div
      style={GLASS}
      className={cn(
        "inline-flex items-center rounded-2xl border border-white/50 px-6 py-4",
        "shadow-[0_20px_50px_-20px_rgba(0,0,0,0.55)]",
        className,
      )}
    >
      <img
        src="/branding/logo-wordmark.png"
        alt="AddaxAI"
        className={cn("w-auto", logoClassName)}
      />
    </div>
  );
}

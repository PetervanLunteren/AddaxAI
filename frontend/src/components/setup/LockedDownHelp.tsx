/**
 * The one line under every setup failure that points at the help page for
 * managed computers and inspected networks.
 *
 * Static on purpose. It never guesses from the error text whether the
 * machine is locked down: the docs page opens with "pick the error you
 * see", so the triage lives there and the app carries no logic. The only
 * variation is the section it lands on, chosen from the backend's
 * `error_kind`, which is tagged server-side and never inferred here.
 */

import { ExternalLink } from "lucide-react";

export const LOCKED_DOWN_HELP_URL =
  "https://docs.addaxai.com/docs/help/locked-down-computers";

interface LockedDownHelpProps {
  /** The backend's `error_kind` for the failure on screen, or null. */
  errorKind: string | null;
}

export function LockedDownHelp({ errorKind }: LockedDownHelpProps) {
  const blocked = errorKind === "network_blocked";
  const href = blocked
    ? `${LOCKED_DOWN_HELP_URL}#only-the-models-are-blocked`
    : LOCKED_DOWN_HELP_URL;
  const lead = blocked
    ? "Your IT team can allow the download servers, or you can bring the models over from another computer."
    : "On a work or university computer?";
  const label = blocked
    ? "Read how"
    : "Read how to set up on a locked-down machine or network";

  return (
    <p className="text-center text-xs text-muted-foreground">
      {lead}{" "}
      <a
        href={href}
        target="_blank"
        rel="noopener noreferrer"
        className="text-primary underline hover:opacity-80 inline-flex items-center gap-1"
      >
        {label}
        <ExternalLink className="h-3 w-3" />
      </a>
    </p>
  );
}

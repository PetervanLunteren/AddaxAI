/**
 * The one line under every setup failure that points at the help page for
 * managed computers and inspected networks.
 *
 * Static on purpose. It never guesses from the error text whether the
 * machine is locked down: the docs page opens with "pick the error you
 * see", so the triage lives there and the app carries no logic. The only
 * variation is the page and section it lands on, chosen from the backend's
 * `error_kind`, which is tagged server-side and never inferred here.
 */

import { ExternalLink } from "lucide-react";

export const LOCKED_DOWN_HELP_URL =
  "https://docs.addaxai.com/docs/help/locked-down-computers";

/** The section that walks through setting a variable on each OS. */
export const SET_VARIABLE_HELP_URL =
  "https://docs.addaxai.com/docs/reference/environment-variables#how-to-set-them";

interface HelpLine {
  lead: string;
  label: string;
  href: string;
}

const DEFAULT_HELP: HelpLine = {
  lead: "On a work or university computer?",
  label: "Read how to set up on a locked-down machine or network",
  href: LOCKED_DOWN_HELP_URL,
};

const HELP_BY_KIND: { [kind: string]: HelpLine | undefined } = {
  network_blocked: {
    lead: "Your IT team can allow the download servers, or you can bring the models over from another computer.",
    label: "Read how",
    href: `${LOCKED_DOWN_HELP_URL}#only-the-models-are-blocked`,
  },
  // The message itself names the variable and a value. What a
  // non-technical user is missing is how to set one at all.
  non_ascii_path: {
    lead: "The message above names the variable to set.",
    label: "Read how to set it",
    href: SET_VARIABLE_HELP_URL,
  },
  // Tagged only when Windows reports Smart App Control as on, for a
  // failure with no known cause. "Can" because it is the likely cause,
  // not a confirmed one.
  smart_app_control: {
    lead: "Windows Smart App Control is on and can block parts of the analysis software.",
    label: "Read how to check and fix it",
    href: `${LOCKED_DOWN_HELP_URL}#windows-smart-app-control-blocks-the-analysis`,
  },
};

interface LockedDownHelpProps {
  /** The backend's `error_kind` for the failure on screen, or null. */
  errorKind: string | null;
}

export function LockedDownHelp({ errorKind }: LockedDownHelpProps) {
  const { lead, label, href } =
    (errorKind !== null && HELP_BY_KIND[errorKind]) || DEFAULT_HELP;

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

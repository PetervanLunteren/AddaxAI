/**
 * Header control to switch species names between common and scientific.
 *
 * The preference is global (per device, localStorage). Changing it reloads
 * the page so every rendered name flips consistently; toggling is rare, so
 * the reload cost is acceptable and keeps the implementation simple.
 *
 * Rendered next to DiagnosticReportButton in page headers that show
 * species names.
 */

import {
  getSpeciesNameMode,
  setSpeciesNameMode,
  type SpeciesNameMode,
} from "../../lib/species-name-mode";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "../ui/tooltip";

const OPTIONS: { value: SpeciesNameMode; label: string }[] = [
  { value: "common", label: "Common" },
  { value: "scientific", label: "Scientific" },
];

export function SpeciesNameToggle() {
  const mode = getSpeciesNameMode();

  return (
    <TooltipProvider delayDuration={200}>
      <Tooltip>
        <TooltipTrigger asChild>
          <div
            className="inline-flex h-9 items-center rounded-md border bg-white p-0.5 text-xs"
            role="group"
            aria-label="Species name display"
          >
            {OPTIONS.map((opt) => (
              <button
                key={opt.value}
                type="button"
                aria-pressed={mode === opt.value}
                onClick={() => {
                  if (mode !== opt.value) setSpeciesNameMode(opt.value);
                }}
                className={
                  "rounded px-2 py-1 transition-colors " +
                  (mode === opt.value
                    ? "bg-slate-900 text-white"
                    : "text-muted-foreground hover:text-foreground")
                }
              >
                {opt.label}
              </button>
            ))}
          </div>
        </TooltipTrigger>
        <TooltipContent>
          Show species as common or scientific names
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}

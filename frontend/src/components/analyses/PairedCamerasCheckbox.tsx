/**
 * The "Paired cameras" checkbox, shared by the add-deployment card and the
 * edit-deployment dialog so the wording cannot drift.
 *
 * One deployment folder with one subfolder per camera. With the box ticked
 * the subfolders count as one camera: their files cluster into one event
 * and the effort counts once. See DEVELOPERS.md "Paired cameras".
 */

import { ExternalLink } from "lucide-react";
import { Checkbox } from "@/components/ui/checkbox";

export const PAIRED_CAMERAS_CAPTION =
  "One subfolder per camera, for dependent cameras that trigger on the same animals. Their files form one event and the trap nights count once.";

/** The docs section that explains the folder layout and what the tick changes. */
export const PAIRED_CAMERAS_DOCS_URL =
  "https://docs.addaxai.com/docs/understanding/how-a-project-is-organised#paired-cameras";

interface PairedCamerasCheckboxProps {
  checked: boolean;
  onChange: (checked: boolean) => void;
}

export function PairedCamerasCheckbox({
  checked,
  onChange,
}: PairedCamerasCheckboxProps) {
  return (
    <div className="space-y-0.5">
      <label className="flex cursor-pointer items-center gap-3">
        <Checkbox
          checked={checked}
          onCheckedChange={(v) => onChange(v === true)}
        />
        <span className="text-sm font-medium">Paired cameras</span>
      </label>
      {/* The caption and its link sit outside the label so a click on
          "Read more" opens the docs instead of toggling the box. */}
      <div className="pl-7">
        <p className="text-xs text-muted-foreground">
          {PAIRED_CAMERAS_CAPTION}{" "}
          <a
            href={PAIRED_CAMERAS_DOCS_URL}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-1 text-primary underline hover:opacity-80"
          >
            Read more
            <ExternalLink className="h-3 w-3" />
          </a>
        </p>
      </div>
    </div>
  );
}

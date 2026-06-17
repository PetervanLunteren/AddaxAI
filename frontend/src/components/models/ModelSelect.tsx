/**
 * Shared model dropdown (detection / classification / embedding).
 *
 * Owns the consistent shell so the three model selects stay identical across
 * the create-project modal, project settings, and folder-run step 1:
 * - the trigger value (emoji + name via ModelSelectValue, or "∅ <noneLabel>"
 *   for the no-model option),
 * - the Radix remount-key workaround,
 * - the "Model details" link that opens the info slideout.
 *
 * Callers pass the option items as children, so a grouped cls list, a flat
 * det / emb list, and the optional none-item all stay at the call site.
 */

import { type ReactNode } from "react";
import {
  Select,
  SelectContent,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { FormControl } from "@/components/ui/form";
import { ModelSelectValue } from "./ModelSelectValue";
import type { ModelInfo } from "@/api/types";

interface ModelSelectProps {
  /** Current value, already defaulted to noneValue when empty (e.g. field.value ?? "none"). */
  value: string;
  onValueChange: (value: string) => void;
  /** Models of this category, used to render the selected model in the trigger. */
  models: ModelInfo[];
  placeholder: string;
  /** Sentinel value for the "no model" option. Omit for required selects (detection). */
  noneValue?: string;
  /** Trigger label when the none option is selected, e.g. "No classification model". */
  noneLabel?: string;
  /** Opens the model info slideout. When set and a real model is selected, a "Model details" link is shown. */
  onShowInfo?: () => void;
  /** SelectContent items: the optional none item plus the (grouped or flat) model items. */
  children: ReactNode;
}

export function ModelSelect({
  value,
  onValueChange,
  models,
  placeholder,
  noneValue,
  noneLabel,
  onShowInfo,
  children,
}: ModelSelectProps) {
  const isNone = noneValue !== undefined && value === noneValue;
  const selected = isNone ? undefined : models.find((m) => m.model_id === value);

  return (
    <div className="space-y-1">
      {/* key remounts on value change: inside a <form> Radix renders a hidden
          native <select> whose <option>s only exist while the dropdown is open,
          so setting the value post-mount with the dropdown closed would coerce
          it to "" and fire onValueChange(""). Keying to the value avoids that. */}
      <Select key={value} value={value} onValueChange={onValueChange}>
        <FormControl>
          <SelectTrigger>
            <SelectValue placeholder={placeholder}>
              {isNone ? (
                <span>∅ {noneLabel}</span>
              ) : selected ? (
                <ModelSelectValue model={selected} />
              ) : null}
            </SelectValue>
          </SelectTrigger>
        </FormControl>
        <SelectContent>{children}</SelectContent>
      </Select>
      {onShowInfo && selected && (
        <p className="pl-3 text-xs">
          <button
            type="button"
            onClick={onShowInfo}
            className="font-medium text-primary hover:underline"
          >
            Model details
          </button>
        </p>
      )}
    </div>
  );
}

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
import type {
  CustomModelInfo,
  CustomModelRegistrationRole,
  ModelInfo,
} from "@/api/types";
import { API_BASE_URL } from "@/lib/api-client";
import { CustomModelManagerDialog } from "./CustomModelManagerDialog";
import { useState } from "react";
import { toast } from "sonner";

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
  /** Enables custom detection/classification pack management for host-local pickers. */
  modelType?: "detection" | "classification";
  /** Lets a parent update coupled detector/classifier fields for a newly registered alias. */
  onModelCreated?: (model: CustomModelInfo, role?: CustomModelRegistrationRole) => void;
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
  modelType,
  onModelCreated,
  children,
}: ModelSelectProps) {
  const [manageOpen, setManageOpen] = useState(false);
  const isNone = noneValue !== undefined && value === noneValue;
  const selected = isNone ? undefined : models.find((m) => m.model_id === value);
  const canManage = modelType !== undefined && isHostLocalUi();

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
      {canManage && modelType && (
        <p className="pl-3 text-xs">
          <button
            type="button"
            onClick={() => setManageOpen(true)}
            className="font-medium text-primary hover:underline"
          >
            Manage custom models
          </button>
        </p>
      )}
      {canManage && modelType && (
        <CustomModelManagerDialog
          open={manageOpen}
          onOpenChange={setManageOpen}
          modelType={modelType}
          canSelectBoth={Boolean(onModelCreated)}
          onCreated={(model, role) => {
            if (model.type === modelType || (role === "both" && modelType === "classification")) {
              if (onModelCreated) {
                onModelCreated(model, role);
              } else if (role === "both" && modelType === "classification") {
                toast.info(
                  "Model registered. This screen keeps its detection model fixed; select this same model for Detection and Classification in Project settings to reuse its labels.",
                );
              } else {
                onValueChange(model.model_id);
              }
            }
          }}
        />
      )}
    </div>
  );
}

export function isHostLocalUi(): boolean {
  if (typeof window === "undefined") return false;
  const isLoopback = (hostname: string) => {
    const host = hostname.replace(/^\[|\]$/g, "").toLowerCase();
    if (host === "localhost" || host === "::1") return true;
    const parts = host.split(".").map(Number);
    return parts.length === 4 && parts[0] === 127 && parts.every((part) => Number.isInteger(part) && part >= 0 && part <= 255);
  };
  try {
    const localWindow = Boolean(window.electronAPI) || isLoopback(window.location.hostname);
    return localWindow && isLoopback(new URL(API_BASE_URL).hostname);
  } catch {
    return false;
  }
}

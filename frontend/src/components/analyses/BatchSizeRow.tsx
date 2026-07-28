/**
 * Performance card row for a single batch size field.
 *
 * Default / Custom dropdown plus a number input that appears when the
 * user picks Custom. Mode is derived from the field value: null = Default,
 * integer = Custom. Switching to Custom prefills with the model's GPU
 * default; switching back to Default sets the field to null.
 *
 * Single source of truth used by:
 * - SettingsPage (project's persistent batch size override)
 * - FolderRunModelStep (one-shot batch size override)
 *
 * The form data type is generic so the same component fits any form
 * that has a `number | null` field at the bound name.
 */

import type {
  Control,
  FieldPath,
  FieldPathByValue,
  FieldValues,
} from "react-hook-form";

import { Input } from "../ui/input";
import { FormControl, FormField, FormMessage } from "../ui/form";

import { SettingRow } from "./SettingRow";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "../ui/select";

interface BatchSizeRowProps<T extends FieldValues> {
  control: Control<T>;
  /** Form field name. Must point at a `number | null` field. */
  name: FieldPathByValue<T, number | null> & FieldPath<T>;
  label: string;
  description: string;
  defaultGpu: number;
  defaultCpu: number;
}

export function BatchSizeRow<T extends FieldValues>({
  control,
  name,
  label,
  description,
  defaultGpu,
  defaultCpu,
}: BatchSizeRowProps<T>) {
  const defaultLabel = `Default (${defaultGpu} on GPU, ${defaultCpu} on CPU)`;
  return (
    <FormField
      control={control}
      name={name}
      render={({ field }) => {
        const isCustom = field.value !== null && field.value !== undefined;
        return (
          <SettingRow
            label={label}
            description={description}
            isCustom={isCustom}
          >
              <div className="flex gap-2">
                <Select
                  value={isCustom ? "custom" : "default"}
                  onValueChange={(value) => {
                    if (value === "default") {
                      field.onChange(null);
                    } else if (field.value === null || field.value === undefined) {
                      // Only prefill with the GPU default when switching
                      // FROM Default. If the field already has a saved
                      // value (e.g. 12 from the DB), keep it. This guard
                      // also prevents Radix Select from overwriting the
                      // value when it fires onValueChange on programmatic
                      // prop changes (e.g. form.reset switching the
                      // Select from "default" to "custom").
                      field.onChange(defaultGpu);
                    }
                  }}
                >
                  <FormControl>
                    <SelectTrigger
                      className={isCustom ? "w-[130px] shrink-0" : ""}
                    >
                      <SelectValue />
                    </SelectTrigger>
                  </FormControl>
                  <SelectContent>
                    <SelectItem value="default">{defaultLabel}</SelectItem>
                    <SelectItem value="custom">Custom</SelectItem>
                  </SelectContent>
                </Select>
                {isCustom && (
                  <Input
                    type="number"
                    min={1}
                    max={256}
                    className="flex-1"
                    value={field.value ?? ""}
                    onChange={(e) => {
                      const raw = e.target.value;
                      if (raw === "") {
                        field.onChange(1);
                        return;
                      }
                      const parsed = parseInt(raw, 10);
                      field.onChange(Number.isNaN(parsed) ? 1 : parsed);
                    }}
                  />
                )}
              </div>
              <FormMessage />
          </SettingRow>
        );
      }}
    />
  );
}

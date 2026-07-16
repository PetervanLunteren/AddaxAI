/**
 * Advanced-settings row for the detection image size.
 *
 * Model default / Custom dropdown plus a number input that appears when the
 * user picks Custom. Mode is derived from the field value: null = Model
 * default, integer = Custom. Switching to Custom prefills a starting value;
 * switching back to Model default sets the field to null.
 *
 * "Model default" carries no number on purpose. null means the backend omits
 * the `--image_size` flag entirely (see ml/inference/megadetector.py), so
 * MegaDetector uses whatever its own model-native size is — that differs per
 * detection model and the app never learns it, so printing a number here
 * would be a guess.
 *
 * Single source of truth used by:
 * - SettingsPage (project's persistent image-size override)
 * - FolderRunModelStep (one-shot image-size override)
 *
 * Mirrors BatchSizeRow's shape and row layout so the two advanced rows read
 * the same. The form data type is generic so the same component fits any form
 * that has a `number | null` field at the bound name.
 */

import type {
  Control,
  FieldPath,
  FieldPathByValue,
  FieldValues,
} from "react-hook-form";

import { Input } from "../ui/input";
import {
  FormControl,
  FormDescription,
  FormField,
  FormLabel,
  FormMessage,
} from "../ui/form";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "../ui/select";

// Bounds mirror the backend's validation (schemas/project.py: ge=320, le=4096).
const MIN_IMAGE_SIZE = 320;
const MAX_IMAGE_SIZE = 4096;
// Starting point when switching to Custom. The reason to override is spotting
// small or distant animals, so we open above MegaDetector's usual native size.
const CUSTOM_PREFILL = 1600;

interface ImageSizeRowProps<T extends FieldValues> {
  control: Control<T>;
  /** Form field name. Must point at a `number | null` field. */
  name: FieldPathByValue<T, number | null> & FieldPath<T>;
  label: string;
  description: string;
}

export function ImageSizeRow<T extends FieldValues>({
  control,
  name,
  label,
  description,
}: ImageSizeRowProps<T>) {
  return (
    <FormField
      control={control}
      name={name}
      render={({ field }) => {
        const isCustom = field.value !== null && field.value !== undefined;
        return (
          <div className="grid grid-cols-2 items-center gap-8 py-6">
            <div className="space-y-1">
              <FormLabel>{label}</FormLabel>
              <FormDescription className="text-sm">
                {description}
              </FormDescription>
            </div>
            <div className="space-y-2">
              <div className="flex gap-2">
                <Select
                  value={isCustom ? "custom" : "default"}
                  onValueChange={(value) => {
                    if (value === "default") {
                      field.onChange(null);
                    } else if (
                      field.value === null ||
                      field.value === undefined
                    ) {
                      // Only prefill when switching FROM Model default. If the
                      // field already has a saved value, keep it. This guard
                      // also stops Radix Select from overwriting the value when
                      // it fires onValueChange on programmatic prop changes
                      // (e.g. form.reset flipping the Select to "custom").
                      field.onChange(CUSTOM_PREFILL);
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
                    <SelectItem value="default">Model default</SelectItem>
                    <SelectItem value="custom">Custom</SelectItem>
                  </SelectContent>
                </Select>
                {isCustom && (
                  <Input
                    type="number"
                    min={MIN_IMAGE_SIZE}
                    max={MAX_IMAGE_SIZE}
                    className="flex-1"
                    value={field.value ?? ""}
                    onChange={(e) => {
                      const raw = e.target.value;
                      if (raw === "") {
                        field.onChange(MIN_IMAGE_SIZE);
                        return;
                      }
                      const parsed = parseInt(raw, 10);
                      field.onChange(
                        Number.isNaN(parsed) ? MIN_IMAGE_SIZE : parsed,
                      );
                    }}
                  />
                )}
              </div>
              <FormMessage />
            </div>
          </div>
        );
      }}
    />
  );
}

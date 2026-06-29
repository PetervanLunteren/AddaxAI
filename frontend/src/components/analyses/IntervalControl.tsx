/**
 * Independence-interval picker: a preset dropdown that switches to a free
 * minutes input when "Custom" is chosen. Mirrors BatchSizeRow's dropdown↔input
 * behaviour.
 *
 * The value is an integer number of SECONDS (the DB column is an integer), but
 * the custom input is in MINUTES and accepts decimals for sub-minute
 * intervals; the entered minutes are rounded to whole seconds.
 *
 * Unlike BatchSizeRow (where mode is derived purely from null-vs-int and can
 * never collide), a custom seconds value can equal a preset (e.g. 30 min =
 * 1800 = the 30-minute preset). So mode is kept as explicit state: it is
 * forced to Custom one-way when a non-preset value arrives from outside, and
 * never forced back to preset, so typing a preset-equal value while in Custom
 * does not snap the input shut.
 *
 * Single source of truth used by SettingsPage (persistent project setting) and
 * FolderRunModelStep (one-shot run setting).
 */

import { useEffect, useState } from "react";

import { Input } from "../ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "../ui/select";

// Preset intervals in seconds. "Custom" is appended in the UI.
const PRESETS: { value: number; label: string }[] = [
  { value: 0, label: "Disabled" },
  { value: 60, label: "1 minute" },
  { value: 300, label: "5 minutes" },
  { value: 900, label: "15 minutes" },
  { value: 1800, label: "30 minutes" },
  { value: 3600, label: "60 minutes" },
];
const PRESET_SECONDS = new Set(PRESETS.map((p) => p.value));
const CUSTOM = "custom";

/** Seconds → a tidy minutes string (no trailing zeros). */
function secondsToMinutes(seconds: number): string {
  if (seconds % 60 === 0) return String(seconds / 60);
  return String(Number((seconds / 60).toFixed(4)));
}

interface IntervalControlProps {
  /** Current interval in seconds. */
  value: number;
  /** Called with the new interval in seconds. */
  onChange: (seconds: number) => void;
}

export function IntervalControl({ value, onChange }: IntervalControlProps) {
  const [mode, setMode] = useState<"preset" | "custom">(
    PRESET_SECONDS.has(value) ? "preset" : "custom",
  );
  // Local text for the minutes input so decimals can be typed without the
  // seconds round-trip clobbering in-progress input (e.g. "2.").
  const [minutesText, setMinutesText] = useState(() => secondsToMinutes(value));

  // A non-preset value arriving from outside (project load / sticky restore)
  // forces Custom. One-way: never forces back to preset.
  useEffect(() => {
    if (!PRESET_SECONDS.has(value)) setMode("custom");
  }, [value]);

  // Sync the minutes text from the seconds value on external changes, but skip
  // when the current text already represents this value (don't clobber typing).
  useEffect(() => {
    if (Math.round(parseFloat(minutesText || "0") * 60) !== value) {
      setMinutesText(secondsToMinutes(value));
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [value]);

  return (
    <div className="flex gap-2">
      <Select
        key={mode === "custom" ? CUSTOM : String(value)}
        value={mode === "custom" ? CUSTOM : String(value)}
        onValueChange={(v) => {
          if (v === CUSTOM) {
            setMode("custom");
          } else {
            setMode("preset");
            onChange(parseInt(v, 10));
          }
        }}
      >
        <SelectTrigger className={mode === "custom" ? "w-[150px] shrink-0" : ""}>
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          {PRESETS.map((p) => (
            <SelectItem key={p.value} value={String(p.value)}>
              {p.label}
            </SelectItem>
          ))}
          <SelectItem value={CUSTOM}>Custom</SelectItem>
        </SelectContent>
      </Select>
      {mode === "custom" && (
        <div className="flex flex-1 items-center gap-2">
          <Input
            type="number"
            min={0}
            step={0.5}
            className="flex-1"
            value={minutesText}
            onChange={(e) => {
              const raw = e.target.value;
              setMinutesText(raw);
              if (raw === "") {
                onChange(0);
                return;
              }
              const mins = parseFloat(raw);
              onChange(
                Number.isNaN(mins) ? 0 : Math.max(0, Math.round(mins * 60)),
              );
            }}
          />
          <span className="text-sm text-muted-foreground">minutes</span>
        </div>
      )}
    </div>
  );
}

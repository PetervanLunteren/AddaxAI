/**
 * Reusable tags editor for key:value metadata pairs.
 *
 * Renders an inline list of key+value input rows with add/remove controls.
 * Rows with empty keys are stripped when converting to dict for the parent.
 * The rows array is the local source of truth while editing; the parent
 * only receives cleaned output via onChange.
 */

import { useState, useCallback } from "react";
import { Plus, X } from "lucide-react";
import { Input } from "./input";
import { Button } from "./button";
import { Label } from "./label";

interface TagRow {
  key: string;
  value: string;
}

interface TagsEditorProps {
  /** Initial tags. Only read on mount (component re-mounts when entity changes). */
  value: Record<string, string>;
  onChange: (tags: Record<string, string>) => void;
}

function toRows(tags: Record<string, string>): TagRow[] {
  return Object.entries(tags).map(([key, value]) => ({ key, value }));
}

function toDict(rows: TagRow[]): Record<string, string> {
  const dict: Record<string, string> = {};
  for (const row of rows) {
    const k = row.key.trim();
    if (k) {
      dict[k] = row.value;
    }
  }
  return dict;
}

export function TagsEditor({ value, onChange }: TagsEditorProps) {
  const [rows, setRows] = useState<TagRow[]>(() => {
    const existing = toRows(value);
    return existing.length > 0 ? existing : [{ key: "", value: "" }];
  });

  const notify = useCallback(
    (next: TagRow[]) => onChange(toDict(next)),
    [onChange]
  );

  const addRow = () => {
    setRows((prev) => [...prev, { key: "", value: "" }]);
  };

  const removeRow = (index: number) => {
    const next = rows.filter((_, i) => i !== index);
    setRows(next);
    notify(next);
  };

  const updateRow = (index: number, field: "key" | "value", val: string) => {
    const next = rows.map((row, i) =>
      i === index ? { ...row, [field]: val } : row
    );
    setRows(next);
    notify(next);
  };

  return (
    <div className="space-y-2">
      <Label>Tags</Label>
      {rows.length > 0 && (
        <div className="space-y-2">
          {rows.map((row, i) => (
            <div key={i} className="flex items-center gap-2">
              <Input
                placeholder="e.g., Canopy cover"
                value={row.key}
                onChange={(e) => updateRow(i, "key", e.target.value)}
                className="flex-1"
              />
              <Input
                placeholder="e.g., Dense"
                value={row.value}
                onChange={(e) => updateRow(i, "value", e.target.value)}
                className="flex-1"
              />
              <Button
                type="button"
                variant="ghost"
                size="icon"
                className="h-8 w-8 shrink-0"
                onClick={() => removeRow(i)}
              >
                <X className="h-4 w-4" />
              </Button>
            </div>
          ))}
        </div>
      )}
      <Button
        type="button"
        variant="outline"
        size="sm"
        onClick={addRow}
      >
        <Plus className="h-4 w-4 mr-1" />
        Add tag
      </Button>
    </div>
  );
}

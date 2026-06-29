/**
 * Reusable tags editor for key:value metadata pairs.
 *
 * Renders an inline list of key+value input rows with add/remove controls.
 * Rows with empty keys are stripped when converting to dict for the parent.
 * The rows array is the local source of truth while editing; the parent
 * only receives cleaned output via onChange.
 */

import { useState, useMemo, useCallback } from "react";
import { Plus, X } from "lucide-react";
import { Input } from "./input";
import { Button } from "./button";
import { Label } from "./label";
import { FieldHeader } from "./field-header";
import { cn } from "../../lib/utils";

const MAX_KEY_LENGTH = 40;
const MAX_VALUE_LENGTH = 150;

interface TagRow {
  key: string;
  value: string;
}

interface TagsEditorProps {
  /** Initial tags. Only read on mount (component re-mounts when entity changes). */
  value: Record<string, string>;
  onChange: (tags: Record<string, string>) => void;
  /** Placeholder shown in the key input. Callers typically pass a
   * context-specific example so users can tell site tags from
   * deployment tags without repeating scope in the field label. */
  keyPlaceholder?: string;
  /** Placeholder shown in the value input. Same rationale as
   * keyPlaceholder. */
  valuePlaceholder?: string;
  /** Optional one-line caption under the "Tags" label explaining what the
   * tags are for in this context. The editor is generic (site tags,
   * deployment tags), so the copy is supplied by the caller. */
  description?: string;
}

function toRows(tags: Record<string, string>): TagRow[] {
  return Object.entries(tags).map(([key, value]) => ({ key, value }));
}

function toDict(rows: TagRow[]): Record<string, string> {
  const dict: Record<string, string> = {};
  for (const row of rows) {
    const k = row.key.trim();
    if (k) {
      dict[k] = row.value.trim();
    }
  }
  return dict;
}

export function TagsEditor({
  value,
  onChange,
  keyPlaceholder = "e.g., Baboon risk",
  valuePlaceholder = "e.g., High",
  description,
}: TagsEditorProps) {
  const [rows, setRows] = useState<TagRow[]>(() => {
    const existing = toRows(value);
    return existing.length > 0 ? existing : [{ key: "", value: "" }];
  });

  const notify = useCallback(
    (next: TagRow[]) => onChange(toDict(next)),
    [onChange]
  );

  // Detect duplicate keys (case-sensitive, trimmed)
  const duplicateKeys = useMemo(() => {
    const seen = new Set<string>();
    const duplicates = new Set<string>();
    for (const row of rows) {
      const k = row.key.trim();
      if (!k) continue;
      if (seen.has(k)) {
        duplicates.add(k);
      } else {
        seen.add(k);
      }
    }
    return duplicates;
  }, [rows]);

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

  const handleBlur = (index: number, field: "key" | "value") => {
    const row = rows[index];
    const trimmed = row[field].trim();
    if (trimmed !== row[field]) {
      updateRow(index, field, trimmed);
    }
  };

  const hasDuplicates = duplicateKeys.size > 0;

  return (
    <div className="space-y-2">
      <FieldHeader label={<Label>Tags</Label>} caption={description} />
      {rows.length > 0 && (
        <div className="space-y-2">
          {rows.map((row, i) => {
            const trimmedKey = row.key.trim();
            const isDuplicate = trimmedKey !== "" && duplicateKeys.has(trimmedKey);
            return (
              <div key={i} className="flex items-center gap-2">
                <Input
                  placeholder={keyPlaceholder}
                  value={row.key}
                  onChange={(e) => updateRow(i, "key", e.target.value)}
                  onBlur={() => handleBlur(i, "key")}
                  maxLength={MAX_KEY_LENGTH}
                  className={cn("flex-1", isDuplicate && "border-destructive focus-visible:ring-destructive")}
                />
                <Input
                  placeholder={valuePlaceholder}
                  value={row.value}
                  onChange={(e) => updateRow(i, "value", e.target.value)}
                  onBlur={() => handleBlur(i, "value")}
                  maxLength={MAX_VALUE_LENGTH}
                  className="flex-1"
                />
                <Button
                  type="button"
                  variant="outline"
                  size="icon"
                  className="shrink-0"
                  onClick={() => removeRow(i)}
                  title="Remove tag"
                >
                  <X className="h-4 w-4" />
                </Button>
              </div>
            );
          })}
        </div>
      )}
      {hasDuplicates && (
        <p className="text-sm text-destructive">
          Duplicate keys are not allowed. Later entries will overwrite earlier ones.
        </p>
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

/**
 * Generic dropdown where each option has a short caption under its label.
 *
 * Mirrors the model dropdown's look and feel: the open list shows the caption
 * per option (muted, smaller), while the closed trigger stays compact and
 * shows only the selected label. Use it for any simple captioned select.
 */

import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "./select";

export interface CaptionedOption {
  value: string;
  label: string;
  /** Shown under the label in the open list, not in the closed trigger. */
  caption?: string;
}

interface CaptionedSelectProps {
  value: string;
  onValueChange: (value: string) => void;
  options: readonly CaptionedOption[];
  placeholder?: string;
}

export function CaptionedSelect({
  value,
  onValueChange,
  options,
  placeholder,
}: CaptionedSelectProps) {
  const selected = options.find((o) => o.value === value);
  return (
    // key remounts on value change: inside a <form> Radix renders a hidden
    // native <select> whose <option>s only exist while the dropdown is open,
    // so a post-mount value set with the dropdown closed could coerce it to
    // "". Keying to the value avoids that (same guard as ModelSelect).
    <Select key={value} value={value} onValueChange={onValueChange}>
      <SelectTrigger>
        <SelectValue placeholder={placeholder}>
          {selected ? selected.label : null}
        </SelectValue>
      </SelectTrigger>
      <SelectContent>
        {options.map((o) => (
          <SelectItem key={o.value} value={o.value}>
            {o.label}
            {o.caption && (
              <>
                <br />
                <span className="text-xs text-muted-foreground">
                  {o.caption}
                </span>
              </>
            )}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  );
}

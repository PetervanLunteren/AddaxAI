/**
 * A list of radio rows, each with a label and a caption under it, the
 * radio on the left, as on every platform. The two-choice sibling of `CaptionedCheckbox`:
 * for a decision whose options need a sentence each, where a dropdown
 * hides the captions and a bare radio hides the consequences.
 */

export interface CaptionedRadioOption<T extends string> {
  value: T;
  label: string;
  caption: string;
}

export function CaptionedRadioGroup<T extends string>({
  name,
  value,
  onChange,
  options,
}: {
  name: string;
  value: T;
  onChange: (value: T) => void;
  options: CaptionedRadioOption<T>[];
}) {
  return (
    <div className="divide-y rounded-md border px-3">
      {options.map((opt) => (
        <label
          key={opt.value}
          className="flex cursor-pointer items-start gap-3 py-3 text-sm"
        >
          <input
            type="radio"
            name={name}
            className="mt-0.5 h-4 w-4 shrink-0 accent-primary"
            checked={value === opt.value}
            onChange={() => onChange(opt.value)}
          />
          <span>
            {opt.label}
            <span className="mt-0.5 block text-xs text-muted-foreground">
              {opt.caption}
            </span>
          </span>
        </label>
      ))}
    </div>
  );
}

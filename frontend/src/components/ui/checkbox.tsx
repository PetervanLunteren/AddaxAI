import * as React from "react";

interface CheckboxProps {
  checked: boolean;
  onCheckedChange: (checked: boolean) => void;
  indeterminate?: boolean;
  disabled?: boolean;
  className?: string;
}

export function Checkbox({
  checked,
  onCheckedChange,
  indeterminate,
  disabled = false,
  className,
}: CheckboxProps) {
  const ref = React.useRef<HTMLInputElement>(null);

  React.useEffect(() => {
    if (ref.current) {
      ref.current.indeterminate = indeterminate || false;
    }
  }, [indeterminate]);

  // Teal accent from the tokens: full = primary ink, half/indeterminate
  // = the lighter middle teal. Both adapt to dark via index.css.
  const checkboxStyle = indeterminate
    ? { accentColor: 'var(--middle)' }
    : { accentColor: 'var(--primary-ink)' };

  return (
    <input
      ref={ref}
      type="checkbox"
      checked={checked}
      disabled={disabled}
      onChange={(e) => onCheckedChange(e.target.checked)}
      style={checkboxStyle}
      className={`h-4 w-4 shrink-0 rounded border-input focus:ring-2 focus:ring-ring focus:ring-offset-2 focus:ring-offset-background disabled:cursor-not-allowed disabled:opacity-50 ${className || ""}`}
    />
  );
}

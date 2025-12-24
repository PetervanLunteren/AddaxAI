import * as React from "react";

interface CheckboxProps {
  checked: boolean;
  onCheckedChange: (checked: boolean) => void;
  indeterminate?: boolean;
  className?: string;
}

export function Checkbox({ checked, onCheckedChange, indeterminate, className }: CheckboxProps) {
  const ref = React.useRef<HTMLInputElement>(null);

  React.useEffect(() => {
    if (ref.current) {
      ref.current.indeterminate = indeterminate || false;
    }
  }, [indeterminate]);

  // Teal colors: #0f6064 (primary - full), #8bb3b4 (lighter - half/indeterminate)
  const checkboxStyle = indeterminate
    ? { accentColor: '#8bb3b4' }
    : { accentColor: '#0f6064' };

  return (
    <input
      ref={ref}
      type="checkbox"
      checked={checked}
      onChange={(e) => onCheckedChange(e.target.checked)}
      style={checkboxStyle}
      className={`h-4 w-4 rounded border-gray-300 focus:ring-2 focus:ring-[#0f6064] focus:ring-offset-2 ${className || ""}`}
    />
  );
}

/**
 * Progress component - simple progress bar for loading states
 */

import * as React from "react";
import { cn } from "../../lib/utils";

interface ProgressProps extends React.HTMLAttributes<HTMLDivElement> {
  value?: number;
  barColor?: string;
}

const Progress = React.forwardRef<HTMLDivElement, ProgressProps>(
  ({ className, value = 0, barColor, ...props }, ref) => {
    return (
      <div
        ref={ref}
        className={cn(
          "relative h-4 w-full overflow-hidden rounded-full bg-secondary",
          className
        )}
        {...props}
      >
        <div
          className="h-full w-full flex-1 bg-primary transition-all duration-500 ease-out"
          style={{
            transform: `translateX(-${100 - (value || 0)}%)`,
            ...(barColor ? { backgroundColor: barColor } : {}),
          }}
        />
      </div>
    );
  }
);
Progress.displayName = "Progress";

export { Progress };

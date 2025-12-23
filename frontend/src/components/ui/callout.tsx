/**
 * Callout - Info/Warning/Success/Error alert boxes
 *
 * Inspired by streamlit-AddaxAI's info_box and warning_box patterns
 */

import { Info, AlertTriangle, CheckCircle2, AlertCircle } from "lucide-react";
import { cn } from "@/lib/utils";

interface CalloutProps {
  children: React.ReactNode;
  variant?: "info" | "warning" | "success" | "error";
  title?: string;
  className?: string;
}

const variantConfig = {
  info: {
    bg: "bg-blue-50 border-blue-200",
    text: "text-blue-900",
    icon: Info,
    iconColor: "text-blue-600",
  },
  warning: {
    bg: "bg-yellow-50 border-yellow-200",
    text: "text-yellow-900",
    icon: AlertTriangle,
    iconColor: "text-yellow-600",
  },
  success: {
    bg: "bg-green-50 border-green-200",
    text: "text-green-900",
    icon: CheckCircle2,
    iconColor: "text-green-600",
  },
  error: {
    bg: "bg-red-50 border-red-200",
    text: "text-red-900",
    icon: AlertCircle,
    iconColor: "text-red-600",
  },
};

export function Callout({
  children,
  variant = "info",
  title,
  className,
}: CalloutProps) {
  const config = variantConfig[variant];
  const Icon = config.icon;

  return (
    <div
      className={cn(
        "p-4 rounded-lg border flex items-start gap-3",
        config.bg,
        className
      )}
    >
      <Icon className={cn("w-5 h-5 shrink-0 mt-0.5", config.iconColor)} />
      <div className="flex-1">
        {title && (
          <p className={cn("font-semibold mb-1", config.text)}>{title}</p>
        )}
        <div className={cn("text-sm", config.text)}>{children}</div>
      </div>
    </div>
  );
}

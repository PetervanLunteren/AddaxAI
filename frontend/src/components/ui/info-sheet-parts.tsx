/**
 * Shared primitives for the deployment / site info sheets.
 *
 * Both sheets use the same `Section` + `Row` layout under
 * `SheetHeader`, and a common "n/a" placeholder for optional values.
 * Extracted so the two components stay visually identical as they
 * evolve.
 */

import { Check, Copy } from "lucide-react";
import { useState } from "react";

export function Section({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <div className="space-y-2">
      <h3 className="text-sm font-semibold">{title}</h3>
      <dl className="space-y-1.5">{children}</dl>
    </div>
  );
}

export function Row({
  label,
  value,
}: {
  label: string;
  value: React.ReactNode;
}) {
  return (
    <div className="grid grid-cols-[minmax(10rem,auto)_1fr] gap-x-4 text-sm">
      <dt className="text-muted-foreground">{label}</dt>
      <dd className="break-words">{value}</dd>
    </div>
  );
}

export function NotSet() {
  return <span className="italic text-muted-foreground">n/a</span>;
}

/**
 * Monospace ID with a one-click copy button. The icon flips to a
 * checkmark for ~1.4 s on success; silent on failure (rare: non-HTTPS
 * contexts in some browsers).
 */
export function IdWithCopy({ value }: { value: string }) {
  const [justCopied, setJustCopied] = useState(false);
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(value);
      setJustCopied(true);
      setTimeout(() => setJustCopied(false), 1400);
    } catch {
      /* ignore; the missing checkmark is the user's cue */
    }
  };
  return (
    <div className="flex items-center gap-2">
      <code className="break-all rounded-sm bg-muted px-1.5 py-0.5 font-mono text-xs">
        {value}
      </code>
      <button
        type="button"
        onClick={copy}
        title="Copy ID"
        aria-label="Copy ID"
        className="inline-flex h-6 w-6 shrink-0 items-center justify-center rounded-sm text-muted-foreground hover:bg-accent hover:text-accent-foreground"
      >
        {justCopied ? (
          <Check className="h-3.5 w-3.5" />
        ) : (
          <Copy className="h-3.5 w-3.5" />
        )}
      </button>
    </div>
  );
}

/** Human-readable byte size. 0 renders as "0 B". */
export function formatBytes(bytes: number): string {
  if (bytes === 0) return "0 B";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let i = 0;
  let n = bytes;
  while (n >= 1024 && i < units.length - 1) {
    n /= 1024;
    i += 1;
  }
  return `${n.toFixed(n < 10 && i > 0 ? 1 : 0)} ${units[i]}`;
}

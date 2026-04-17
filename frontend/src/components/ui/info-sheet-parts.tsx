/**
 * Shared primitives for the deployment / site info sheets.
 *
 * Both sheets use the same `Section` + `Row` layout under
 * `SheetHeader`, and a common "n/a" placeholder for optional values.
 * Extracted so the two components stay visually identical as they
 * evolve.
 */

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

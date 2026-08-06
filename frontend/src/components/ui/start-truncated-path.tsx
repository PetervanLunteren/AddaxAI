/**
 * Path renderer that keeps the end of the path visible and puts the ellipsis
 * at the start when the container is too narrow.
 *
 * The tail of a path carries the meaningful bits (site, deployment name);
 * the leading `/Users/peter/...` prefix is noise that repeats across every
 * row. CSS `direction: rtl` flips the overflow side so ellipsis sits on the
 * left; a nested `<bdi>` keeps the text itself rendering left-to-right.
 * Full path is in the hover title for when the user needs the prefix.
 */
export function StartTruncatedPath({
  path,
  className,
  emptyLabel,
}: {
  path: string | null;
  className?: string;
  emptyLabel?: string;
}) {
  if (!path) {
    return (
      <span className={`italic text-muted-foreground ${className ?? ""}`}>
        {emptyLabel ?? ""}
      </span>
    );
  }
  return (
    <span
      className={`truncate block ${className ?? ""}`}
      style={{ direction: "rtl", textAlign: "left" }}
      title={path}
    >
      <bdi>{path}</bdi>
    </span>
  );
}

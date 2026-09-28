import type { TaxonomyNode } from "../api/types";

const UNMAPPED_LABEL_PREFIX = "unmapped:";

/** Decode the API's comma-safe ID for an unmapped label, if present. */
export function unmappedLabelFromFilterId(id: string): string | null {
  if (!id.startsWith(UNMAPPED_LABEL_PREFIX)) return null;
  const body = id.slice(UNMAPPED_LABEL_PREFIX.length);
  if (!/^[A-Za-z0-9_-]+$/.test(body)) return null;

  try {
    const base64 = body.replace(/-/g, "+").replace(/_/g, "/");
    const padded = base64 + "=".repeat((4 - (base64.length % 4)) % 4);
    const bytes = Uint8Array.from(atob(padded), (character) =>
      character.charCodeAt(0),
    );
    const label = new TextDecoder("utf-8", { fatal: true }).decode(bytes);
    return label || null;
  } catch {
    return null;
  }
}

/** Human-readable chip label; UUID mapping wins, encoded raw names are decoded. */
export function labelFilterDisplayName(
  id: string,
  displayLabels?: Record<string, string>,
): string {
  const unmapped = unmappedLabelFromFilterId(id);
  const displayName = displayLabels?.[id];
  if (unmapped) {
    const name = displayName ?? unmapped;
    return name.endsWith(" (unmapped)") ? name : `${name} (unmapped)`;
  }
  return displayName ?? id;
}

export function labelTreeDisplayNames(
  nodes: TaxonomyNode[] | undefined,
): Record<string, string> {
  const names: Record<string, string> = {};
  const visit = (items: TaxonomyNode[]) => {
    for (const node of items) {
      names[node.id] = node.name;
      visit(node.children);
    }
  };
  visit(nodes ?? []);
  return names;
}

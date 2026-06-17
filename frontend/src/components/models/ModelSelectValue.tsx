/**
 * Single-line value shown inside a model Select trigger: emoji + friendly name.
 *
 * The descriptive caption (description_short) is intentionally omitted here so
 * the collapsed trigger stays one line. Captions remain in the open dropdown
 * list (see ClassificationModelGroupedItems and the per-item descriptions), so
 * users still see them while browsing. Used by every model dropdown
 * (detection / classification / embedding) so they all look the same.
 */

interface ModelSelectValueProps {
  model: { emoji?: string | null; friendly_name: string };
}

export function ModelSelectValue({ model }: ModelSelectValueProps) {
  return (
    <span className="truncate">
      {model.emoji ? `${model.emoji} ` : ""}
      {model.friendly_name}
    </span>
  );
}

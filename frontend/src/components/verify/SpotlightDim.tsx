/**
 * SpotlightDim — darkens everything OUTSIDE the union of the given boxes,
 * leaving the boxes (and any overlap between them) fully bright.
 *
 * Uses an SVG mask: a white field with black box-shaped holes. Overlapping
 * black holes simply union, so the bright region is always the union of the
 * boxes. (An evenodd "outer rect minus holes" path re-fills the overlap of
 * two boxes — the bug this replaces.) Render inside an <svg> whose viewBox is
 * `0 0 width height`; boxes are in those same pixel coordinates.
 */

import { useId } from "react";

interface SpotlightBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

interface SpotlightDimProps {
  width: number;
  height: number;
  boxes: SpotlightBox[];
  /** Corner radius (px) for the bright holes. */
  rx: number;
  /** Dim fill, e.g. "rgba(0,0,0,0.55)". */
  fill: string;
}

export function SpotlightDim({ width, height, boxes, rx, fill }: SpotlightDimProps) {
  // useId can contain ":", which is fine in HTML ids but trips up url(#...)
  // resolution in some engines; strip it.
  const maskId = `spotlight-${useId().replace(/:/g, "")}`;
  return (
    <>
      <defs>
        <mask id={maskId}>
          <rect width={width} height={height} fill="white" />
          {boxes.map((b, i) => (
            <rect
              key={i}
              x={b.x}
              y={b.y}
              width={b.width}
              height={b.height}
              rx={rx}
              fill="black"
            />
          ))}
        </mask>
      </defs>
      <rect width={width} height={height} fill={fill} mask={`url(#${maskId})`} />
    </>
  );
}

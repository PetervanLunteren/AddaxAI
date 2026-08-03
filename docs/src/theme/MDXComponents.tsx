/**
 * Register table-row anchors with Docusaurus's broken-link checker.
 *
 * `src/remark/table-row-anchors.ts` puts an `id` on every column row so a
 * single column can be linked. Docusaurus does not discover anchors by
 * scanning the built HTML: they are collected at render time, and only by
 * components that call `collectAnchor` (headings and `<Link id>` do).
 *
 * Without this, all 59 links the plugin generates are reported as broken
 * anchors on every build. Setting `onBrokenAnchors: 'ignore'` would silence
 * them, but it would silence the real ones too, and this page is exactly
 * the kind that accumulates stale links.
 *
 * So the row registers itself, the same way `<Details>` does in
 * theme-common. Rows with no `id` (header rows, prose tables) are untouched.
 */

import React from "react";
import MDXComponents from "@theme-original/MDXComponents";
import useBrokenLinks from "@docusaurus/useBrokenLinks";

function TableRow(props: React.ComponentProps<"tr">): React.ReactElement {
  const brokenLinks = useBrokenLinks();
  if (props.id) {
    brokenLinks.collectAnchor(props.id);
  }
  return <tr {...props} />;
}

export default {
  ...MDXComponents,
  tr: TableRow,
};

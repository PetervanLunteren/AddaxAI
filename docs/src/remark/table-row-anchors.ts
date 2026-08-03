/**
 * Give every table row its own link, so a single column can be shared.
 *
 * Docusaurus already anchors headings, which gets you as far as the table
 * ("...#files"). This adds the level below: each row of a column table
 * becomes "...#files-classification_label".
 *
 * The anchor is scoped to the enclosing `##` heading on purpose. Eight
 * column names appear in more than one table on the exports page
 * (`deployment_id` is in all four), so an unscoped `#deployment_id` would
 * be a duplicate id and only ever reach the first table.
 *
 * A row is anchored when its first cell starts with inline code, which is
 * how every column table is written. That skips header and separator rows
 * without needing to know anything about them, and it leaves prose tables
 * with a plain-text first column alone rather than inventing anchors from
 * a sentence.
 *
 * The injected link reuses Infima's `.hash-link` class, so it inherits the
 * "#" glyph and the fade-in that headings already use. `custom.css` adds
 * the row-hover rule and the `:target` highlight.
 *
 * No dependency on `unist-util-visit`: the walk is eight lines and this
 * way the plugin does not lean on a package we only get transitively.
 */

type Node = {
  type: string;
  value?: string;
  depth?: number;
  url?: string;
  children?: Node[];
  data?: { hProperties?: Record<string, unknown> };
};

/** Lowercase, non-word runs to single hyphens, no leading/trailing hyphen. */
function slug(text: string): string {
  return text
    .toLowerCase()
    .replace(/[^\w]+/g, "-")
    .replace(/^-+|-+$/g, "");
}

/** Concatenate the text of a node tree, for heading titles. */
function textOf(node: Node): string {
  if (typeof node.value === "string") return node.value;
  return (node.children ?? []).map(textOf).join("");
}

/**
 * The column name a row is about: the first cell's leading inline code.
 * Returns null for header rows and for tables whose first column is prose.
 */
function columnName(row: Node): string | null {
  const firstCell = row.children?.[0];
  const firstChild = firstCell?.children?.[0];
  if (firstChild?.type !== "inlineCode") return null;
  const value = firstChild.value?.trim();
  return value ? value : null;
}

export default function tableRowAnchors() {
  return (tree: Node): void => {
    let section = "";
    const used = new Set<string>();

    const walk = (node: Node): void => {
      // Depth-first in document order, so `section` is always the heading
      // the row actually sits under.
      if (node.type === "heading" && node.depth === 2) {
        section = slug(textOf(node));
      }

      if (node.type === "tableRow") {
        const name = columnName(node);
        if (name) {
          const base = section ? `${section}-${slug(name)}` : slug(name);
          // Same column twice under one heading would still collide; suffix
          // rather than silently drop the second one.
          let id = base;
          for (let n = 2; used.has(id); n += 1) id = `${base}-${n}`;
          used.add(id);

          node.data = { ...node.data, hProperties: { ...node.data?.hProperties, id } };

          // The link lives at the end of the first cell, after the column
          // name, which is where a heading's hash link sits too.
          node.children?.[0]?.children?.push({
            type: "link",
            url: `#${id}`,
            data: {
              hProperties: {
                class: "hash-link",
                "aria-label": `Direct link to ${name}`,
                title: `Direct link to ${name}`,
              },
            },
            children: [],
          });
        }
      }

      for (const child of node.children ?? []) walk(child);
    };

    walk(tree);
  };
}

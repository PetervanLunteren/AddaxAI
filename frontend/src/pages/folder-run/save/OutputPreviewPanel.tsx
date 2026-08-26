/**
 * Live folder-tree preview for the Save outputs step.
 *
 * The user ticks options on the left; this panel renders what the
 * actual run will produce on disk: which subfolders appear, how many
 * files land in each, and the cumulative byte estimate.
 *
 * Numbers come from the project DB via `/api/folder-runs/{id}/output-
 * preview`. They are exact placement counts, not heuristics, because
 * the placement rules are deterministic given the file rows and the
 * project threshold. The preview matches what the postprocess run
 * writes.
 *
 * Byte totals are summed from `File.size_bytes` where present; the
 * summary line notes when the estimate is partial (some files have
 * NULL size, typically because they were ingested before that
 * column was populated).
 */

import { useMemo } from "react";
import { FileText, Folder } from "lucide-react";

import { Card, CardContent } from "../../../components/ui/card";
import type { OutputPreview } from "../../../api/folder-runs";
import type { UseSaveOutputsFormResult } from "./useSaveOutputsForm";

const RUN_NAME_FALLBACK = "results";

interface SubFolder {
  name: string;
  /** Optional count to render to the right of the folder name. */
  count?: number;
  /** Children rendered indented under this folder. */
  children?: SubFolder[];
}

interface FileEntry {
  name: string;
}

export function OutputPreviewPanel({
  form,
  preview,
  runName,
  isLoading,
}: {
  form: UseSaveOutputsFormResult;
  preview: OutputPreview | undefined;
  runName: string;
  isLoading: boolean;
}) {
  const { separate, visualise, anonymise, exportOpts } = form;

  const tree = useMemo<{ folders: SubFolder[]; files: FileEntry[] }>(
    () => buildTree({ preview, separate, exportOpts }),
    [preview, separate, exportOpts],
  );

  const anyPicked =
    separate.enabled ||
    visualise.enabled ||
    anonymise.enabled ||
    (exportOpts.enabled &&
      (exportOpts.spreadsheet ||
        exportOpts.recognitionJson ||
        exportOpts.summary));

  // Tree root is the folder the user is writing to (the last segment of
  // the output path), not the project name — that's the folder these
  // files actually land in. Falls back to the run name, then a literal.
  const outputBasename = form.effectiveOutputDir
    .replace(/[\\/]+$/, "")
    .split(/[\\/]/)
    .pop();
  const treeRoot = outputBasename || runName || RUN_NAME_FALLBACK;

  return (
    // Cap the card at the viewport height (sticky 1.5rem below the top, a
    // little breathing room at the bottom) and lay it out as a flex column,
    // so the tree area flexes to whatever height is left and scrolls inside.
    // A short tree still shrinks the card to its content; a long one fills
    // the screen. Adapts to large and small displays without a fixed height.
    <Card className="sticky top-6 flex max-h-[calc(100vh-3rem)] flex-col">
      <CardContent className="flex min-h-0 flex-1 flex-col space-y-4 p-6">
        <div>
          <h3 className="text-sm font-semibold">Output preview</h3>
          <p className="mt-0.5 text-xs text-muted-foreground">
            What the run will write into your output folder
          </p>
        </div>

        {isLoading && !preview ? (
          <Placeholder>Loading file counts...</Placeholder>
        ) : !preview ? (
          <Placeholder tone="error">
            Could not load the run's file counts.
          </Placeholder>
        ) : !anyPicked ? (
          <Placeholder>
            Pick at least one output on the left to see the preview
          </Placeholder>
        ) : (
          <div className="flex min-h-0 flex-1 flex-col overflow-hidden rounded-md border bg-muted/30">
            <TreeView
              runName={treeRoot}
              folders={tree.folders}
              files={tree.files}
            />
            <SummaryFooter
              preview={preview}
              form={form}
            />
          </div>
        )}
      </CardContent>
    </Card>
  );
}

/** Centred message in the same framed box the populated preview uses,
 * so the card keeps one consistent framed area in every state. */
function Placeholder({
  children,
  tone = "muted",
}: {
  children: React.ReactNode;
  tone?: "muted" | "error";
}) {
  return (
    <div
      className={`rounded-md border bg-muted/30 p-6 text-center text-xs ${
        tone === "error" ? "text-destructive" : "text-muted-foreground"
      }`}
    >
      {children}
    </div>
  );
}

function buildTree({
  preview,
  separate,
  exportOpts,
}: {
  preview: OutputPreview | undefined;
  separate: UseSaveOutputsFormResult["separate"];
  exportOpts: UseSaveOutputsFormResult["exportOpts"];
}): { folders: SubFolder[]; files: FileEntry[] } {
  const folders: SubFolder[] = [];
  const files: FileEntry[] = [];

  // Media copies, all under the addaxai-media subfolder.
  // ``by_media_tree`` is the real on-disk tree for the chosen layout:
  // species / observation folders and the preserved source subfolders
  // combined in the chosen order (or just the source subfolders under
  // "No subfolders"). Loose files with no folder at all ("No
  // subfolders" with source-root files) fall back to a capped filename
  // list inside addaxai-media.
  if (separate.enabled && preview) {
    const mediaChildren: SubFolder[] = nestedFoldersFromPaths(
      preview.by_media_tree,
    );
    const inFolders = Object.values(preview.by_media_tree).reduce(
      (a, n) => a + n,
      0,
    );
    const rootTotal = preview.in_scope_files - inFolders;
    if (rootTotal > 0) {
      const shown =
        rootTotal <= MAX_CHILDREN_PER_LEVEL
          ? preview.root_files
          : preview.root_files.slice(0, MAX_CHILDREN_PER_LEVEL - 1);
      for (const name of shown) mediaChildren.push({ name });
      if (rootTotal > shown.length) {
        mediaChildren.push({
          name: `… ${(rootTotal - shown.length).toLocaleString()} more`,
        });
      }
    }
    folders.push({ name: "addaxai-media/", children: mediaChildren });
  }

  if (exportOpts.enabled) {
    if (exportOpts.spreadsheet) {
      if (exportOpts.format === "csv") {
        files.push({ name: "addaxai-summary.csv" });
        files.push({ name: "addaxai-files.csv" });
        files.push({ name: "addaxai-detections.csv" });
      } else {
        files.push({ name: "addaxai-spreadsheet.xlsx" });
      }
    }
    if (exportOpts.recognitionJson)
      files.push({ name: "addaxai-recognitions.json" });
    if (exportOpts.summary) files.push({ name: "addaxai-run-info.txt" });
  }

  return { folders, files };
}

/** Turn a flat map of slash-paths into a nested SubFolder tree.
 *
 * The backend's ``by_media_tree`` is keyed by full destination paths like
 * ``mammalia/carnivora/canidae/canis/dog/cam01`` (or ``cam01/dog`` under
 * species-last); each value is the leaf placement count. We parse into a
 * tree, then roll up counts so internal nodes show the cumulative total
 * of their subtree. */
function nestedFoldersFromPaths(
  paths: Record<string, number>,
): SubFolder[] {
  const root: SubFolder[] = [];
  for (const [path, count] of Object.entries(paths)) {
    const parts = path.split("/").filter((p) => p.length > 0);
    if (parts.length === 0) continue;
    let level = root;
    for (let i = 0; i < parts.length; i++) {
      const part = parts[i];
      let node = level.find((n) => n.name === `${part}/`);
      if (!node) {
        node = { name: `${part}/`, count: 0, children: [] };
        level.push(node);
      }
      if (i === parts.length - 1) {
        node.count = (node.count ?? 0) + count;
      }
      level = node.children!;
    }
  }
  // Roll up internal node counts from leaves.
  rollUpCounts(root);
  // Sort each level by count desc so dominant taxa surface first.
  sortLevels(root);
  return root;
}

function rollUpCounts(nodes: SubFolder[]): number {
  let total = 0;
  for (const node of nodes) {
    if (node.children && node.children.length > 0) {
      node.count = rollUpCounts(node.children);
    }
    total += node.count ?? 0;
  }
  return total;
}

function sortLevels(nodes: SubFolder[]): void {
  nodes.sort((a, b) => (b.count ?? 0) - (a.count ?? 0));
  for (const node of nodes) {
    if (node.children && node.children.length > 0) {
      sortLevels(node.children);
    }
  }
}

// Per-level cap on visible children. When a folder has more than
// MAX_CHILDREN_PER_LEVEL children, the first (cap - 1) are shown,
// followed by a "…" row at that same level. The marker tells the
// user "more at this level" without enumerating; the bias is to
// show the tree's shape, not every leaf.
const MAX_CHILDREN_PER_LEVEL = 4;

// Global cap on folder rows. Even with the per-level cap, a deep
// taxonomic layout (class > order > family > genus > species >
// deployment) across many branches stacks up into a very tall tree.
// The preview only needs to convey the grouping shape, so we stop
// after this many folder rows and drop a single "… more folders"
// marker. The export / summary files always render after it (they're
// short and the user wants to see them), and the box itself scrolls.
const MAX_TREE_ROWS = 40;

function TreeView({
  runName,
  folders,
  files,
}: {
  runName: string;
  folders: SubFolder[];
  files: FileEntry[];
}) {
  // Recursive flattener: walks an arbitrary-depth nested tree and
  // produces one prefixed entry per line. Each level's `isLast`
  // controls whether the leading branch char is `└─` vs `├─` and
  // whether descendants get `   ` (under a last child) vs `│  `
  // (under a continuing sibling).
  interface Entry {
    prefix: string;
    name: string;
    count?: number;
  }

  // Count only on leaf folders — the ones files actually land in. Every
  // level above (station, deployment, taxonomy chain) is just a container;
  // showing its rolled-up total repeats the same number down the tree and
  // reads as if files live at each level. The grand total is in the footer.
  function countFor(node: SubFolder): number | undefined {
    const isLeaf = (node.children?.length ?? 0) === 0;
    return isLeaf ? node.count : undefined;
  }

  const entries: Entry[] = [];
  entries.push({ prefix: "", name: `${runName}/` });

  // Loose data files (CSV / XLSX / JSON / summary) render at the top, then
  // the media tree below — the small, always-there outputs are easier to
  // scan up top, and the media tree (which can be deep and capped) sits
  // last. Only folder rows are budgeted by the global cap; the truncation
  // marker is emitted inside the walk at the level it runs out, so it stays
  // correctly indented within the media subtree.
  let folderRows = 0;
  let folderTruncated = false;

  function walkFolder(node: SubFolder, ancestorPrefix: string): void {
    const allChildren = node.children ?? [];
    const truncated = allChildren.length > MAX_CHILDREN_PER_LEVEL;
    const visibleCount = truncated
      ? MAX_CHILDREN_PER_LEVEL - 1
      : allChildren.length;
    const visible = allChildren.slice(0, visibleCount);

    for (let i = 0; i < visible.length; i++) {
      if (folderRows >= MAX_TREE_ROWS) {
        entries.push({ prefix: ancestorPrefix + "└─ ", name: "… more folders" });
        folderTruncated = true;
        return;
      }
      const child = visible[i];
      const isLastEntryAtThisLevel = !truncated && i === visible.length - 1;
      const branch = isLastEntryAtThisLevel ? "└─ " : "├─ ";
      const descend = isLastEntryAtThisLevel ? "   " : "│  ";
      entries.push({
        prefix: ancestorPrefix + branch,
        name: child.name,
        count: countFor(child),
      });
      folderRows++;
      if (child.children && child.children.length > 0) {
        walkFolder(child, ancestorPrefix + descend);
        if (folderTruncated) return;
      }
    }

    if (truncated) {
      entries.push({ prefix: ancestorPrefix + "└─ ", name: "…" });
    }
  }

  const hasFolders = folders.length > 0;

  // Files at the top. They're last-overall only when nothing follows them
  // (separation off, so no media tree).
  files.forEach((file, i) => {
    const isLastOverall = !hasFolders && i === files.length - 1;
    entries.push({ prefix: isLastOverall ? "└─ " : "├─ ", name: file.name });
  });

  // Media tree below the files.
  folders.forEach((folder, i) => {
    if (folderTruncated) return;
    if (folderRows >= MAX_TREE_ROWS) {
      entries.push({ prefix: "└─ ", name: "… more folders" });
      folderTruncated = true;
      return;
    }
    const isLast = i === folders.length - 1;
    entries.push({
      prefix: isLast ? "└─ " : "├─ ",
      name: folder.name,
      count: countFor(folder),
    });
    folderRows++;
    walkFolder(folder, isLast ? "   " : "│  ");
  });

  return (
    <pre className="min-h-0 flex-1 overflow-y-auto overflow-x-hidden p-3 font-mono text-[11px] leading-relaxed text-foreground">
      {entries.map((entry, idx) => (
        <TreeRow key={idx} {...entry} />
      ))}
    </pre>
  );
}

function TreeRow({
  prefix,
  name,
  count,
}: {
  prefix: string;
  name: string;
  count?: number;
}) {
  const countLabel = count !== undefined ? count.toLocaleString() : "";
  // The "… N more" placeholder is neither a folder nor a file.
  const isMarker = name.startsWith("…");
  const isFolder = name.endsWith("/");
  // The icon conveys folder-ness, so drop the trailing slash from display.
  const display = isFolder ? name.replace(/\/$/, "") : name;

  return (
    <div className="flex items-center gap-3">
      <span className="flex min-w-0 flex-1 items-center">
        <span className="shrink-0 whitespace-pre">{prefix}</span>
        {/* Fixed-width icon slot (empty for markers) so names stay aligned. */}
        <span className="mr-1.5 inline-flex h-3.5 w-3.5 shrink-0 items-center justify-center">
          {!isMarker &&
            (isFolder ? (
              <Folder className="h-3.5 w-3.5 text-primary/70" />
            ) : (
              <FileText className="h-3.5 w-3.5 text-muted-foreground" />
            ))}
        </span>
        {/* Long names are cut off with an ellipsis rather than overflowing;
            the title exposes the full name on hover (deep nesting leaves
            little room, so leaf names truncate early). */}
        <span className="min-w-0 truncate" title={isMarker ? undefined : display}>
          {display}
        </span>
      </span>
      {countLabel && (
        <span className="shrink-0 text-muted-foreground">{countLabel}</span>
      )}
    </div>
  );
}

function SummaryFooter({
  preview,
  form,
}: {
  preview: OutputPreview;
  form: UseSaveOutputsFormResult;
}) {
  const { separate } = form;

  // Each in-scope file lands in exactly one folder, so the number
  // written equals the in-scope count. Boxes / blur overwrite those
  // same copies in place (and only run when separation is on), so they
  // never add files. Exports + README are small data files, not counted.
  const writtenTotal = separate.enabled ? preview.in_scope_files : 0;
  // Videos are written as their best-frame JPEG, which in_scope_bytes
  // already reflects, so this is the real media footprint on disk.
  const estimatedBytes = separate.enabled ? preview.in_scope_bytes : 0;
  const partialSize =
    preview.files_with_known_size > 0 &&
    preview.files_with_known_size < preview.total_files;

  // Account for the whole source → written gap so the numbers add up:
  // species-filtered files, plus empty captures skipped when "copy
  // empties" is off (everything not filtered and not written is empty).
  const filtered = preview.dropped_by_filter;
  const emptySkipped =
    preview.total_files - filtered - preview.in_scope_files;
  const reasons: string[] = [];
  if (filtered > 0) reasons.push(`${filtered.toLocaleString()} filtered`);
  if (emptySkipped > 0) {
    reasons.push(`${emptySkipped.toLocaleString()} empty`);
  }

  return (
    <div className="space-y-1 border-t p-3 text-xs">
      <p className="font-medium text-foreground">
        {preview.total_files.toLocaleString()} source{" "}
        {preview.total_files === 1 ? "file" : "files"}
        {separate.enabled && (
          <>
            {" → "}
            {writtenTotal.toLocaleString()} written
            {reasons.length > 0 && (
              <span className="font-normal text-muted-foreground">
                {" "}
                ({reasons.join(", ")})
              </span>
            )}
          </>
        )}
      </p>
      {estimatedBytes > 0 && (
        <p className="text-muted-foreground">
          {"~"}
          {formatBytes(estimatedBytes)}
          {partialSize && (
            <span className="ml-1">(estimate, partial size data)</span>
          )}
        </p>
      )}
    </div>
  );
}

// Binary units (1024-based) with SI-style labels. Each unit rolls over
// to the next at ~1000 rather than 1024, so the user never sees an
// awkward "1024 KB" / "1024 MB"; it becomes "1.0 MB" / "1.0 GB". B and
// KB are whole; MB and GB carry one decimal.
function formatBytes(n: number): string {
  if (n < 1024) return `${Math.round(n)} B`;
  const kb = n / 1024;
  if (kb < 999.5) return `${Math.round(kb)} KB`;
  const mb = kb / 1024;
  if (mb < 999.95) return `${mb.toFixed(1)} MB`;
  return `${(mb / 1024).toFixed(1)} GB`;
}

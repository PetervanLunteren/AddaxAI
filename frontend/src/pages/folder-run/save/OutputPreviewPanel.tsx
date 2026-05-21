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
  /** Files vs photos copies vs raw count - controls the trailing
   * unit on the count. */
  unit?: "files" | "images" | "items";
}

interface FileEntry {
  name: string;
  hint?: string;
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
    () => buildTree({ preview, separate, visualise, anonymise, exportOpts }),
    [preview, separate, visualise, anonymise, exportOpts],
  );

  const anyPicked =
    separate.enabled ||
    visualise.enabled ||
    anonymise.enabled ||
    (exportOpts.enabled &&
      (exportOpts.csv || exportOpts.xlsx || exportOpts.recognitionJson));

  const safeRunName = runName || RUN_NAME_FALLBACK;

  return (
    <Card className="sticky top-6">
      <CardContent className="space-y-4 p-6">
        <div>
          <h3 className="text-sm font-semibold">Output preview</h3>
          <p className="mt-1 text-xs text-muted-foreground">
            What the run will write into your output folder.
          </p>
        </div>

        {isLoading && !preview ? (
          <p className="text-xs text-muted-foreground">
            Loading file counts...
          </p>
        ) : !preview ? (
          <p className="text-xs text-destructive">
            Could not load the run's file counts.
          </p>
        ) : !anyPicked ? (
          <p className="text-xs text-muted-foreground">
            Pick at least one output on the left to see the preview.
          </p>
        ) : (
          <>
            <TreeView
              runName={safeRunName}
              folders={tree.folders}
              files={tree.files}
            />
            <SummaryFooter
              preview={preview}
              form={form}
              folderCount={tree.folders.length}
            />
          </>
        )}
      </CardContent>
    </Card>
  );
}

function buildTree({
  preview,
  separate,
  visualise,
  anonymise,
  exportOpts,
}: {
  preview: OutputPreview | undefined;
  separate: UseSaveOutputsFormResult["separate"];
  visualise: UseSaveOutputsFormResult["visualise"];
  anonymise: UseSaveOutputsFormResult["anonymise"];
  exportOpts: UseSaveOutputsFormResult["exportOpts"];
}): { folders: SubFolder[]; files: FileEntry[] } {
  const folders: SubFolder[] = [];
  const files: FileEntry[] = [];

  // Media copies. "taxonomic" / "flat" lay them out in per-label
  // subfolders at the output root (counts from the preview); "none"
  // drops them flat at the root, surfaced as a single line. Boxes /
  // blur render onto these same copies, so they add no separate tree.
  if (separate.enabled && preview) {
    if (separate.groupBy === "taxonomic") {
      folders.push(...nestedFoldersFromPaths(preview.by_taxonomic_tree));
    } else if (separate.groupBy === "flat") {
      folders.push(...flatFoldersFromMap(preview.by_flat));
    } else {
      // Flat copy: one file each. in_scope_files already reflects the
      // empties skip, so this line tracks the "copy empties" toggle.
      const count = preview.in_scope_files;
      const annotated = visualise.enabled || anonymise.enabled;
      files.push({
        name: `${count.toLocaleString()} media file${
          count === 1 ? "" : "s"
        }`,
        hint: annotated
          ? annotatedHint(visualise.enabled, anonymise.enabled)
          : "copied",
      });
    }
  }

  if (exportOpts.enabled) {
    if (exportOpts.csv) files.push({ name: "observations.csv" });
    if (exportOpts.xlsx) files.push({ name: "observations.xlsx" });
    if (exportOpts.recognitionJson)
      files.push({ name: "timelapse_recognition_file.json" });
  }

  // README is always written.
  files.push({ name: "README.txt" });

  return { folders, files };
}

function annotatedHint(visualise: boolean, anonymise: boolean): string {
  if (visualise && anonymise) return "boxes drawn + people blurred";
  if (visualise) return "boxes drawn";
  return "people blurred";
}

/** Turn a flat map of slash-paths into a nested SubFolder tree.
 *
 * The backend's ``by_taxonomic_tree`` is keyed by paths like
 * ``Mammalia/Carnivora/Canidae/Canis/dog``; each value is the
 * leaf placement count. We parse into a tree, then roll up counts
 * so internal nodes show the cumulative total of their subtree. */
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
        node = { name: `${part}/`, count: 0, unit: "items", children: [] };
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

/** Turn a flat ``label -> count`` map into the leaf SubFolders for
 * the "Flat by species" mode. Sorted by count desc, capped at 20
 * entries so very-many-species runs don't blow up the preview. */
function flatFoldersFromMap(
  buckets: Record<string, number>,
): SubFolder[] {
  return Object.entries(buckets)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 20)
    .map(([name, count]) => ({
      name: `${name}/`,
      count,
      unit: "items" as const,
    }));
}

// Per-level cap on visible children. When a folder has more than
// MAX_CHILDREN_PER_LEVEL children, the first (cap - 1) are shown,
// followed by a "…" row at that same level. The marker tells the
// user "more at this level" without enumerating; the bias is to
// show the tree's shape, not every leaf.
const MAX_CHILDREN_PER_LEVEL = 4;

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
    unit?: SubFolder["unit"];
    hint?: string;
  }

  const entries: Entry[] = [];
  entries.push({ prefix: "", name: `${runName}/` });

  type TopItem =
    | { kind: "folder"; folder: SubFolder }
    | { kind: "file"; file: FileEntry };

  const topLevel: TopItem[] = [
    ...folders.map((f): TopItem => ({ kind: "folder", folder: f })),
    ...files.map((f): TopItem => ({ kind: "file", file: f })),
  ];

  function walkFolder(node: SubFolder, ancestorPrefix: string): void {
    const allChildren = node.children ?? [];
    const truncated = allChildren.length > MAX_CHILDREN_PER_LEVEL;
    const visibleCount = truncated
      ? MAX_CHILDREN_PER_LEVEL - 1
      : allChildren.length;
    const visible = allChildren.slice(0, visibleCount);

    visible.forEach((child, i) => {
      const isLastEntryAtThisLevel =
        !truncated && i === visible.length - 1;
      const branch = isLastEntryAtThisLevel ? "└─ " : "├─ ";
      const descend = isLastEntryAtThisLevel ? "   " : "│  ";
      entries.push({
        prefix: ancestorPrefix + branch,
        name: child.name,
        count: child.count,
        unit: child.unit,
      });
      if (child.children && child.children.length > 0) {
        walkFolder(child, ancestorPrefix + descend);
      }
    });

    if (truncated) {
      entries.push({
        prefix: ancestorPrefix + "└─ ",
        name: "…",
      });
    }
  }

  topLevel.forEach((item, i) => {
    const isLast = i === topLevel.length - 1;
    const branch = isLast ? "└─ " : "├─ ";
    const descend = isLast ? "   " : "│  ";
    if (item.kind === "folder") {
      entries.push({
        prefix: branch,
        name: item.folder.name,
        count: item.folder.count,
        unit: item.folder.unit,
      });
      walkFolder(item.folder, descend);
    } else {
      entries.push({
        prefix: branch,
        name: item.file.name,
        hint: item.file.hint,
      });
    }
  });

  return (
    <pre className="overflow-x-auto rounded-md border bg-muted/30 p-3 font-mono text-[11px] leading-relaxed text-foreground">
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
  unit,
  hint,
}: {
  prefix: string;
  name: string;
  count?: number;
  unit?: SubFolder["unit"];
  hint?: string;
}) {
  const trailingUnit =
    unit === "files"
      ? count === 1
        ? "file"
        : "files"
      : unit === "images"
        ? count === 1
          ? "image"
          : "images"
        : "";

  const countLabel =
    count !== undefined
      ? trailingUnit
        ? `${count.toLocaleString()} ${trailingUnit}`
        : count.toLocaleString()
      : "";

  return (
    <div className="flex items-baseline justify-between gap-4">
      <span className="whitespace-pre">
        {prefix}
        {name}
        {hint && (
          <span className="ml-2 text-muted-foreground">— {hint}</span>
        )}
      </span>
      {countLabel && (
        <span className="text-muted-foreground">{countLabel}</span>
      )}
    </div>
  );
}

function SummaryFooter({
  preview,
  form,
  folderCount,
}: {
  preview: OutputPreview;
  form: UseSaveOutputsFormResult;
  folderCount: number;
}) {
  const { separate, visualise, anonymise } = form;
  const annotateOn = visualise.enabled || anonymise.enabled;
  const placementsPerSourceFile = countCopiesPerFile({
    separate,
    annotateOn,
  });

  // Total files written. Separation produces one placement per source
  // file × number of label folders it landed in (multi-species
  // inflates the bucket sum). When separation is also on, annotated
  // copies are written INTO the separated folders (same files,
  // overwritten with effects) — so they don't add to the placement
  // count. When separation is off, annotated copies write one image
  // per in-scope file at the root.
  const separatedSource =
    separate.groupBy === "taxonomic"
      ? preview.by_taxonomic_tree
      : preview.by_flat;
  const separatedPlacements = separate.enabled
    ? Object.values(separatedSource).reduce((a, n) => a + n, 0)
    : 0;

  const scopedImageCount =
    preview.dropped_by_filter > 0
      ? preview.in_scope_image_count
      : preview.image_count;
  const annotatedWritten =
    annotateOn && !separate.enabled ? scopedImageCount : 0;

  const writtenTotal = separatedPlacements + annotatedWritten;

  // Byte estimate uses in-scope average so exclusion-affected runs
  // give a number that matches what'll actually land on disk.
  const sizeReferenceFiles =
    preview.in_scope_files > 0
      ? preview.in_scope_files
      : preview.total_files;
  const sizeReferenceBytes =
    preview.in_scope_files > 0
      ? preview.in_scope_bytes
      : preview.total_bytes;
  const partialSize =
    preview.files_with_known_size > 0 &&
    preview.files_with_known_size < preview.total_files;
  const avgBytesPerFile =
    sizeReferenceFiles > 0 ? sizeReferenceBytes / sizeReferenceFiles : 0;
  const estimatedBytes = avgBytesPerFile * writtenTotal;

  return (
    <div className="space-y-1.5 rounded-md border bg-card-background p-3 text-xs">
      <p className="font-medium text-foreground">
        {preview.total_files.toLocaleString()} source{" "}
        {preview.total_files === 1 ? "file" : "files"}
        {preview.image_count > 0 && preview.video_count > 0 && (
          <span className="text-muted-foreground">
            {" "}
            ({preview.image_count.toLocaleString()} images,{" "}
            {preview.video_count.toLocaleString()} videos)
          </span>
        )}
      </p>
      {preview.dropped_by_filter > 0 && (
        <p className="text-muted-foreground">
          {preview.dropped_by_filter.toLocaleString()}{" "}
          {preview.dropped_by_filter === 1 ? "file" : "files"} skipped
          by the species filter,{" "}
          {preview.in_scope_files.toLocaleString()} in scope.
        </p>
      )}
      {writtenTotal > 0 && (
        <p className="text-muted-foreground">
          {writtenTotal.toLocaleString()}{" "}
          {writtenTotal === 1 ? "file" : "files"} will be written
          {placementsPerSourceFile > 1 && (
            <>
              {" "}
              ({"~"}
              {placementsPerSourceFile}× per source file)
            </>
          )}
          {folderCount > 0 && (
            <>
              {" into "}
              {folderCount} subfolder{folderCount === 1 ? "" : "s"}
            </>
          )}
          .
        </p>
      )}
      {separate.enabled && preview.multi_species_files > 0 && (
        <p className="text-muted-foreground">
          {preview.multi_species_files.toLocaleString()}{" "}
          {preview.multi_species_files === 1 ? "file" : "files"} appear
          in more than one leaf folder (multi-species shots).
        </p>
      )}
      {avgBytesPerFile > 0 && writtenTotal > 0 && (
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

function countCopiesPerFile({
  separate,
  annotateOn,
}: {
  separate: UseSaveOutputsFormResult["separate"];
  annotateOn: boolean;
}): number {
  // Separation and annotation share the same on-disk file when both
  // are on (annotation mutates the separated copy in place), so they
  // do not multiply with each other.
  if (separate.enabled) return 1;
  if (annotateOn) return 1;
  return 0;
}

function formatBytes(n: number): string {
  if (n < 1024) return `${Math.round(n)} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(0)} KB`;
  if (n < 1024 * 1024 * 1024)
    return `${(n / (1024 * 1024)).toFixed(0)} MB`;
  return `${(n / (1024 * 1024 * 1024)).toFixed(1)} GB`;
}

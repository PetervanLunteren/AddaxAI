/**
 * Reusable building blocks for the four Save-outputs layout variants.
 *
 * Each variant mounts these the same way; the only difference is the
 * container chrome (tabs vs accordion vs flat list).
 */

import { useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  AlertTriangle,
  ArrowLeft,
  CheckCircle2,
  FolderOpen,
  Save,
} from "lucide-react";

import { Button } from "../../../components/ui/button";
import { Card, CardContent } from "../../../components/ui/card";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "../../../components/ui/dialog";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "../../../components/ui/select";
import { PromoteDialog } from "../../../components/folder-run/PromoteDialog";
import { LabelFilterModal } from "../../../components/verify/LabelFilterModal";
import { FolderSelector } from "../../../components/analyses/FolderSelector";
import type { SaveOutputsResult } from "../../../api/folder-runs";
import type { UseSaveOutputsFormResult } from "./useSaveOutputsForm";

// ─────────────────────────────────────────────────────────────────
// Small primitives
// ─────────────────────────────────────────────────────────────────

/** One row inside a body: a label on the left, a control on the
 * right. Two equal-width columns so the dropdowns get half the
 * card and the row layout stays predictable across cards. */
/** A settings row: title + muted caption on the left, control on the
 * right, vertically centred against the two-line text. Shared by the
 * export and media cards so every option lines up the same way. The
 * parent wraps these in a ``divide-y`` list for the hairline rules. */
function CaptionedCheckbox({
  checked,
  onChange,
  label,
  caption,
}: {
  checked: boolean;
  onChange: (v: boolean) => void;
  label: string;
  caption: string;
}) {
  return (
    <label className="flex cursor-pointer items-center justify-between gap-3 py-3 text-sm">
      <span>
        {label}
        <span className="mt-0.5 block text-xs text-muted-foreground">
          {caption}
        </span>
      </span>
      <input
        type="checkbox"
        className="h-4 w-4 shrink-0 accent-primary"
        checked={checked}
        onChange={(e) => onChange(e.target.checked)}
      />
    </label>
  );
}

// ─────────────────────────────────────────────────────────────────
// Output folder field
// ─────────────────────────────────────────────────────────────────

export function OutputFolderField({
  form,
}: {
  form: UseSaveOutputsFormResult;
}) {
  return (
    <Card>
      <CardContent className="space-y-3 p-6">
        <div>
          <span className="block text-sm font-semibold">Output folder</span>
          <span className="mt-0.5 block text-xs text-muted-foreground">
            Where everything gets written
          </span>
        </div>
        <FolderSelector
          value={form.outputDir || null}
          onChange={form.setOutputDir}
          hideLabel
          hideScanResult
          noScan
        />
        {form.sourceFolderConflict && (
          <p className="text-xs text-destructive">
            Saving into the source folder itself would overwrite your
            originals. Pick a subfolder or another location.
          </p>
        )}
      </CardContent>
    </Card>
  );
}

// ─────────────────────────────────────────────────────────────────
// Group bodies — used inside every variant
// ─────────────────────────────────────────────────────────────────

export function MediaBody({
  form,
}: {
  form: UseSaveOutputsFormResult;
}) {
  const {
    separate,
    setSeparate,
    labelTree,
    visualise,
    setVisualise,
    anonymise,
    setAnonymise,
  } = form;
  // Grouping only makes sense once there are species subfolders; with
  // "No subfolders" every file lands flat at the root.
  const showGrouping = separate.groupBy !== "none";
  return (
    <div className="divide-y [&>*:first-child]:pt-0 [&>*:last-child]:pb-0">
      <div className="grid grid-cols-2 items-center gap-3 py-3 text-sm">
        <span>
          Folder structure
          <span className="mt-0.5 block text-xs text-muted-foreground">
            How the copies are organised
          </span>
        </span>
        <Select
          value={separate.groupBy}
          onValueChange={(v) =>
            setSeparate({ ...separate, groupBy: v as typeof separate.groupBy })
          }
        >
          <SelectTrigger>
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="none">No subfolders</SelectItem>
            <SelectItem value="flat">One folder per species</SelectItem>
            <SelectItem value="taxonomic">Nested by taxonomy</SelectItem>
          </SelectContent>
        </Select>
      </div>

      {showGrouping && (
        <CaptionedCheckbox
          checked={separate.groupEvents}
          onChange={(v) => setSeparate({ ...separate, groupEvents: v })}
          label="Keep events together"
          caption="All photos and videos from one event go to the same folder"
        />
      )}

      {labelTree && labelTree.tree.length > 0 && (
        <LabelFilterRow form={form} labelTree={labelTree} />
      )}

      <CaptionedCheckbox
        checked={visualise.enabled}
        onChange={(v) => setVisualise({ enabled: v })}
        label="Draw detection boxes"
        caption="Boxes and labels on each file"
      />
      <CaptionedCheckbox
        checked={anonymise.enabled}
        onChange={(v) => setAnonymise({ enabled: v })}
        label="Blur people and vehicles"
        caption="People and vehicles blurred on each file"
      />
      <CaptionedCheckbox
        checked={separate.copyEmpties}
        onChange={(v) => setSeparate({ ...separate, copyEmpties: v })}
        label="Also copy empty files"
        caption="Images and videos with no animals, people, or vehicles"
      />
    </div>
  );
}

/** Row that opens the shared label-tree modal to limit which labels
 * (species, higher taxa, person, vehicle) get copied and visualised.
 * Inclusion model: an empty selection means "all", so the request only
 * sends a filter when the user picks a real subset. The data exports
 * are never filtered. */
function LabelFilterRow({
  form,
  labelTree,
}: {
  form: UseSaveOutputsFormResult;
  labelTree: NonNullable<UseSaveOutputsFormResult["labelTree"]>;
}) {
  const { separate, setSeparate } = form;
  const [open, setOpen] = useState(false);

  const allCount = labelTree.all_leaf_ids.length;
  const includedCount = separate.includedLabelIds.length;
  const isAll = includedCount === 0 || includedCount >= allCount;

  return (
    <div className="grid grid-cols-2 items-center gap-3 py-3 text-sm">
      <span>
        Labels
        <span className="mt-0.5 block text-xs text-muted-foreground">
          Which labels to copy and visualise
        </span>
      </span>
      <Button
        variant="outline"
        className="w-full justify-start font-normal"
        onClick={() => setOpen(true)}
      >
        <span className="truncate">
          {isAll ? "All labels" : `${includedCount} of ${allCount} labels`}
        </span>
      </Button>
      <LabelFilterModal
        preBuiltTree={labelTree.tree}
        allLeafIds={labelTree.all_leaf_ids}
        selectedLabels={separate.includedLabelIds}
        onApply={(labels) => {
          // Treat "all selected" as no filter so the request stays empty.
          const next = labels.length >= allCount ? [] : labels;
          setSeparate({ ...separate, includedLabelIds: next });
        }}
        open={open}
        onOpenChange={setOpen}
        countUnit={labelTree.count_unit}
      />
    </div>
  );
}

export function ExportBody({
  form,
}: {
  form: UseSaveOutputsFormResult;
}) {
  const { exportOpts, setExportOpts } = form;
  return (
    <div className="divide-y [&>*:first-child]:pt-0 [&>*:last-child]:pb-0">
      <CaptionedCheckbox
        checked={exportOpts.csv}
        onChange={(v) => setExportOpts({ ...exportOpts, csv: v })}
        label="CSV"
        caption="One row per detection"
      />
      <CaptionedCheckbox
        checked={exportOpts.xlsx}
        onChange={(v) => setExportOpts({ ...exportOpts, xlsx: v })}
        label="XLSX"
        caption="The same table, as an Excel file"
      />
      <CaptionedCheckbox
        checked={exportOpts.recognitionJson}
        onChange={(v) => setExportOpts({ ...exportOpts, recognitionJson: v })}
        label="JSON"
        caption="Recognition file you can load into Timelapse"
      />
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────
// Common end-of-form pieces
// ─────────────────────────────────────────────────────────────────

export function SaveErrorLine({ error }: { error: Error | null }) {
  if (!error) return null;
  return (
    <p className="text-sm text-destructive">
      Could not save outputs: {error.message}
    </p>
  );
}

// ─────────────────────────────────────────────────────────────────
// Completion dialog
// ─────────────────────────────────────────────────────────────────

/** Success dialog shown over the still-mounted form once a save
 * finishes. Saving is repeatable, so closing the dialog (the X) just
 * clears the result and returns to the form with the settings intact —
 * ready to tweak and save again. */
export function CompletionDialog({
  runId,
  runName,
  form,
}: {
  runId: string;
  runName: string;
  form: UseSaveOutputsFormResult;
}) {
  const navigate = useNavigate();
  const { result, promoteOpen, setPromoteOpen, handleOpenResults } = form;
  if (!result) return null;

  const issues = collectIssues(result);
  const sourceCount = result.source_file_count;

  // Promote-to-research path is intentionally hidden from the folder-run
  // completion — beta testers found the supporting jargon (Camtrap-DP,
  // GeoJSON, Shapefile, dashboards) intimidating, and any extra control
  // at the moment of success reads as upsell. The PromoteDialog + form
  // state + backend promote endpoint stay wired so re-enabling is a
  // one-block uncomment if we add an entry point elsewhere (e.g. the
  // Step 1 "you analysed this folder before" notice card).
  //
  // Known tradeoff: users who later decide they want dashboards on this
  // folder must spin up a fresh research project and re-run analysis on
  // the same media. The redo cost is real but rare.
  //
  // To re-enable, drop this block into the dialog above the footer (also
  // re-import Sparkles from lucide-react):
  //
  // <div className="flex items-start gap-3 rounded-md border p-3">
  //   <Sparkles className="mt-0.5 h-4 w-4 shrink-0 text-primary" />
  //   <div className="flex-1">
  //     <p className="text-xs text-muted-foreground">
  //       Want dashboards, insights, and full exports? Turn this
  //       into a research project.
  //     </p>
  //     <Button
  //       onClick={() => setPromoteOpen(true)}
  //       variant="outline"
  //       size="sm"
  //       className="mt-2"
  //     >
  //       Promote to research project
  //     </Button>
  //   </div>
  // </div>

  return (
    <>
      <Dialog
        open
        onOpenChange={(v) => {
          if (!v) form.clearResult();
        }}
      >
        <DialogContent>
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <CheckCircle2 className="h-5 w-5 text-primary" />
              Outputs saved
            </DialogTitle>
            <DialogDescription>
              {sourceCount > 0
                ? `${sourceCount.toLocaleString()} source ${
                    sourceCount === 1 ? "file" : "files"
                  } processed`
                : "Your outputs were written to disk"}
            </DialogDescription>
          </DialogHeader>

          <div className="rounded-md border bg-muted/30 p-3 text-xs">
            <p className="mb-1 text-muted-foreground">Saved to</p>
            <code className="break-all font-mono">{result.output_dir}</code>
          </div>

          {issues.length > 0 && <IssuesPanel issues={issues} />}

          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => navigate("/folder-runs/new")}
            >
              Analyse another folder
            </Button>
            <Button onClick={handleOpenResults} className="gap-2">
              <FolderOpen className="h-4 w-4" />
              Open output folder
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <PromoteDialog
        open={promoteOpen}
        onOpenChange={setPromoteOpen}
        runId={runId}
        defaultName={runName}
      />
    </>
  );
}

/** Roll every module's reported errors + missing-source counts into a
 * flat list. Returns an empty array when nothing went wrong, which
 * is the case the completion screen optimises for: no banner. */
function collectIssues(result: SaveOutputsResult): string[] {
  const out: string[] = [];

  // Friendly "couldn't find these files" summary per module that
  // reports missing sources separately. Surfacing this matters because
  // it usually means the user's source folder moved or was edited
  // between analysis and save.
  const missing =
    (result.separate_folders?.skipped_missing_source ?? 0) +
    (result.annotated_copies?.skipped_missing_source ?? 0);
  if (missing > 0) {
    out.push(
      `${missing.toLocaleString()} source file${
        missing === 1 ? "" : "s"
      } could not be found on disk.`,
    );
  }

  // Per-module errors[] — actual exceptions during write.
  const all = [
    ...(result.separate_folders?.errors ?? []),
    ...(result.annotated_copies?.errors ?? []),
    ...(result.recognition_json?.errors ?? []),
    ...(result.csv?.errors ?? []),
    ...(result.xlsx?.errors ?? []),
    ...(result.run_readme?.errors ?? []),
  ];
  out.push(...all);

  return out;
}

function IssuesPanel({ issues }: { issues: string[] }) {
  const shown = issues.slice(0, 5);
  const extra = issues.length - shown.length;
  return (
    <div className="flex items-start gap-2 rounded-md border border-amber-200 bg-amber-50 p-3 text-xs text-amber-700">
      <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0 text-amber-600" />
      <div className="flex-1 space-y-1">
        <p className="font-medium">
          {issues.length} issue{issues.length === 1 ? "" : "s"} during
          save
        </p>
        <ul className="space-y-0.5">
          {shown.map((msg, i) => (
            <li key={i} className="break-all">
              {msg}
            </li>
          ))}
          {extra > 0 && (
            <li className="italic">
              and {extra} more (see ~/AddaxAI/logs/backend.log)
            </li>
          )}
        </ul>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────
// Bottom Back / Save bar used by all variants except A (which uses
// per-tab save buttons).
// ─────────────────────────────────────────────────────────────────

export function BackSaveBar({
  runId,
  form,
}: {
  runId: string;
  form: UseSaveOutputsFormResult;
}) {
  const navigate = useNavigate();
  return (
    <div className="flex items-center justify-between">
      <Button
        variant="outline"
        onClick={() => navigate(`/folder-runs/${runId}/overview`)}
        className="gap-2"
      >
        <ArrowLeft className="h-4 w-4" />
        Back
      </Button>
      <Button
        onClick={form.saveAll}
        disabled={!form.canSave}
        className="gap-2"
        size="lg"
      >
        <Save className="h-4 w-4" />
        {form.isSpawning ? "Starting..." : "Save outputs"}
      </Button>
    </div>
  );
}

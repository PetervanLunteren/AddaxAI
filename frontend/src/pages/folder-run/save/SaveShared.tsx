/**
 * Reusable building blocks for the four Save-outputs layout variants.
 *
 * Each variant mounts these the same way; the only difference is the
 * container chrome (tabs vs accordion vs flat list).
 */

import { useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  ArrowLeft,
  CheckCircle2,
  FolderOpen,
  Save,
  Sparkles,
} from "lucide-react";

import { Button } from "../../../components/ui/button";
import { Callout } from "../../../components/ui/callout";
import { Card, CardContent } from "../../../components/ui/card";
import { ConfidenceSlider } from "../../../components/ui/confidence-slider";
import { NextStepRow } from "../../../components/ui/next-step-row";
import {
  DEFAULT_COUNTING_THRESHOLD,
  DETECTION_CONFIDENCE_ADVICE,
  formatConfidencePct,
} from "../../../lib/confidence";
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
            Where everything gets written. Defaults to the folder you
            analysed. Your originals are never overwritten.
          </span>
        </div>
        <FolderSelector
          value={form.outputDir || null}
          onChange={form.setOutputDir}
          hideLabel
          hideScanResult
          noScan
        />
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
  // Rows read as three blocks: where the copies go (folder structure and
  // the two rows it gates), what gets copied (labels, confidence,
  // empties), then what the copies look like (boxes, blur). The rows
  // carry no headings, so the order is the only thing grouping them.
  return (
    <div className="divide-y [&>*:first-child]:pt-0 [&>*:last-child]:pb-0">
      <div className="grid grid-cols-[2fr_1fr] items-center gap-3 py-3 text-sm">
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
          caption="The whole event goes to the folder of its most confident species"
        />
      )}

      {showGrouping && (
        <div className="grid grid-cols-[2fr_1fr] items-center gap-3 py-3 text-sm">
          <span>
            Folder order
            <span className="mt-0.5 block text-xs text-muted-foreground">
              Whether species or your original folders sit on top
            </span>
          </span>
          <Select
            value={separate.speciesLast ? "species-last" : "species-first"}
            onValueChange={(v) =>
              setSeparate({ ...separate, speciesLast: v === "species-last" })
            }
          >
            <SelectTrigger>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="species-first">
                Species folder first
              </SelectItem>
              <SelectItem value="species-last">
                Species folder last
              </SelectItem>
            </SelectContent>
          </Select>
        </div>
      )}

      {labelTree && labelTree.tree.length > 0 && (
        <LabelFilterRow form={form} labelTree={labelTree} />
      )}

      <div className="grid grid-cols-[2fr_1fr] items-center gap-3 py-3 text-sm">
        <span>
          Confidence
          <span className="mt-0.5 block text-xs text-muted-foreground">
            Detections below this score are left out, except ones you
            verified
          </span>
        </span>
        <ConfidenceSlider
          value={separate.mediaThreshold}
          onChange={(vals) =>
            setSeparate({ ...separate, mediaThreshold: vals[0] })
          }
          adviseBelow={DETECTION_CONFIDENCE_ADVICE}
          valueLabel={
            <span className="min-w-[3rem] shrink-0 text-right text-sm font-medium">
              {formatConfidencePct(separate.mediaThreshold)}
            </span>
          }
          onReset={() =>
            setSeparate({
              ...separate,
              mediaThreshold: DEFAULT_COUNTING_THRESHOLD,
            })
          }
          resetDisabled={
            Math.abs(separate.mediaThreshold - DEFAULT_COUNTING_THRESHOLD) <
            1e-6
          }
        />
      </div>

      <CaptionedCheckbox
        checked={separate.copyEmpties}
        onChange={(v) => setSeparate({ ...separate, copyEmpties: v })}
        label="Copy empty files"
        caption="Images and videos with no animals, people, or vehicles"
      />

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
    <div className="grid grid-cols-[2fr_1fr] items-center gap-3 py-3 text-sm">
      <span>
        Labels
        <span className="mt-0.5 block text-xs text-muted-foreground">
          Which labels to copy and draw boxes for
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
        caption="Tables for files and detections"
      />
      <CaptionedCheckbox
        checked={exportOpts.xlsx}
        onChange={(v) => setExportOpts({ ...exportOpts, xlsx: v })}
        label="XLSX"
        caption="The same tables in one Excel workbook"
      />
      <CaptionedCheckbox
        checked={exportOpts.recognitionJson}
        onChange={(v) => setExportOpts({ ...exportOpts, recognitionJson: v })}
        label="JSON"
        caption="Recognition file for Timelapse, detections only"
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

          {/* One line, ellipsis at the start, so the leaf folder (the
              part the user recognises) always survives. An rtl base
              direction puts the ellipsis on the left; `bdi` isolates the
              path so its own left-to-right order is kept. Without it the
              leading "/" is a neutral character and bidi reordering
              moves it to the far right. Full path on hover. */}
          <div className="rounded-md border bg-muted/30 p-3 text-xs">
            <p className="mb-1 text-muted-foreground">Saved to</p>
            <code
              className="block truncate text-left font-mono [direction:rtl]"
              title={result.output_dir}
            >
              <bdi>{result.output_dir}</bdi>
            </code>
          </div>

          {issues.length > 0 && <IssuesPanel issues={issues} />}

          {/* Same shape as the projects-mode completion modal: the steps
              that take you somewhere are rows, and "start over" stays in
              the footer. "Turn into a project" is the bridge to projects
              mode; a folder run deliberately does no ecological
              interpretation, and this is the one place we point at where
              that lives. */}
          <div className="space-y-2 pt-1">
            <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
              What next?
            </p>
            <NextStepRow
              icon={FolderOpen}
              title="Open output folder"
              description="Show the files AddaxAI wrote."
              onClick={handleOpenResults}
            />
            <NextStepRow
              icon={Sparkles}
              title="Turn into a project"
              description="Get species counts, dashboards, and maps for this folder."
              onClick={() => setPromoteOpen(true)}
            />
          </div>

          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => navigate("/folder-runs/new")}
            >
              Analyse another folder
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
    <Callout variant="warning" size="compact">
      <div className="space-y-1">
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
    </Callout>
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
        onClick={() => navigate(`/folder-runs/${runId}/labels`)}
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

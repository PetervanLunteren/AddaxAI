/**
 * Reusable building blocks for the four Save-outputs layout variants.
 *
 * Each variant mounts these the same way; the only difference is the
 * container chrome (tabs vs accordion vs flat list).
 */

import { useState } from "react";
import { useNavigate } from "react-router-dom";

import { StepActionBar } from "../../../components/folder-run/StepActionBar";
import {
  ArrowLeft,
  CheckCircle2,
  FolderOpen,
  Save,
  SlidersHorizontal,
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
import { ExportScopeSelect } from "../../../components/export/ExportScopeSelect";
import { DateRangePicker } from "../../../components/ui/date-range-picker";
import { collectIssues } from "./media-options";
import type { MediaFormHandles } from "./useMediaOptions";
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
export function CaptionedCheckbox({
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

/** A settings row with a control on the right: title + muted caption on
 * the left, the control in a fixed-width column. Fixed, not a fraction,
 * so a row inside `ChildRows` (indented) gets a control exactly as wide
 * as the rows around it; with `2fr_1fr` the indent used to shrink the
 * child's control and the dropdowns no longer lined up. */
export function SettingRow({
  title,
  caption,
  children,
}: {
  title: string;
  caption: string;
  children: React.ReactNode;
}) {
  return (
    <div className="grid grid-cols-[minmax(0,1fr)_16rem] items-center gap-3 py-3 text-sm">
      <span>
        {title}
        <span className="mt-0.5 block text-xs text-muted-foreground">
          {caption}
        </span>
      </span>
      {children}
    </div>
  );
}

/** Rows that exist only because of the row above them, and that govern
 * only that row's output.
 *
 * Flat, a format picker and a threshold read as siblings of the
 * checkboxes around them, so the threshold looked like it governed the
 * recognition JSON too (it does not) and the format looked like a
 * separate output. Nesting says which parent owns them without a word of
 * explanation.
 *
 * Shared by both bodies below: the spreadsheet's format and threshold,
 * and the folder-structure options the media card only shows once
 * subfolders are on. Keep new nesting going through here so the two
 * cards cannot drift into two different visual languages. */
function ChildRows({ children }: { children: React.ReactNode }) {
  // The indent is the whole signal, and the dividers between children
  // fall inside it, so they start at the child's text and stop short of
  // the full-width rules above and below the group. That difference in
  // length is what the eye reads, and it needs no rule, elbow or dot
  // drawn on top of it: those repeat something already said once.
  // When the group closes a panel, its own last row drops its bottom
  // padding too, so the panel ends with the same space it starts with.
  return (
    <div className="pl-6 [&>*+*]:border-t [&:last-child>*:last-child]:pb-0">
      {children}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────
// Output folder field
// ─────────────────────────────────────────────────────────────────

export function OutputFolderField({
  value,
  onChange,
  caption,
}: {
  value: string;
  onChange: (v: string) => void;
  /** What lands there and what is safe. Each surface says its own: the
   * folder run defaults to the analysed folder and writes data files at
   * its root too; the project dialog writes only addaxai-media. */
  caption: string;
}) {
  return (
    <Card>
      <CardContent className="space-y-3 p-6">
        <div>
          <span className="block text-sm font-semibold">Output folder</span>
          <span className="mt-0.5 block text-xs text-muted-foreground">
            {caption}
          </span>
        </div>
        <FolderSelector
          value={value || null}
          onChange={onChange}
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

export function MediaBody({ form }: { form: MediaFormHandles }) {
  const {
    separate,
    setSeparate,
    labelTree,
    visualise,
    setVisualise,
    anonymise,
    setAnonymise,
    showDeploymentPicker,
    dateBounds,
  } = form;
  // Grouping only makes sense once there are species subfolders; with
  // "No subfolders" every file lands flat at the root.
  const showGrouping = separate.groupBy !== "none";
  // Three headed blocks: where the copies go, which files are copied,
  // what the copies look like. The same body serves the folder-run Save
  // step and the project image copies dialog, so a user learns it once.
  return (
    <div className="space-y-4">
      <Block title="Folders">
        <SettingRow
          title="Folder structure"
          caption="How the copies are organised"
        >
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
        </SettingRow>

        {/* Both only exist once there are species subfolders, so they are
            children of the row above rather than siblings of it. */}
        {showGrouping && (
          <ChildRows>
            <CaptionedCheckbox
              checked={separate.groupEvents}
              onChange={(v) => setSeparate({ ...separate, groupEvents: v })}
              label="Keep events together"
              caption="Every file in an event goes to one folder: the species you verified most often, or else the strongest detection"
            />
            <SettingRow
              title="Folder order"
              caption="Whether species or your original folders sit on top"
            >
              <Select
                value={separate.speciesLast ? "species-last" : "species-first"}
                onValueChange={(v) =>
                  setSeparate({
                    ...separate,
                    speciesLast: v === "species-last",
                  })
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
            </SettingRow>
          </ChildRows>
        )}
      </Block>

      <Block title="Which files">
        {/* Only when there is something to pick: a folder run has one
            deployment, a one-camera project too, and neither sees this
            row. The picker owns its selection and reports the ids. */}
        {showDeploymentPicker && (
          <SettingRow
            title="Deployments"
            caption="Which cameras to copy from"
          >
            <ExportScopeSelect
              projectId={form.projectId}
              onChange={(scope) =>
                setSeparate({ ...separate, deploymentIds: scope?.deploymentIds })
              }
              className="w-full min-w-0"
            />
          </SettingRow>
        )}

        {labelTree && labelTree.tree.length > 0 && (
          <LabelFilterRow form={form} labelTree={labelTree} />
        )}

        {/* Named for the media on purpose: the Export results card above
            carries the run's own threshold, and two rows both called
            "Confidence" on one screen is how you get the wrong one. */}
        <SettingRow
          title="Media confidence"
          caption="Detections below this score are left out of the copies, except ones you verified."
        >
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
        </SettingRow>

        <SettingRow
          title="Capture dates"
          caption="Files without a capture time are left out once a date is set"
        >
          <DateRangePicker
            from={separate.dateFrom}
            to={separate.dateTo}
            onChange={(r) =>
              setSeparate({ ...separate, dateFrom: r.from, dateTo: r.to })
            }
            minDate={dateBounds?.min}
            maxDate={dateBounds?.max}
          />
        </SettingRow>

        <CaptionedCheckbox
          checked={separate.likedOnly}
          onChange={(v) => setSeparate({ ...separate, likedOnly: v })}
          label="Only liked files"
          caption="The files you marked with a heart on the Labels page"
        />
        <CaptionedCheckbox
          checked={separate.verifiedOnly}
          onChange={(v) => setSeparate({ ...separate, verifiedOnly: v })}
          label="Only verified files"
          caption="Files where every box was verified"
        />

        <CaptionedCheckbox
          checked={separate.copyEmpties}
          onChange={(v) => setSeparate({ ...separate, copyEmpties: v })}
          label="Copy empty files"
          caption="Images and videos with no animals, people, or vehicles"
        />
      </Block>

      <Block title="Boxes and blur">
        <CaptionedCheckbox
          checked={visualise.enabled}
          onChange={(v) => setVisualise({ enabled: v })}
          label="Draw detection boxes"
          caption="Boxes and labels drawn on each image. A video gets a still image with the boxes next to it, the video itself is not changed"
        />
        <CaptionedCheckbox
          checked={anonymise.enabled}
          onChange={(v) => setAnonymise({ enabled: v })}
          label="Blur people and vehicles"
          caption="People and vehicles blurred on each image. A video cannot be blurred, so it is saved as a blurred still image only"
        />
      </Block>
    </div>
  );
}

/** A group of rows in an inset panel with a header strip. The frame and
 * fill are the ones the preview panel and the "Saved to" box use, so
 * the step has one inset style. Chosen over a bare caps label, which
 * nothing else in the app uses, and over whitespace alone, which left
 * twelve rows without a signpost. */
function Block({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <div className="overflow-hidden rounded-md border bg-muted/30">
      <p className="border-b bg-muted/50 px-4 py-2 text-sm font-medium">
        {title}
      </p>
      <div className="divide-y px-4 py-3 [&>*:first-child]:pt-0 [&>*:last-child]:pb-0">
        {children}
      </div>
    </div>
  );
}

/** Row that opens the shared label-tree modal to limit which labels
 * (species, higher taxa, person, vehicle) get copied and visualised.
 * Inclusion model: an empty selection means every label, so the
 * request only sends a filter when the user picks a real subset. The
 * same on both surfaces; the preview count is what tells the user how
 * much a whole-project copy is. The data exports are never filtered. */
function LabelFilterRow({
  form,
  labelTree,
}: {
  form: MediaFormHandles;
  labelTree: NonNullable<MediaFormHandles["labelTree"]>;
}) {
  const { separate, setSeparate } = form;
  const [open, setOpen] = useState(false);

  const allCount = labelTree.all_leaf_ids.length;
  const includedCount = separate.includedLabelIds.length;
  const isAll = includedCount === 0 || includedCount >= allCount;
  const buttonText = isAll
    ? "All labels"
    : `${includedCount} of ${allCount} labels`;

  return (
    <SettingRow
      title="Labels"
      caption="Which labels to copy and draw boxes for"
    >
      <Button
        variant="outline"
        className="w-full justify-between font-normal"
        onClick={() => setOpen(true)}
      >
        <span className="truncate text-left">{buttonText}</span>
        <SlidersHorizontal className="ml-1.5 h-3.5 w-3.5 shrink-0 opacity-50" />
      </Button>
      <LabelFilterModal
        preBuiltTree={labelTree.tree}
        allLeafIds={labelTree.all_leaf_ids}
        selectedLabels={separate.includedLabelIds}
        onApply={(labels) => {
          // Every label ticked is no filter, so the state stays empty.
          const next = labels.length >= allCount ? [] : labels;
          setSeparate({ ...separate, includedLabelIds: next });
        }}
        open={open}
        onOpenChange={setOpen}
        countUnit={labelTree.count_unit}
      />
    </SettingRow>
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
        checked={exportOpts.spreadsheet}
        onChange={(v) => setExportOpts({ ...exportOpts, spreadsheet: v })}
        label="Spreadsheet"
        caption="Tables for the species summary, files and detections"
      />
      {exportOpts.spreadsheet && (
        <ChildRows>
          <SettingRow
            title="Format"
            caption="The same tables either way"
          >
            <Select
              value={exportOpts.format}
              onValueChange={(v) =>
                setExportOpts({
                  ...exportOpts,
                  format: v as typeof exportOpts.format,
                })
              }
            >
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="csv">CSV, one file per table</SelectItem>
                <SelectItem value="xlsx">
                  Excel, one sheet per table
                </SelectItem>
              </SelectContent>
            </Select>
          </SettingRow>
          {/* Nested because it governs these tables and not the
              recognition file below, which is unfiltered by design. The
              caption carries the other half: this is the run's own
              threshold, so moving it here moves the labels grid and the
              counts with it. It is one number, not an export-only copy,
              which is what keeps the tables from ever disagreeing with
              what the user signed off in step 2. */}
          <SettingRow
            title="Detections included"
            caption="Everything at this detection confidence or above, plus everything you verified."
          >
            <ConfidenceSlider
              value={form.countingThreshold}
              onChange={(vals) => form.setCountingThreshold(vals[0])}
              onCommit={(vals) => form.commitCountingThreshold(vals[0])}
              valueLabel={
                <span className="min-w-[3rem] shrink-0 text-right text-sm font-medium">
                  {formatConfidencePct(form.countingThreshold)}
                </span>
              }
              // Move the handle first, then save. Committing alone left it
              // sitting on the old value for as long as the write took,
              // which on a large run is a second or more of the button
              // looking dead. A drag never had this, because dragging
              // moves the handle through onChange on the way.
              onReset={() => {
                form.setCountingThreshold(DEFAULT_COUNTING_THRESHOLD);
                form.commitCountingThreshold(DEFAULT_COUNTING_THRESHOLD);
              }}
              resetDisabled={
                Math.abs(
                  form.countingThreshold - DEFAULT_COUNTING_THRESHOLD,
                ) < 1e-6
              }
            />
          </SettingRow>
        </ChildRows>
      )}
      <CaptionedCheckbox
        checked={exportOpts.recognitionJson}
        onChange={(v) => setExportOpts({ ...exportOpts, recognitionJson: v })}
        label="JSON"
        caption="Recognition file for Timelapse, MegaDetector output format, detections only"
      />
      <CaptionedCheckbox
        checked={exportOpts.summary}
        onChange={(v) => setExportOpts({ ...exportOpts, summary: v })}
        label="Run details"
        caption="A text file with the models, settings, and details behind this run"
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
    <p className="line-clamp-2 min-w-0 flex-1 break-words text-sm text-destructive">
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

export function IssuesPanel({ issues }: { issues: string[] }) {
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
              and {extra} more (see logs/backend.log in the app's data
              folder)
            </li>
          )}
        </ul>
      </div>
    </Callout>
  );
}

// ─────────────────────────────────────────────────────────────────
// Bottom Back / Save bar used by all variants except A (which uses
// per-tab save buttons). Sticky via the shared StepActionBar, the same
// bar the Labels step uses, so the two can't drift apart. Must be
// rendered as a direct child of the step root, not inside the options
// column: see the layout contract in StepActionBar.
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
    <StepActionBar>
      <Button
        variant="outline"
        onClick={() => navigate(`/folder-runs/${runId}/labels`)}
        className="gap-2"
      >
        <ArrowLeft className="h-4 w-4" />
        Back
      </Button>
      {/* In the sticky bar, not under the form: the bar overlays the
          end of the column while scrolled, which clipped the message
          exactly when it appeared. Here it sits beside the button the
          user just clicked and is always visible. */}
      <SaveErrorLine error={form.saveError} />
      <Button
        onClick={form.saveAll}
        disabled={!form.canSave}
        className="gap-2"
        size="lg"
      >
        <Save className="h-4 w-4" />
        {form.isSpawning ? "Starting..." : "Save outputs"}
      </Button>
    </StepActionBar>
  );
}

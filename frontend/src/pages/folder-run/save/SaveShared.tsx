/**
 * Reusable building blocks for the four Save-outputs layout variants.
 *
 * Each variant mounts these the same way; the only difference is the
 * container chrome (tabs vs accordion vs flat list).
 */

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
import { Checkbox } from "../../../components/ui/checkbox";
import { Label } from "../../../components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "../../../components/ui/select";
import { PromoteDialog } from "../../../components/folder-run/PromoteDialog";
import { FolderSelector } from "../../../components/analyses/FolderSelector";
import { isElectron } from "../../../lib/platform";
import type { SaveOutputsResult } from "../../../api/folder-runs";
import type { UseSaveOutputsFormResult } from "./useSaveOutputsForm";

// ─────────────────────────────────────────────────────────────────
// Small primitives
// ─────────────────────────────────────────────────────────────────

/** One row inside a body: a label on the left, a control on the
 * right. Two equal-width columns so the dropdowns get half the
 * card and the row layout stays predictable across cards. */
function Row({
  label,
  children,
}: {
  label: string;
  children: React.ReactNode;
}) {
  return (
    <div className="grid grid-cols-2 items-center gap-3">
      <span className="text-sm">{label}</span>
      <div>{children}</div>
    </div>
  );
}

/** Compact checkbox toggle: checkbox on the left, label after.
 * Conventional shape for "pick which of these to do" lists, and
 * matches the top-of-card enable checkbox. */
function CheckboxRow({
  id,
  checked,
  onChange,
  label,
}: {
  id: string;
  checked: boolean;
  onChange: (v: boolean) => void;
  label: string;
}) {
  return (
    <label
      htmlFor={id}
      className="flex cursor-pointer items-center gap-2"
    >
      <Checkbox
        id={id}
        checked={checked}
        onCheckedChange={(v) => onChange(Boolean(v))}
      />
      <span className="text-sm">{label}</span>
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
      <CardContent className="space-y-2 p-6">
        <Label>Output folder</Label>
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
    visualise,
    setVisualise,
    anonymise,
    setAnonymise,
  } = form;
  return (
    <div className="space-y-3">
      <Row label="Folder structure">
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
      </Row>

      <label className="flex cursor-pointer items-center gap-2 text-sm">
        <input
          type="checkbox"
          className="h-4 w-4 accent-primary"
          checked={visualise.enabled}
          onChange={(e) => setVisualise({ enabled: e.target.checked })}
        />
        Draw detection boxes
      </label>

      <label className="flex cursor-pointer items-center gap-2 text-sm">
        <input
          type="checkbox"
          className="h-4 w-4 accent-primary"
          checked={anonymise.enabled}
          onChange={(e) => setAnonymise({ enabled: e.target.checked })}
        />
        Blur people and vehicles
      </label>

      <label className="flex cursor-pointer items-start gap-2 text-sm">
        <input
          type="checkbox"
          className="mt-0.5 h-4 w-4 accent-primary"
          checked={separate.copyEmpties}
          onChange={(e) =>
            setSeparate({ ...separate, copyEmpties: e.target.checked })
          }
        />
        <span>
          Also copy empty captures
          <span className="mt-0.5 block text-xs text-muted-foreground">
            Images and videos with no animals, people, or vehicles.
          </span>
        </span>
      </label>
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
    <div className="space-y-2">
      <CheckboxRow
        id="export-csv"
        checked={exportOpts.csv}
        onChange={(v) => setExportOpts({ ...exportOpts, csv: v })}
        label="CSV"
      />
      <CheckboxRow
        id="export-xlsx"
        checked={exportOpts.xlsx}
        onChange={(v) => setExportOpts({ ...exportOpts, xlsx: v })}
        label="XLSX"
      />
      <CheckboxRow
        id="export-json"
        checked={exportOpts.recognitionJson}
        onChange={(v) =>
          setExportOpts({ ...exportOpts, recognitionJson: v })
        }
        label="JSON"
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
// Completion screen
// ─────────────────────────────────────────────────────────────────

export function CompletionScreen({
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

  // Promote-to-research path is intentionally hidden from the folder-run
  // completion screen — beta testers found the supporting jargon
  // (Camtrap-DP, GeoJSON, Shapefile, dashboards) intimidating, and any
  // extra control at the moment of success reads as upsell. The
  // PromoteDialog + form state + backend promote endpoint stay wired
  // so re-enabling is a one-block uncomment if we add an entry point
  // elsewhere (e.g. the Step 1 "you analysed this folder before"
  // notice card).
  //
  // Known tradeoff: users who later decide they want dashboards on
  // this folder must spin up a fresh research project and re-run
  // analysis on the same media. The redo cost is real but rare.
  //
  // To re-enable, drop this block back inside the Card above the
  // bottom nav (also re-import Sparkles from lucide-react):
  //
  // <div className="border-t pt-4">
  //   <div className="flex items-start gap-3">
  //     <Sparkles className="mt-0.5 h-4 w-4 shrink-0 text-primary" />
  //     <div className="flex-1">
  //       <p className="text-xs text-muted-foreground">
  //         Want dashboards, insights, and full exports? Turn
  //         this into a research project.
  //       </p>
  //       <Button
  //         onClick={() => setPromoteOpen(true)}
  //         variant="outline"
  //         size="sm"
  //         className="mt-2"
  //       >
  //         Promote to research project
  //       </Button>
  //     </div>
  //   </div>
  // </div>

  return (
    <div className="space-y-6">
      <Card>
        <CardContent className="space-y-4 p-6">
          <div className="flex items-start gap-3">
            <CheckCircle2 className="mt-0.5 h-5 w-5 shrink-0 text-primary" />
            <div className="flex-1 space-y-1">
              <p className="text-sm font-semibold">
                Folder analysis complete
              </p>
              <p className="text-xs text-muted-foreground">
                Saved to{" "}
                <code className="font-mono">{result.output_dir}</code>
              </p>
              {result.source_file_count > 0 && (
                <p className="text-xs text-muted-foreground">
                  {result.source_file_count.toLocaleString()} source
                  file{result.source_file_count === 1 ? "" : "s"}{" "}
                  processed
                </p>
              )}
            </div>
          </div>

          {issues.length > 0 && <IssuesPanel issues={issues} />}

          {isElectron() && (
            <Button onClick={handleOpenResults} className="gap-2" size="lg">
              <FolderOpen className="h-4 w-4" />
              Open results folder
            </Button>
          )}
        </CardContent>
      </Card>

      <div className="flex items-center justify-between">
        <Button variant="outline" onClick={() => navigate("/")}>
          Back to home
        </Button>
        <Button
          variant="outline"
          onClick={() => navigate("/folder-runs/new")}
        >
          Analyse another folder
        </Button>
      </div>

      <PromoteDialog
        open={promoteOpen}
        onOpenChange={setPromoteOpen}
        runId={runId}
        defaultName={runName}
      />
    </div>
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

/**
 * Reusable building blocks for the four Save-outputs layout variants.
 *
 * Each variant mounts these the same way; the only difference is the
 * container chrome (tabs vs accordion vs flat list).
 */

import { useNavigate } from "react-router-dom";
import {
  ArrowLeft,
  CheckCircle2,
  FolderOpen,
  Save,
  Sparkles,
} from "lucide-react";

import { Button } from "../../../components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "../../../components/ui/card";
import { Checkbox } from "../../../components/ui/checkbox";
import { Input } from "../../../components/ui/input";
import { Label } from "../../../components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "../../../components/ui/select";
import { PromoteDialog } from "../../../components/folder-run/PromoteDialog";
import { isElectron } from "../../../lib/platform";
import type { SaveOutputsResponse } from "../../../api/folder-runs";
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
      <CardHeader>
        <CardTitle>Save outputs</CardTitle>
      </CardHeader>
      <CardContent className="space-y-2">
        <Label htmlFor="output-dir">Output folder</Label>
        <div className="flex gap-2">
          <Input
            id="output-dir"
            value={form.effectiveOutputDir}
            onChange={(e) => form.setOutputDir(e.target.value)}
            placeholder="Pick where the outputs should land"
            className="font-mono text-xs"
          />
          {isElectron() && (
            <Button
              variant="outline"
              type="button"
              onClick={form.handleBrowse}
              className="shrink-0"
            >
              Browse...
            </Button>
          )}
        </div>
      </CardContent>
    </Card>
  );
}

// ─────────────────────────────────────────────────────────────────
// Group bodies — used inside every variant
// ─────────────────────────────────────────────────────────────────

export function SeparateBody({
  form,
}: {
  form: UseSaveOutputsFormResult;
}) {
  const { separate, setSeparate } = form;
  const winElectron =
    typeof window !== "undefined" &&
    !!window.electronAPI &&
    window.electronAPI.platform === "win32";
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
            <SelectItem value="taxonomic">Nested by taxonomy</SelectItem>
            <SelectItem value="flat">Flat by species</SelectItem>
          </SelectContent>
        </Select>
      </Row>

      <Row label="File placement">
        <Select
          value={separate.method}
          onValueChange={(v) => {
            if (v === "symlink" && winElectron) return;
            setSeparate({ ...separate, method: v as typeof separate.method });
          }}
        >
          <SelectTrigger>
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="copy">Copy</SelectItem>
            <SelectItem value="move">Move</SelectItem>
            <SelectItem value="symlink" disabled={winElectron}>
              Symbolic link
            </SelectItem>
          </SelectContent>
        </Select>
      </Row>

      {separate.method === "symlink" && winElectron && (
        <p className="text-xs text-destructive">
          Symbolic links need Windows Developer Mode. Use Copy or
          Move instead.
        </p>
      )}
    </div>
  );
}

export function VisualiseBody({
  form,
}: {
  form: UseSaveOutputsFormResult;
}) {
  const { visualise, setVisualise } = form;
  return (
    <CheckboxRow
      id="visualise-blur"
      checked={visualise.blur}
      onChange={(v) => setVisualise({ ...visualise, blur: v })}
      label="Also blur people and vehicles"
    />
  );
}

export function WriteExifBody({
  form,
}: {
  form: UseSaveOutputsFormResult;
}) {
  const { exif, setExif } = form;
  return (
    <div className="space-y-3">
      <Row label="Where to write">
        <Select
          value={exif.mode}
          onValueChange={(v) =>
            setExif({ ...exif, mode: v as typeof exif.mode })
          }
        >
          <SelectTrigger>
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="copy">Save tagged copies</SelectItem>
            <SelectItem value="overwrite">
              Overwrite originals in place
            </SelectItem>
          </SelectContent>
        </Select>
      </Row>
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
// Completion screen + result panels
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

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <CheckCircle2 className="h-5 w-5 text-primary" />
            Folder analysis complete
          </CardTitle>
          <CardDescription>
            Results saved to{" "}
            <code className="font-mono text-xs">
              {result.output_dir}
            </code>
            .
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {result.separate_folders && (
            <SeparateResultPanel data={result.separate_folders} />
          )}
          {(result.visualised_images || result.blur_people) && (
            <AnnotatedCopiesPanel
              visualised={result.visualised_images}
              blurred={result.blur_people}
            />
          )}
          {result.write_exif && (
            <WriteExifResultPanel data={result.write_exif} />
          )}
          {(result.csv || result.xlsx || result.recognition_json) && (
            <ExportResultPanel
              csv={result.csv}
              xlsx={result.xlsx}
              json={result.recognition_json}
            />
          )}
          {result.run_readme && (
            <RunReadmeResultPanel data={result.run_readme} />
          )}

          <div className="rounded-md border bg-card-background p-4 text-sm">
            <div className="flex items-start gap-3">
              <Sparkles className="mt-0.5 h-5 w-5 shrink-0 text-primary" />
              <div>
                <p className="font-medium">
                  Keep this as a research project
                </p>
                <p className="mt-1 text-muted-foreground">
                  Promote to access dashboards, insights, full export
                  formats (Camtrap-DP, GeoJSON, Shapefile, etc.), and
                  long-term verification history.
                </p>
                <Button
                  onClick={() => setPromoteOpen(true)}
                  variant="outline"
                  size="sm"
                  className="mt-3"
                >
                  Promote to research project
                </Button>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      <div className="flex items-center justify-between">
        <Button variant="outline" onClick={() => navigate("/")}>
          Back to home
        </Button>
        <div className="flex gap-2">
          <Button
            variant="outline"
            onClick={() => navigate("/folder-runs/new")}
          >
            Analyse another folder
          </Button>
          <Button variant="ghost" onClick={form.clearResult}>
            Save again
          </Button>
          {isElectron() && (
            <Button onClick={handleOpenResults} className="gap-2">
              <FolderOpen className="h-4 w-4" />
              Open results folder
            </Button>
          )}
        </div>
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

function SeparateResultPanel({
  data,
}: {
  data: NonNullable<SaveOutputsResponse["separate_folders"]>;
}) {
  const verb =
    data.moved_count > 0
      ? "Moved"
      : data.linked_count > 0
        ? "Linked"
        : "Copied";
  return (
    <div className="rounded-md border bg-card-background p-4 text-sm">
      <p className="font-medium">
        {verb} {data.written_count} file
        {data.written_count === 1 ? "" : "s"} into{" "}
        {Object.keys(data.by_label).length} label subfolder
        {Object.keys(data.by_label).length === 1 ? "" : "s"}.
      </p>
      {data.multi_placement_count > 0 && (
        <p className="mt-1 text-xs text-muted-foreground">
          {data.multi_placement_count} file
          {data.multi_placement_count === 1 ? "" : "s"} had multiple
          species and appear in more than one folder.
        </p>
      )}
      {data.renamed_count > 0 && (
        <p className="mt-1 text-xs text-muted-foreground">
          {data.renamed_count} file
          {data.renamed_count === 1 ? " was" : "s were"} renamed to
          avoid collisions.
        </p>
      )}
      {data.skipped_missing_source > 0 && (
        <p className="mt-1 text-xs text-destructive">
          {data.skipped_missing_source} file
          {data.skipped_missing_source === 1 ? "" : "s"} could not be
          found on disk.
        </p>
      )}
      <ul className="mt-3 space-y-0.5 text-xs text-muted-foreground">
        {Object.entries(data.by_label)
          .sort((a, b) => b[1] - a[1])
          .map(([label, n]) => (
            <li key={label}>
              <span className="font-medium text-foreground">
                {label}
              </span>
              : {n}
            </li>
          ))}
      </ul>
    </div>
  );
}

function AnnotatedCopiesPanel({
  visualised,
  blurred,
}: {
  visualised: SaveOutputsResponse["visualised_images"];
  blurred: SaveOutputsResponse["blur_people"];
}) {
  return (
    <div className="rounded-md border bg-card-background p-4 text-sm">
      <p className="font-medium">Annotated copies</p>
      {visualised && (
        <p className="mt-1 text-xs text-muted-foreground">
          Visualised: {visualised.written_count} file
          {visualised.written_count === 1 ? "" : "s"} written with
          rounded boxes and pill labels.
        </p>
      )}
      {blurred && (
        <p className="mt-1 text-xs text-muted-foreground">
          Blurred: {blurred.blurred_box_count} person
          {blurred.blurred_box_count === 1 ? "" : "s"} / vehicle
          {blurred.blurred_box_count === 1 ? "" : "s"} hidden across{" "}
          {blurred.written_count} file
          {blurred.written_count === 1 ? "" : "s"}.
        </p>
      )}
    </div>
  );
}

function WriteExifResultPanel({
  data,
}: {
  data: NonNullable<SaveOutputsResponse["write_exif"]>;
}) {
  const mode = data.mode === "overwrite" ? "in place" : "as tagged copies";
  return (
    <div className="rounded-md border bg-card-background p-4 text-sm">
      <p className="font-medium">
        Wrote EXIF metadata on {data.written_count} file
        {data.written_count === 1 ? "" : "s"} {mode}.
      </p>
      {data.skipped_no_detections > 0 && (
        <p className="mt-1 text-xs text-muted-foreground">
          {data.skipped_no_detections} file
          {data.skipped_no_detections === 1 ? "" : "s"} had no
          detections to write.
        </p>
      )}
      {data.skipped_video > 0 && (
        <p className="mt-1 text-xs text-muted-foreground">
          {data.skipped_video} video
          {data.skipped_video === 1 ? " was" : "s were"} skipped
          (videos don't carry EXIF in this output; their best frames
          do via the Visualise output).
        </p>
      )}
      {data.skipped_missing_source > 0 && (
        <p className="mt-1 text-xs text-destructive">
          {data.skipped_missing_source} source file
          {data.skipped_missing_source === 1 ? "" : "s"} could not be
          found on disk.
        </p>
      )}
    </div>
  );
}

function RunReadmeResultPanel({
  data,
}: {
  data: NonNullable<SaveOutputsResponse["run_readme"]>;
}) {
  return (
    <div className="rounded-md border bg-card-background p-4 text-sm">
      <p className="font-medium">Run summary written.</p>
      <p className="mt-1 text-xs text-muted-foreground font-mono">
        {data.output_path}
      </p>
      <p className="mt-2 text-xs text-muted-foreground">
        Plain-text manifest with app version, models, all settings,
        and a results summary. Open it later to remember exactly what
        produced this folder.
      </p>
    </div>
  );
}

function ExportResultPanel({
  csv,
  xlsx,
  json,
}: {
  csv: SaveOutputsResponse["csv"];
  xlsx: SaveOutputsResponse["xlsx"];
  json: SaveOutputsResponse["recognition_json"];
}) {
  return (
    <div className="rounded-md border bg-card-background p-4 text-sm">
      <p className="font-medium">Exports</p>
      {csv && (
        <p className="mt-1 text-xs text-muted-foreground">
          CSV: {csv.row_count} observation
          {csv.row_count === 1 ? "" : "s"} —{" "}
          <span className="font-mono">{csv.output_path}</span>
        </p>
      )}
      {xlsx && (
        <p className="mt-1 text-xs text-muted-foreground">
          XLSX: {xlsx.row_count} observation
          {xlsx.row_count === 1 ? "" : "s"} —{" "}
          <span className="font-mono">{xlsx.output_path}</span>
        </p>
      )}
      {json && (
        <p className="mt-1 text-xs text-muted-foreground">
          Recognition JSON: {json.image_count} image
          {json.image_count === 1 ? "" : "s"},{" "}
          {json.detection_count} detection
          {json.detection_count === 1 ? "" : "s"} —{" "}
          <span className="font-mono">{json.output_path}</span>
        </p>
      )}
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

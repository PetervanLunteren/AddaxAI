/**
 * The retroactive analysis settings rows: independence interval,
 * smoothing, and taxonomic rollup. One source of truth for the two
 * surfaces that expose them — the project Settings page and the
 * folder-run Labels step's analysis panel — so wording, layout, and
 * control behaviour never drift.
 *
 * Presentation-only: each surface owns its form state and maps the
 * three callbacks onto it. Captions come from SETTING_CAPTIONS; a
 * surface can append a short context note per row (the Settings page
 * adds retroactivity notes).
 */

import { IntervalControl } from "../analyses/IntervalControl";
import { CaptionedSelect } from "../ui/captioned-select";
import { Switch } from "../ui/switch";
import { SETTING_CAPTIONS } from "../../lib/settingCaptions";
import { SMOOTHING_LEVELS } from "../../lib/smoothing";

export type SmoothingLevel = "off" | "mild" | "normal" | "aggressive";

export interface AnalysisSettingsValues {
  event_smoothing: boolean;
  smoothing_strength: "mild" | "normal" | "aggressive";
  taxonomic_rollup: boolean;
  independence_interval: number;
}

function Row({
  label,
  caption,
  children,
}: {
  label: string;
  caption: string;
  children: React.ReactNode;
}) {
  return (
    <div className="grid grid-cols-2 items-center gap-8 py-6">
      <div className="space-y-1">
        <span className="block text-sm font-medium">{label}</span>
        <p className="text-sm text-muted-foreground">{caption}</p>
      </div>
      <div className="space-y-2">{children}</div>
    </div>
  );
}

export function AnalysisSettingsRows({
  values,
  onIntervalChange,
  onSmoothingChange,
  onRollupChange,
  showClassifierFields,
  intervalNote,
  smoothingNote,
  rollupNote,
}: {
  values: AnalysisSettingsValues;
  onIntervalChange: (seconds: number) => void;
  /** "off" clears event_smoothing; a strength enables it at that level. */
  onSmoothingChange: (level: SmoothingLevel) => void;
  onRollupChange: (enabled: boolean) => void;
  /** Smoothing + rollup only make sense with a classification model. */
  showClassifierFields: boolean;
  /** Optional per-surface caption suffixes (e.g. retroactivity notes). */
  intervalNote?: string;
  smoothingNote?: string;
  rollupNote?: string;
}) {
  const withNote = (base: string, note?: string) =>
    note ? `${base} ${note}` : base;

  return (
    <>
      <Row
        label="Independence interval"
        caption={withNote(SETTING_CAPTIONS.independenceInterval, intervalNote)}
      >
        <IntervalControl
          value={values.independence_interval}
          onChange={onIntervalChange}
        />
      </Row>

      {showClassifierFields && (
        <Row
          label="Smoothing"
          caption={withNote(SETTING_CAPTIONS.smoothing, smoothingNote)}
        >
          <CaptionedSelect
            value={
              values.event_smoothing ? values.smoothing_strength : "off"
            }
            onValueChange={(value) => {
              if (!value) return;
              onSmoothingChange(value as SmoothingLevel);
            }}
            options={SMOOTHING_LEVELS}
          />
        </Row>
      )}

      {showClassifierFields && (
        <Row
          label="Taxonomic rollup"
          caption={withNote(SETTING_CAPTIONS.taxonomicRollup, rollupNote)}
        >
          <Switch
            checked={values.taxonomic_rollup}
            onCheckedChange={onRollupChange}
          />
        </Row>
      )}
    </>
  );
}

/**
 * The Data type toggle (camera trap / underwater) of the three setup forms.
 *
 * Only the control: each form puts it in its own row layout, with
 * `SETTING_CAPTIONS.dataType` beside it. What choosing a type does to the
 * form is `applyDataType` in lib/data-type.ts, called by the form.
 */

import { SegmentedControl } from "../ui/segmented-control";
import { DATA_TYPE_OPTIONS, type DataType } from "../../lib/data-type";

export function DataTypeToggle({
  value,
  onChange,
}: {
  value: DataType;
  onChange: (value: DataType) => void;
}) {
  return (
    <SegmentedControl
      options={DATA_TYPE_OPTIONS.map((o) => ({
        value: o.value,
        title: o.label,
        label: o.label,
      }))}
      value={value}
      onChange={(v) => onChange(v as DataType)}
    />
  );
}

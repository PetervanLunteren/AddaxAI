/**
 * Render a grouped list of classification model `<SelectItem>`s, with
 * region headers and separators between regions.
 *
 * Used inside `<SelectContent>` next to the project's classification
 * model picker. Two render sites consume it today — CreateProjectDialog
 * and the project SettingsPage — and any future cls dropdown should
 * use it too so the visual treatment stays consistent.
 */

import { Fragment } from "react";

import { SelectGroup, SelectItem, SelectLabel, SelectSeparator } from "../ui/select";
import { groupClassificationModels } from "../../utils/cls-model-groups";
import type { ModelInfo } from "../../api/types";

interface Props {
  /** Cls models from /api/ml/models/classification, with the "none"
   *  entry filtered out (the parent renders that one explicitly). */
  models: ModelInfo[];
}

export function ClassificationModelGroupedItems({ models }: Props) {
  const groups = groupClassificationModels(models);
  return (
    <>
      {groups.map((group, idx) => (
        <Fragment key={group.region}>
          {idx > 0 && <SelectSeparator />}
          <SelectGroup>
            <SelectLabel className="pl-3 py-2 text-xs font-medium uppercase tracking-wider text-muted-foreground">
              {group.label}
            </SelectLabel>
            {group.models.map((model) => (
              <SelectItem key={model.model_id} value={model.model_id}>
                {model.emoji} {model.friendly_name}
                {model.description_short && (
                  <>
                    <br />
                    <span className="text-xs text-muted-foreground">
                      {model.description_short}
                    </span>
                  </>
                )}
              </SelectItem>
            ))}
          </SelectGroup>
        </Fragment>
      ))}
    </>
  );
}

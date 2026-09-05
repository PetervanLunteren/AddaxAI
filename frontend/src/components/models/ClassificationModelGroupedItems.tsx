/**
 * Render a grouped list of classification model `<SelectItem>`s, with
 * region headers and separators between regions.
 *
 * Used inside `<SelectContent>` next to the project's classification
 * model picker. Three render sites consume it (CreateProjectDialog, the
 * project SettingsPage, the folder-run step), and any future cls dropdown
 * should use it too so the visual treatment stays consistent. It is also
 * the one place the catalog's `min_app_version` gates the picker.
 */

import { Fragment } from "react";

import { SelectGroup, SelectItem, SelectLabel, SelectSeparator } from "../ui/select";
import { groupClassificationModels } from "../../utils/cls-model-groups";
import { useAppVersion } from "../../hooks/useAppVersion";
import { formatVersion, isReleaseBuild, satisfiesMinVersion } from "../../lib/version";
import type { ModelInfo } from "../../api/types";

interface Props {
  /** Cls models from /api/ml/models/classification, with the "none"
   *  entry filtered out (the parent renders that one explicitly). */
  models: ModelInfo[];
}

export function ClassificationModelGroupedItems({ models }: Props) {
  const groups = groupClassificationModels(models);
  // `min_app_version` is the release a model first works on (a new env,
  // a new non-label class). Older builds must not be able to pick it,
  // or they download and run a model their code cannot handle. The gate
  // only applies to a release build: a dev tree reports 0.0.0-dev and
  // would otherwise lose every model, and an unknown version (no /health
  // yet) is not a known mismatch.
  const currentVersion = useAppVersion();
  const tooOld = (model: ModelInfo) =>
    !!model.min_app_version &&
    isReleaseBuild(currentVersion) &&
    satisfiesMinVersion(currentVersion!, model.min_app_version) === false;
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
              <SelectItem
                key={model.model_id}
                value={model.model_id}
                disabled={tooOld(model)}
              >
                {model.emoji} {model.friendly_name}
                {tooOld(model) ? (
                  <>
                    <br />
                    <span className="text-xs text-muted-foreground">
                      Needs AddaxAI {formatVersion(model.min_app_version!)} or newer
                    </span>
                  </>
                ) : (
                  model.description_short && (
                    <>
                      <br />
                      <span className="text-xs text-muted-foreground">
                        {model.description_short}
                      </span>
                    </>
                  )
                )}
              </SelectItem>
            ))}
          </SelectGroup>
        </Fragment>
      ))}
    </>
  );
}

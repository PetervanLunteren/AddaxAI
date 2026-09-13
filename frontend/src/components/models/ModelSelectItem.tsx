/**
 * One model option inside a `ModelSelect`: emoji, name and short
 * description, for detection, classification and embedding alike.
 *
 * It is also the one place the catalog's `min_app_version` gates a
 * picker, so a model that needs a newer app is disabled the same way in
 * every dropdown (CreateProjectDialog, the project SettingsPage, the
 * folder-run step).
 */

import { SelectItem } from "../ui/select";
import { useAppVersion } from "../../hooks/useAppVersion";
import { formatVersion, isReleaseBuild, satisfiesMinVersion } from "../../lib/version";
import type { ModelInfo } from "../../api/types";

export function ModelSelectItem({ model }: { model: ModelInfo }) {
  // `min_app_version` is the release a model first works on (a new env,
  // a new non-label class). Older builds must not be able to pick it,
  // or they download and run a model their code cannot handle. The gate
  // only applies to a release build: a dev tree reports 0.0.0-dev and
  // would otherwise lose every model, and an unknown version (no /health
  // yet) is not a known mismatch.
  const currentVersion = useAppVersion();
  const tooOld =
    !!model.min_app_version &&
    isReleaseBuild(currentVersion) &&
    satisfiesMinVersion(currentVersion!, model.min_app_version) === false;
  const caption = tooOld
    ? `Needs AddaxAI ${formatVersion(model.min_app_version!)} or newer`
    : model.description_short;
  return (
    <SelectItem value={model.model_id} disabled={tooOld}>
      {model.emoji} {model.friendly_name}
      {caption && (
        <>
          <br />
          <span className="text-xs text-muted-foreground">{caption}</span>
        </>
      )}
    </SelectItem>
  );
}

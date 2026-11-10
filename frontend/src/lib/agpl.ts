/**
 * The licence step for models that run on the ultralytics package
 * (AGPL-3.0). The backend is the authority: a model whose catalog env is
 * "ultralytics" cannot be prepared until the user has accepted the licence
 * once (backend/app/ml/agpl_acceptance.py). These are the two facts the
 * frontend needs to follow it, kept in one place.
 */

import { ApiError } from "./api-client";
import type { ModelInfo } from "../api/types";

/** The env name the backend gates on. */
export const AGPL_ENV = "ultralytics";
export const AGPL_LICENCE_URL = "https://www.ultralytics.com/license";

/** True when the model runs on ultralytics. */
export function runsOnUltralytics(model: Pick<ModelInfo, "env">): boolean {
  return model.env === AGPL_ENV;
}

/** True when a prepare was refused because the licence is not accepted. */
export function isAgplRefusal(err: unknown): boolean {
  return (
    err instanceof ApiError &&
    err.status === 409 &&
    typeof err.detail === "object" &&
    err.detail !== null &&
    (err.detail as { error_kind?: unknown }).error_kind === "agpl_acceptance_required"
  );
}

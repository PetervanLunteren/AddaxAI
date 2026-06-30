/**
 * First-run setup wizard API.
 *
 * Backend gates the wizard: until env-addaxai-base is installed, the rest
 * of the app is unreachable. Default model weights are already on disk by
 * the time the wizard runs (copied from the bundle by app/main.py
 * lifespan), so the wizard's only real job is to install the conda env.
 */

import { api } from "../lib/api-client";

export interface SetupStatus {
  ready: boolean;
  models_installed: boolean;
  env_installed: boolean;
  install_in_progress: boolean;
  progress_pct: number;
  message: string;
  error: string | null;
  user_data_dir: string;
}

export const setupApi = {
  getStatus: () => api.get<SetupStatus>("/api/setup/status"),
  installEnv: () => api.post<{ status: string }>("/api/setup/install-env", {}),
  /** Wipe and rebuild specific environments (the env-drift "Update now"
   *  button). Same endpoint as installEnv, with force_envs set. */
  rebuildEnvs: (forceEnvs: string[]) =>
    api.post<{ status: string }>("/api/setup/install-env", {
      force_envs: forceEnvs,
    }),
};

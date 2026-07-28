/**
 * Cross-component state for the model-setup-required dialog.
 *
 * Two consumers open the same dialog:
 *  - AppLayout, on project open, auto-opens when readiness is not ok.
 *  - QueueCard, as a safety net before starting an analysis run.
 *
 * The dialog itself lives in AppLayout. QueueCard forces it back open
 * by calling `useModelSetupGate.getState().requestOpen()`. The flag
 * resets the moment the dialog closes, so the next analysis attempt
 * starts from a clean slate.
 */

import { create } from "zustand";

interface ModelSetupGate {
  forceOpen: boolean;
  requestOpen: () => void;
  reset: () => void;
}

export const useModelSetupGate = create<ModelSetupGate>((set) => ({
  forceOpen: false,
  requestOpen: () => set({ forceOpen: true }),
  reset: () => set({ forceOpen: false }),
}));

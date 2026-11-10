/**
 * The one way the UI starts a model download. It posts the prepare, and
 * when the backend refuses because the model runs on ultralytics and the
 * licence is not accepted yet, it shows the licence step, records the
 * acceptance and retries. Callers only see the outcome:
 *
 * - `onStarted(modelId, taskId)`: the download and env build are running;
 *   follow `taskId` with `useTaskProgress`.
 * - `onError(modelId, message)`: the request failed for another reason.
 * - `onDeclined(modelId)`: the user cancelled the licence step. Optional;
 *   a form has nothing to do, it simply stays as it was.
 *
 * Render `agplDialog` once next to the caller's own UI.
 */

import { useRef, useState, type ReactNode } from "react";

import { modelsApi } from "../api/models";
import { AgplLicenceDialog } from "../components/models/AgplLicenceDialog";
import { isAgplRefusal } from "../lib/agpl";

interface Handlers {
  onStarted: (modelId: string, taskId: string) => void;
  onError: (modelId: string, message: string) => void;
  onDeclined?: (modelId: string) => void;
}

export function useModelPrepare(handlers: Handlers): {
  prepare: (modelId: string) => Promise<void>;
  /** A request is in flight or the licence step is open. */
  isStarting: boolean;
  agplDialog: ReactNode;
} {
  // The latest handlers, so a prepare that resolves after a re-render
  // reports to the caller's current state, not to a stale closure.
  const latest = useRef(handlers);
  latest.current = handlers;

  const [requesting, setRequesting] = useState(false);
  const [waitingModelId, setWaitingModelId] = useState<string | null>(null);
  const [accepting, setAccepting] = useState(false);
  const [saveFailed, setSaveFailed] = useState(false);

  // A ref, not the state above: a double-click lands before React re-renders.
  const inFlight = useRef(false);

  const prepare = async (modelId: string) => {
    if (inFlight.current) return;
    inFlight.current = true;
    setRequesting(true);
    try {
      const response = await modelsApi.prepareModel(modelId);
      latest.current.onStarted(modelId, response.task_id);
    } catch (err) {
      if (isAgplRefusal(err)) {
        setSaveFailed(false);
        setWaitingModelId(modelId);
      } else {
        latest.current.onError(modelId, err instanceof Error ? err.message : String(err));
      }
    } finally {
      inFlight.current = false;
      setRequesting(false);
    }
  };

  const accept = async () => {
    const modelId = waitingModelId;
    if (modelId === null) return;
    setAccepting(true);
    try {
      await modelsApi.acceptAgpl();
    } catch {
      // Saving the choice is the whole point, so a failure keeps the
      // dialog open rather than retrying a prepare that is refused again.
      setSaveFailed(true);
      setAccepting(false);
      return;
    }
    setAccepting(false);
    setWaitingModelId(null);
    await prepare(modelId);
  };

  const decline = () => {
    const modelId = waitingModelId;
    setWaitingModelId(null);
    if (modelId !== null) latest.current.onDeclined?.(modelId);
  };

  return {
    prepare,
    isStarting: requesting || accepting || waitingModelId !== null,
    agplDialog: (
      <AgplLicenceDialog
        modelId={waitingModelId}
        accepting={accepting}
        saveFailed={saveFailed}
        onAccept={() => void accept()}
        onCancel={decline}
      />
    ),
  };
}

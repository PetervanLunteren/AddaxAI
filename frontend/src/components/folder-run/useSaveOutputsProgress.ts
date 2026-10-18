/**
 * The save-outputs worker's progress, decoded for `SaveOutputsProgress`.
 *
 * Every progress event from `folder_run_save_outputs_worker` carries the
 * ordered module list, the active module (or null at start and end) and
 * the index into that list, on top of the message and the per-module
 * fraction `useTaskProgress` already surfaces. Both places that run the
 * job (the folder-run Save step and the project image copies dialog)
 * need the same decoding, so it lives here once.
 */

import { useState } from "react";

import type { SaveOutputsResult } from "../../api/folder-runs";
import { useTaskProgress } from "../../hooks/useTaskProgress";

interface UseSaveOutputsProgressOptions {
  jobId: string | null;
  onComplete: (result: SaveOutputsResult) => void;
  onError: (message: string) => void;
  onCancelled: (message: string) => void;
}

export function useSaveOutputsProgress({
  jobId,
  onComplete,
  onError,
  onCancelled,
}: UseSaveOutputsProgressOptions) {
  const [modules, setModules] = useState<string[]>([]);
  const [currentModule, setCurrentModule] = useState<string | null>(null);
  const [moduleIndex, setModuleIndex] = useState(0);
  const [totalModules, setTotalModules] = useState(0);
  const [isCancelling, setIsCancelling] = useState(false);

  const progress = useTaskProgress({
    taskId: jobId,
    onProgress: (msg) => {
      const d = msg.data ?? {};
      if (Array.isArray(d.modules)) setModules(d.modules as string[]);
      if ("current_module" in d) {
        setCurrentModule(d.current_module as string | null);
      }
      if (typeof d.module_index === "number") {
        setModuleIndex(d.module_index);
      }
      if (typeof d.total_modules === "number") {
        setTotalModules(d.total_modules);
      }
    },
    onComplete: (data) => {
      onComplete(data as unknown as SaveOutputsResult);
      setIsCancelling(false);
    },
    onError: (msg) => {
      onError(msg);
      setIsCancelling(false);
    },
    onCancelled: (msg) => {
      onCancelled(msg);
      setIsCancelling(false);
    },
  });

  const cancel = () => {
    setIsCancelling(true);
    progress.cancel();
  };

  return {
    modules,
    currentModule,
    moduleIndex,
    totalModules,
    message: progress.message,
    phaseProgress: progress.phaseProgress,
    isCancelling,
    cancel,
  };
}

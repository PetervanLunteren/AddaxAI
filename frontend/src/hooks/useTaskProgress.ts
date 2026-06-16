/**
 * WebSocket hook for tracking task progress (model preparation, job execution, etc.)
 */

import { useCallback, useEffect, useRef, useState } from "react";
import { flushSync } from "react-dom";
import { API_BASE_URL } from "../lib/api-client";

export interface TqdmMetrics {
  raw_line: string;
  current?: number;
  total?: number;
  elapsed?: string;
  remaining?: string;
  rate?: number;
  unit?: string;
}

export interface ProgressMessage {
  type: "progress" | "complete" | "error" | "cancelled";
  job_id: string;
  message: string;
  progress?: number; // 0.0-1.0 (overall progress, for backward compatibility)
  phase?: "init" | "video_detection" | "video_classification" | "image_detection" | "image_classification" | "saving" | "embedding" | "finalize";
  phase_progress?: number; // 0.0-1.0 (progress within current phase)
  success?: boolean;
  data?: {
    deployment_index?: number;
    total_deployments?: number;
    video_count?: number;
    image_count?: number;
    has_classifier?: boolean;
    has_embedding?: boolean;
    metrics?: TqdmMetrics;
    [key: string]: unknown;
  };
}

export interface DeploymentContext {
  deploymentIndex: number;
  totalDeployments: number;
  videoCount: number;
  imageCount: number;
  hasClassifier: boolean;
  hasEmbedding: boolean;
}

interface UseTaskProgressOptions {
  taskId: string | null;
  onComplete?: (data?: Record<string, unknown>) => void;
  onError?: (message: string) => void;
  onCancelled?: (message: string) => void;
  /** Fired for every progress event with the raw message. Use this
   * when a worker emits custom fields under ``data`` that the hook
   * doesn't surface as first-class state (e.g. per-module job
   * checklists). */
  onProgress?: (message: ProgressMessage) => void;
}

export function useTaskProgress({
  taskId,
  onComplete,
  onError,
  onCancelled,
  onProgress,
}: UseTaskProgressOptions) {
  const [progress, setProgress] = useState(0);
  const [message, setMessage] = useState("");
  const [phase, setPhase] = useState<"init" | "video_detection" | "video_classification" | "image_detection" | "image_classification" | "saving" | "embedding" | "finalize" | null>(null);
  const [phaseProgress, setPhaseProgress] = useState(0);
  const [isConnected, setIsConnected] = useState(false);
  const [deploymentContext, setDeploymentContext] = useState<DeploymentContext | null>(null);
  const [metrics, setMetrics] = useState<TqdmMetrics | null>(null);
  const [computeDevice, setComputeDevice] = useState<string | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const updateTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const hasSetDeploymentContextRef = useRef<boolean>(false);
  const deploymentIndexRef = useRef<number | undefined>(undefined);
  const reconnectAttemptRef = useRef(0);
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const taskCompletedRef = useRef(false);
  const pingIntervalRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const pendingUpdateRef = useRef<{
    message: string;
    progress: number;
    phase: "init" | "video_detection" | "video_classification" | "image_detection" | "image_classification" | "saving" | "embedding" | "finalize" | null;
    phaseProgress: number;
    metrics: TqdmMetrics | null;
  } | null>(null);

  useEffect(() => {
    if (!taskId) {
      return;
    }

    // Reset all state for new task (prevents stale data from previous task)
    setProgress(0);
    setMessage("");
    setPhase(null);
    setPhaseProgress(0);
    setIsConnected(false);
    setDeploymentContext(null);
    setMetrics(null);
    setComputeDevice(null);

    // Reset refs for new task
    hasSetDeploymentContextRef.current = false;
    deploymentIndexRef.current = undefined;
    taskCompletedRef.current = false;
    reconnectAttemptRef.current = 0;

    // Derive WebSocket URL from API_BASE_URL (stays in sync with VITE_API_URL)
    const apiUrl = new URL(API_BASE_URL);
    const wsProtocol = apiUrl.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${wsProtocol}//${apiUrl.host}/ws/jobs/${taskId}`;

    function connect() {
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onopen = () => {
        console.debug(`[useTaskProgress] WebSocket connected for task ${taskId}`);
        setIsConnected(true);
        reconnectAttemptRef.current = 0;

        // Signal backend that WebSocket is ready — backend starts work on this signal
        ws.send(JSON.stringify({ type: "ready" }));

        // Start heartbeat ping to keep connection alive through proxies
        pingIntervalRef.current = setInterval(() => {
          if (ws.readyState === WebSocket.OPEN) {
            ws.send("ping");
          }
        }, 30000);
      };

      ws.onmessage = (event) => {
        try {
          const data: ProgressMessage = JSON.parse(event.data);

          if (data.type === "progress") {
            // Hand the raw progress event to the consumer first so
            // job-specific fields under `data.data` can flow into
            // bespoke state (per-module checklists, etc.) before
            // the hook updates its own internal state below.
            if (onProgress) {
              onProgress(data);
            }
            // Extract deployment context and update when deployment_index changes
            if (data.data?.deployment_index !== undefined) {
              const newContext = {
                deploymentIndex: data.data.deployment_index,
                totalDeployments: data.data.total_deployments ?? 1,
                videoCount: data.data.video_count ?? 0,
                imageCount: data.data.image_count ?? 0,
                hasClassifier: data.data.has_classifier ?? false,
                hasEmbedding: data.data.has_embedding ?? false,
              };

              // Update context only on first receipt or when deployment_index actually changes
              if (!hasSetDeploymentContextRef.current ||
                  deploymentIndexRef.current !== newContext.deploymentIndex) {
                hasSetDeploymentContextRef.current = true;
                deploymentIndexRef.current = newContext.deploymentIndex;
                setDeploymentContext(newContext);
              }
            }

            // Extract compute device. The server keeps this sticky in
            // its cached state, so a reconnecting client (page reload
            // mid-run) gets the value back in the replayed message.
            if (data.data?.compute_device) {
              setComputeDevice(data.data.compute_device as string);
            }

            // Store the pending update
            pendingUpdateRef.current = {
              message: data.message,
              progress: data.progress ?? 0,
              phase: data.phase ?? null,
              phaseProgress: data.phase_progress ?? 0,
              metrics: data.data?.metrics ?? null,
            };

            // Clear any existing timeout
            if (updateTimeoutRef.current) {
              clearTimeout(updateTimeoutRef.current);
            }

            // Schedule update with a small delay to allow browser to paint
            // This ensures visual updates are visible to the user
            updateTimeoutRef.current = setTimeout(() => {
              if (pendingUpdateRef.current) {
                const { message: msg, progress: prog, phase: ph, phaseProgress: phprog, metrics: met } = pendingUpdateRef.current;
                flushSync(() => {
                  setMessage(msg);
                  setProgress(prog);
                  setPhase(ph);
                  setPhaseProgress(phprog);
                  setMetrics(met);
                });
                pendingUpdateRef.current = null;
              }
            }, 16); // ~60fps (16ms) - faster updates for responsive progress bars
          } else if (data.type === "complete") {
            console.debug(`[useTaskProgress] COMPLETE received for job_id=${data.job_id}, taskId=${taskId}, message=${data.message}`);
            taskCompletedRef.current = true;

            // Clear any pending updates
            if (updateTimeoutRef.current) {
              clearTimeout(updateTimeoutRef.current);
            }

            flushSync(() => {
              setMessage(data.message);
              setProgress(1.0);
            });
            if (onComplete) {
              onComplete(data.data);
            }
          } else if (data.type === "error") {
            console.error(`[WS ${taskId}] ERROR:`, data.message);
            taskCompletedRef.current = true;

            // Clear any pending updates
            if (updateTimeoutRef.current) {
              clearTimeout(updateTimeoutRef.current);
            }

            flushSync(() => {
              setMessage(data.message);
            });
            if (onError) {
              onError(data.message);
            }
          } else if (data.type === "cancelled") {
            console.debug(`[useTaskProgress] CANCELLED received for job_id=${data.job_id}`);
            taskCompletedRef.current = true;

            if (updateTimeoutRef.current) {
              clearTimeout(updateTimeoutRef.current);
            }

            flushSync(() => {
              setMessage(data.message);
            });
            if (onCancelled) {
              onCancelled(data.message);
            }
          }
        } catch (error) {
          console.error("Failed to parse WebSocket message:", error);
        }
      };

      ws.onerror = (error) => {
        console.error("WebSocket error:", error);
        setIsConnected(false);
      };

      ws.onclose = () => {
        console.debug(`[useTaskProgress] WebSocket closed for task ${taskId}`);
        setIsConnected(false);

        // Clear heartbeat
        if (pingIntervalRef.current) {
          clearInterval(pingIntervalRef.current);
          pingIntervalRef.current = null;
        }

        // Only reconnect if:
        // 1. Task hasn't completed or errored
        // 2. This WS is still the active one (prevents stale WS from reconnecting
        //    after taskId changes — the cleanup resets taskCompletedRef before the
        //    old WS's onclose fires asynchronously)
        if (!taskCompletedRef.current && wsRef.current === ws) {
          const attempt = reconnectAttemptRef.current;
          const delay = Math.min(1000 * Math.pow(2, attempt), 10000); // 1s, 2s, 4s, 8s, cap 10s
          console.debug(`[useTaskProgress] Reconnecting in ${delay}ms (attempt ${attempt + 1})...`);
          reconnectAttemptRef.current = attempt + 1;
          reconnectTimerRef.current = setTimeout(connect, delay);
        }
      };
    }

    connect();

    // Cleanup on unmount or taskId change
    return () => {
      taskCompletedRef.current = true; // Prevent reconnection during cleanup

      if (reconnectTimerRef.current) {
        clearTimeout(reconnectTimerRef.current);
        reconnectTimerRef.current = null;
      }
      if (pingIntervalRef.current) {
        clearInterval(pingIntervalRef.current);
        pingIntervalRef.current = null;
      }
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        wsRef.current.close();
      }
      if (updateTimeoutRef.current) {
        clearTimeout(updateTimeoutRef.current);
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [taskId]); // Only re-run when taskId changes, not when callbacks change

  const cancel = useCallback(() => {
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN || !taskId) {
      console.warn("[useTaskProgress] cancel() called but socket not open");
      return;
    }
    ws.send(JSON.stringify({ type: "cancel", job_id: taskId }));
  }, [taskId]);

  return {
    progress,
    message,
    phase,
    phaseProgress,
    isConnected,
    deploymentContext,
    metrics,
    computeDevice,
    cancel,
  };
}

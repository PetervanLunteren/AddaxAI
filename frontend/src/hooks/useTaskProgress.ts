/**
 * WebSocket hook for tracking task progress (model preparation, job execution, etc.)
 */

import { useEffect, useRef, useState } from "react";
import { flushSync } from "react-dom";

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
  type: "progress" | "complete" | "error";
  job_id: string;
  message: string;
  progress?: number; // 0.0-1.0 (overall progress, for backward compatibility)
  phase?: "init" | "video_detection" | "video_classification" | "image_detection" | "image_classification" | "finalize";
  phase_progress?: number; // 0.0-1.0 (progress within current phase)
  success?: boolean;
  data?: {
    deployment_index?: number;
    total_deployments?: number;
    video_count?: number;
    image_count?: number;
    has_classifier?: boolean;
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
}

interface UseTaskProgressOptions {
  taskId: string | null;
  onComplete?: (data?: Record<string, unknown>) => void;
  onError?: (message: string) => void;
}

export function useTaskProgress({
  taskId,
  onComplete,
  onError,
}: UseTaskProgressOptions) {
  const [progress, setProgress] = useState(0);
  const [message, setMessage] = useState("");
  const [phase, setPhase] = useState<"init" | "video_detection" | "video_classification" | "image_detection" | "image_classification" | "finalize" | null>(null);
  const [phaseProgress, setPhaseProgress] = useState(0);
  const [isConnected, setIsConnected] = useState(false);
  const [deploymentContext, setDeploymentContext] = useState<DeploymentContext | null>(null);
  const [metrics, setMetrics] = useState<TqdmMetrics | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const updateTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const hasSetDeploymentContextRef = useRef<boolean>(false);
  const pendingUpdateRef = useRef<{
    message: string;
    progress: number;
    phase: "init" | "video_detection" | "video_classification" | "image_detection" | "image_classification" | "finalize" | null;
    phaseProgress: number;
    metrics: TqdmMetrics | null;
  } | null>(null);

  useEffect(() => {
    if (!taskId) {
      return;
    }

    // Reset deployment context ref for new task
    hasSetDeploymentContextRef.current = false;

    // Determine WebSocket URL based on current location
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${protocol}//${window.location.hostname}:8000/ws/jobs/${taskId}`;

    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    ws.onopen = () => {
      console.log(`[useTaskProgress ${new Date().toISOString()}] WebSocket connected for task ${taskId}`);
      setIsConnected(true);
    };

    ws.onmessage = (event) => {
      try {
        const data: ProgressMessage = JSON.parse(event.data);

        // Log all incoming WebSocket messages for debugging with timestamp
        const timestamp = new Date().toISOString().split('T')[1].slice(0, -1); // HH:MM:SS.mmm
        console.log(`[${timestamp}] [WS ${taskId}] ${data.type}:`, {
          progress: data.progress,
          message: data.message,
          phase: data.phase,
          phase_progress: data.phase_progress,
          success: data.success,
          data: data.data,
        });

        if (data.type === "progress") {
          // Extract deployment context from first progress message with data
          // Use ref to prevent multiple state updates
          if (data.data?.deployment_index !== undefined && !hasSetDeploymentContextRef.current) {
            const context = {
              deploymentIndex: data.data.deployment_index,
              totalDeployments: data.data.total_deployments ?? 1,
              videoCount: data.data.video_count ?? 0,
              imageCount: data.data.image_count ?? 0,
              hasClassifier: data.data.has_classifier ?? false,
            };
            console.log(`[useTaskProgress ${new Date().toISOString()}] Setting deployment context:`, context);
            hasSetDeploymentContextRef.current = true;
            setDeploymentContext(context);
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
          console.log(`[WS ${taskId}] ✅ COMPLETE MESSAGE RECEIVED`);

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
          console.error(`[WS ${taskId}] ❌ ERROR:`, data.message);

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
      console.log(`WebSocket closed for task ${taskId}`);
      setIsConnected(false);
    };

    // Cleanup on unmount or taskId change
    return () => {
      if (ws.readyState === WebSocket.OPEN) {
        ws.close();
      }
      if (updateTimeoutRef.current) {
        clearTimeout(updateTimeoutRef.current);
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [taskId]); // Only re-run when taskId changes, not when callbacks change

  return {
    progress,
    message,
    phase,
    phaseProgress,
    isConnected,
    deploymentContext,
    metrics,
  };
}

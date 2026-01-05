/**
 * WebSocket hook for tracking task progress (model preparation, job execution, etc.)
 */

import { useEffect, useRef, useState } from "react";
import { flushSync } from "react-dom";

export interface ProgressMessage {
  type: "progress" | "complete" | "error";
  job_id: string;
  message: string;
  progress?: number; // 0.0-1.0
  success?: boolean;
  data?: Record<string, unknown>;
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
  const [isConnected, setIsConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const updateTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const pendingUpdateRef = useRef<{ message: string; progress: number } | null>(null);

  useEffect(() => {
    if (!taskId) {
      return;
    }

    // Determine WebSocket URL based on current location
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${protocol}//${window.location.hostname}:8000/ws/jobs/${taskId}`;

    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    ws.onopen = () => {
      console.log(`WebSocket connected for task ${taskId}`);
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
          success: data.success,
          data: data.data,
        });

        if (data.type === "progress") {
          // Store the pending update
          pendingUpdateRef.current = {
            message: data.message,
            progress: data.progress ?? 0,
          };

          // Clear any existing timeout
          if (updateTimeoutRef.current) {
            clearTimeout(updateTimeoutRef.current);
          }

          // Schedule update with a small delay to allow browser to paint
          // This ensures visual updates are visible to the user
          updateTimeoutRef.current = setTimeout(() => {
            if (pendingUpdateRef.current) {
              const { message: msg, progress: prog } = pendingUpdateRef.current;
              flushSync(() => {
                setMessage(msg);
                setProgress(prog);
              });
              pendingUpdateRef.current = null;
            }
          }, 100); // 100ms delay allows browser to paint between updates
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
    isConnected,
  };
}

/**
 * WebSocket hook for tracking task progress (model preparation, job execution, etc.)
 */

import { useEffect, useRef, useState } from "react";

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

        if (data.type === "progress") {
          setMessage(data.message);
          if (data.progress !== undefined) {
            setProgress(data.progress);
          }
        } else if (data.type === "complete") {
          setMessage(data.message);
          setProgress(1.0);
          if (onComplete) {
            onComplete(data.data);
          }
        } else if (data.type === "error") {
          setMessage(data.message);
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
    };
  }, [taskId, onComplete, onError]);

  return {
    progress,
    message,
    isConnected,
  };
}

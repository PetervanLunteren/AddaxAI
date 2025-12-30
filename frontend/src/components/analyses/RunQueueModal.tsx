/**
 * Run Queue Modal Component
 *
 * Blocking modal that shows progress while processing queue.
 * For MVP: Mock progress with simulated stages.
 * Real implementation will be done in separate session.
 */

import { useState, useEffect } from "react";
import { Loader2, CheckCircle2, XCircle } from "lucide-react";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";

interface RunQueueModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  queueCount: number;
}

type Stage = "scanning" | "detection" | "classification" | "complete" | "error";

export function RunQueueModal({ open, onOpenChange, queueCount }: RunQueueModalProps) {
  const [currentStage, setCurrentStage] = useState<Stage>("scanning");
  const [currentItem, setCurrentItem] = useState(1);
  const [progress, setProgress] = useState(0);

  // Mock progress simulation
  useEffect(() => {
    if (!open) {
      // Reset when modal closes
      setCurrentStage("scanning");
      setCurrentItem(1);
      setProgress(0);
      return;
    }

    // Simulate processing stages
    const stages: Stage[] = ["scanning", "detection", "classification", "complete"];
    let stageIndex = 0;
    let itemProgress = 0;

    const interval = setInterval(() => {
      itemProgress += 2; // Increment progress

      if (itemProgress >= 100) {
        // Move to next stage or next item
        stageIndex++;

        if (stageIndex >= stages.length - 1) {
          // Completed all stages for current item
          if (currentItem < queueCount) {
            // Move to next item
            setCurrentItem((prev) => prev + 1);
            stageIndex = 0;
            itemProgress = 0;
          } else {
            // All items complete
            setCurrentStage("complete");
            setProgress(100);
            clearInterval(interval);
            return;
          }
        }

        setCurrentStage(stages[stageIndex]);
        itemProgress = 0;
      }

      setProgress(itemProgress);
    }, 100);

    return () => clearInterval(interval);
  }, [open, currentItem, queueCount]);

  // Stage display info
  const stageInfo: Record<Stage, { label: string; icon: React.ReactNode }> = {
    scanning: {
      label: `Scanning files... (${currentItem}/${queueCount})`,
      icon: <Loader2 className="h-5 w-5 text-blue-600 animate-spin" />,
    },
    detection: {
      label: `Running detection model... (${currentItem}/${queueCount})`,
      icon: <Loader2 className="h-5 w-5 text-blue-600 animate-spin" />,
    },
    classification: {
      label: `Running classification model... (${currentItem}/${queueCount})`,
      icon: <Loader2 className="h-5 w-5 text-blue-600 animate-spin" />,
    },
    complete: {
      label: "Queue processing complete!",
      icon: <CheckCircle2 className="h-5 w-5 text-green-600" />,
    },
    error: {
      label: "Processing failed",
      icon: <XCircle className="h-5 w-5 text-red-600" />,
    },
  };

  const currentInfo = stageInfo[currentStage];
  const isComplete = currentStage === "complete";
  const isError = currentStage === "error";

  return (
    <Dialog open={open} onOpenChange={isComplete || isError ? onOpenChange : undefined}>
      <DialogContent className="sm:max-w-md" hideClose={!isComplete && !isError}>
        <DialogHeader>
          <DialogTitle>Processing queue</DialogTitle>
          <DialogDescription>
            {isComplete
              ? "All deployments have been processed successfully."
              : "Please wait while we process your deployments..."}
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4 py-4">
          {/* Status */}
          <div className="flex items-center gap-3">
            {currentInfo.icon}
            <span className="text-sm font-medium">{currentInfo.label}</span>
          </div>

          {/* Progress bar */}
          {!isComplete && !isError && (
            <div className="space-y-2">
              <Progress value={progress} className="h-2" />
              <p className="text-xs text-gray-500 text-center">{progress.toFixed(0)}%</p>
            </div>
          )}

          {/* Mock notice */}
          <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-3 mt-4">
            <p className="text-xs text-yellow-800">
              <strong>Note:</strong> This is a mock progress simulation. Real processing will be
              implemented in a separate session.
            </p>
          </div>
        </div>

        <DialogFooter>
          {isComplete || isError ? (
            <Button onClick={() => onOpenChange(false)}>Close</Button>
          ) : (
            <Button variant="outline" onClick={() => onOpenChange(false)}>
              Cancel (mock)
            </Button>
          )}
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

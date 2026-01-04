/**
 * Model Preparation View Component
 *
 * Replaces modal content during model preparation.
 * Shows progress bar and real-time updates via WebSocket.
 */

import { useState } from "react";
import { Loader2, X } from "lucide-react";
import { Button } from "../ui/button";
import { Progress } from "../ui/progress";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "../ui/alert-dialog";
import {
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "../ui/dialog";

interface ModelPreparationViewProps {
  modelName: string;
  modelEmoji: string;
  progress: number; // 0.0-1.0
  message: string;
  onCancel: () => void;
}

export function ModelPreparationView({
  modelName,
  modelEmoji,
  progress,
  message,
  onCancel,
}: ModelPreparationViewProps) {
  const [showCancelDialog, setShowCancelDialog] = useState(false);

  const handleCancelClick = () => {
    setShowCancelDialog(true);
  };

  const handleConfirmCancel = () => {
    setShowCancelDialog(false);
    onCancel();
  };

  const progressPercent = Math.round(progress * 100);

  return (
    <>
      <DialogHeader>
        <DialogTitle>Preparing model...</DialogTitle>
        <DialogDescription>
          This may take several minutes. Please don't close this window.
        </DialogDescription>
      </DialogHeader>

      <div className="py-6 space-y-6 min-w-0">
        {/* Model Info */}
        <div className="flex flex-col items-center gap-3 text-center">
          <span className="text-5xl">{modelEmoji}</span>
          <div>
            <h3 className="font-semibold text-lg">{modelName}</h3>
            <p className="text-sm text-muted-foreground">Downloading and installing...</p>
          </div>
        </div>

        {/* Progress Bar */}
        <div className="space-y-2 min-w-0">
          <Progress value={progressPercent} className="h-2" />
          <div className="flex justify-end items-center text-sm">
            <span className="text-muted-foreground">{progressPercent}%</span>
          </div>
        </div>

        {/* Current Message */}
        <div className="bg-muted/50 rounded-md px-3 py-2 min-w-0">
          <p className="text-[11px] leading-none text-muted-foreground font-mono truncate">{message || "Preparing..."}</p>
        </div>

        {/* Info note */}
        <div className="text-xs text-muted-foreground text-center">
          <p>You can cancel and prepare the model later if needed.</p>
        </div>
      </div>

      {/* Cancel Button */}
      <div className="flex justify-end">
        <Button type="button" variant="outline" onClick={handleCancelClick}>
          Cancel
        </Button>
      </div>

      {/* Cancel Confirmation Dialog */}
      <AlertDialog open={showCancelDialog} onOpenChange={setShowCancelDialog}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Cancel preparation?</AlertDialogTitle>
            <AlertDialogDescription>
              Model preparation is {progressPercent}% complete. Canceling will discard partial
              downloads and you'll need to start over.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Continue preparing</AlertDialogCancel>
            <AlertDialogAction onClick={handleConfirmCancel}>Cancel preparation</AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </>
  );
}

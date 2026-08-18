/**
 * A confirmation dialog gated on typing an exact word, the shared shape
 * behind every "type X to confirm" destructive/heavy action (delete
 * project/site/deployment, reset app, restore backup, regroup events).
 *
 * The caller owns the action (`onConfirm`) and the warning body
 * (`children`); this component owns the typed-word gate, the reset on
 * close, and the Cancel/Confirm footer. `disabled` is ANDed with the
 * typed word for actions that need an extra precondition (e.g. a file
 * must be picked first).
 */

import { useEffect, useState } from "react";
import { AlertTriangle, Check, Copy, Loader2 } from "lucide-react";

import { Button } from "./button";
import { Callout } from "./callout";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "./dialog";
import { Input } from "./input";
import { Label } from "./label";

interface TypeToConfirmDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  title: string;
  description?: React.ReactNode;
  /** The exact string the user must type to enable the confirm button. */
  confirmWord: string;
  confirmLabel: string;
  /** Confirm button text while `isPending`. Defaults to `confirmLabel`. */
  pendingLabel?: string;
  onConfirm: () => void;
  isPending?: boolean;
  /**
   * Why the action failed, shown in the dialog. Pass
   * `mutation.error?.message ?? null` and nothing else is needed: the
   * dialog stays open on failure, so this is where the user is looking.
   * Without it a failed delete was completely silent, and the only trace
   * of a disconnected drive was a line in the console.
   */
  error?: string | null;
  /** Extra precondition ANDed with the typed word (e.g. a file was chosen). */
  disabled?: boolean;
  /** "destructive" (red, deletes data) or "warning" (amber, resets/regroups). */
  variant?: "destructive" | "warning";
  /** The warning body: callouts, lists, illustration. */
  children?: React.ReactNode;
}

export function TypeToConfirmDialog({
  open,
  onOpenChange,
  title,
  description,
  confirmWord,
  confirmLabel,
  pendingLabel,
  onConfirm,
  isPending = false,
  error = null,
  disabled = false,
  variant = "destructive",
  children,
}: TypeToConfirmDialogProps) {
  const [confirmText, setConfirmText] = useState("");
  const [justCopied, setJustCopied] = useState(false);
  const [elapsedSeconds, setElapsedSeconds] = useState(0);

  useEffect(() => {
    if (!open) setConfirmText("");
  }, [open]);

  // Count up while the action runs. These actions have no progress to
  // report (a delete is one database transaction, nothing is visible
  // from outside until it commits), so a clock is the honest signal
  // that the app is working rather than wedged.
  useEffect(() => {
    if (!isPending) return;
    const startedAt = Date.now();
    const timer = setInterval(
      () => setElapsedSeconds(Math.floor((Date.now() - startedAt) / 1000)),
      1000,
    );
    return () => {
      clearInterval(timer);
      setElapsedSeconds(0);
    };
  }, [isPending]);

  // Held back for a few seconds so a confirm that finishes quickly, which
  // is most of them, does not flash a timer at the user.
  const showElapsed = isPending && elapsedSeconds >= 3;
  const elapsedLabel = `${Math.floor(elapsedSeconds / 60)}:${String(
    elapsedSeconds % 60,
  ).padStart(2, "0")}`;

  const copyWord = async () => {
    try {
      await navigator.clipboard.writeText(confirmWord);
      setJustCopied(true);
      setTimeout(() => setJustCopied(false), 1400);
    } catch {
      /* ignore; the user can still type it by hand */
    }
  };

  const canConfirm = confirmText === confirmWord && !disabled && !isPending;
  const iconClass =
    variant === "warning" ? "text-amber-600" : "text-destructive";
  const buttonVariant = variant === "warning" ? "default" : "destructive";

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      {/* Not dismissable while the action runs. Cancel is already
          disabled, so leaving Escape, the X and the backdrop live meant
          the dialog could vanish mid-delete: the row stayed in the list
          with nothing to say it was going, and a second attempt started
          a second delete of the same thing. */}
      <DialogContent className="max-w-lg" nonDismissable={isPending}>
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <AlertTriangle className={`h-5 w-5 ${iconClass}`} />
            {title}
          </DialogTitle>
          {description && <DialogDescription>{description}</DialogDescription>}
        </DialogHeader>

        <div className="space-y-4">
          {children}

          {error && <Callout variant="error">{error}</Callout>}

          <div className="space-y-2">
            <Label htmlFor="type-to-confirm">
              Please type{" "}
              <span className="font-mono font-semibold bg-muted px-1.5 py-0.5 rounded">
                {confirmWord}
              </span>
              <button
                type="button"
                onClick={copyWord}
                title="Copy"
                aria-label={`Copy ${confirmWord}`}
                className="ml-1 inline-flex h-5 w-5 shrink-0 items-center justify-center rounded-sm align-middle text-muted-foreground hover:bg-accent hover:text-accent-foreground"
              >
                {justCopied ? (
                  <Check className="h-3.5 w-3.5" />
                ) : (
                  <Copy className="h-3.5 w-3.5" />
                )}
              </button>{" "}
              to confirm
            </Label>
            <Input
              id="type-to-confirm"
              value={confirmText}
              onChange={(e) => setConfirmText(e.target.value)}
              placeholder={confirmWord}
              autoComplete="off"
            />
          </div>
        </div>

        <DialogFooter className="sm:justify-between">
          <p
            className="flex items-center gap-2 text-sm text-muted-foreground"
            aria-live="polite"
          >
            {showElapsed && (
              <>
                <Loader2 className="h-3.5 w-3.5 animate-spin" />
                Still working, {elapsedLabel} so far
              </>
            )}
          </p>
          <div className="flex gap-2">
            <Button
              type="button"
              variant="outline"
              onClick={() => onOpenChange(false)}
              disabled={isPending}
            >
              Cancel
            </Button>
            <Button
              type="button"
              variant={buttonVariant}
              onClick={onConfirm}
              disabled={!canConfirm}
            >
              {isPending && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
              {isPending ? (pendingLabel ?? confirmLabel) : confirmLabel}
            </Button>
          </div>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

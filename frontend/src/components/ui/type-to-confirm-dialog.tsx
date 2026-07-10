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
import { AlertTriangle, Check, Copy } from "lucide-react";

import { Button } from "./button";
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
  disabled = false,
  variant = "destructive",
  children,
}: TypeToConfirmDialogProps) {
  const [confirmText, setConfirmText] = useState("");
  const [justCopied, setJustCopied] = useState(false);

  useEffect(() => {
    if (!open) setConfirmText("");
  }, [open]);

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
      <DialogContent className="max-w-lg">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <AlertTriangle className={`h-5 w-5 ${iconClass}`} />
            {title}
          </DialogTitle>
          {description && <DialogDescription>{description}</DialogDescription>}
        </DialogHeader>

        <div className="space-y-4">
          {children}

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

        <DialogFooter>
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
            {isPending ? (pendingLabel ?? confirmLabel) : confirmLabel}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

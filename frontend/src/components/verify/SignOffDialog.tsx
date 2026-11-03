/**
 * The one brake before a sign-off nobody has looked at in full: the
 * Files viewer's "Verify all" on a clip, and the grid's Verify when the
 * selected tiles hold a clip. A photo needs none, one glance covers
 * every box; a clip of 2,289 tracks seen through one frame of 17 does
 * (2026-11-03). Numbers in, so the dialog says what one click commits
 * to and nothing else. A plain confirm, not the type-to-confirm gate an
 * irreversible change gets: Unverify reverses this in one click. Enter
 * confirms, so a person who means it presses Enter twice.
 */

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

export function SignOffDialog({
  open,
  onOpenChange,
  clips,
  detections,
  unchecked,
  onConfirm,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  clips: number;
  detections: number;
  unchecked: number;
  onConfirm: () => void;
}) {
  const n = (v: number) => v.toLocaleString();
  return (
    <AlertDialog open={open} onOpenChange={onOpenChange}>
      <AlertDialogContent>
        <AlertDialogHeader>
          <AlertDialogTitle>Mark every detection correct?</AlertDialogTitle>
          <AlertDialogDescription>
            {clips === 1 ? "This clip holds" : `These ${n(clips)} clips hold`} {n(detections)}{" "}
            detections, {n(unchecked)} of them still unchecked. One click marks them all
            correct with the label the AI gave, as if you had looked at each one. Unverify
            takes that back.
          </AlertDialogDescription>
        </AlertDialogHeader>
        <AlertDialogFooter>
          <AlertDialogCancel>Cancel</AlertDialogCancel>
          <AlertDialogAction onClick={onConfirm}>Mark them all correct</AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  );
}

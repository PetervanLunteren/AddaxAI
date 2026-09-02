/**
 * The frame both Labels detail views sit in.
 *
 * Detections and Files open different things (one detection versus one
 * whole file) but a person is doing the same job in both, so they look
 * and work the same: picture on the left, what it is on the top right,
 * what you can do about it on the bottom right, navigation in between.
 *
 * Only the frame lives here. Zoom, panning, the canvas and every action
 * stay with the view that owns them, because those genuinely differ:
 * the crop view pans a static image, the file view hands the whole
 * panel to `AnnotationCanvas`, which does its own. The shell takes the
 * image panel's props rather than rendering blind, so the panel classes
 * are still written once.
 */

import type { ReactNode } from "react";
import type { HTMLAttributes } from "react";
import { ChevronLeft, ChevronRight, ChevronsRight, X } from "lucide-react";

import { Button } from "../ui/button";
import { Dialog, DialogContent, DialogTitle } from "../ui/dialog";

export type NavDirection = "prev" | "next" | "nextUnverified";

interface VerifyDetailShellProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  /** Screen-reader name for the dialog; nothing renders it visibly. */
  title: string;
  width: number | string;
  height: number | string;
  /** "3 of 48", shown left of the navigation arrows. */
  position?: string;
  onNavigate?: (direction: NavDirection) => void;
  /** Vertical strip of tools that act on the picture, e.g. draw mode
   *  and the species applied to a new box. Omitted where there are
   *  none. */
  toolbar?: ReactNode;
  /** Spread onto the left panel so the owner can attach its own ref,
   *  cursor and pointer handlers without restating the classes. */
  imagePanelProps?: HTMLAttributes<HTMLDivElement> & {
    ref?: React.Ref<HTMLDivElement>;
  };
  image: ReactNode;
  /** Scrolling column of cards: what this file is, when, where. */
  details: ReactNode;
  /** Pinned to the bottom, primary action last. */
  actions: ReactNode;
}

/**
 * One card in the sidebar's scrolling column.
 *
 * The same wrapper, heading size and padding were spelled out at every
 * card in both detail views, so the three could drift apart a class at
 * a time. Body padding lives here too, so a caller passes content and
 * nothing else.
 */
export function DetailCard({
  title,
  children,
}: {
  title: ReactNode;
  children: ReactNode;
}) {
  return (
    <div className="mx-3 mt-3 rounded-lg border bg-muted/40">
      <h3 className="px-3 pt-3 pb-2 text-sm font-semibold">{title}</h3>
      <div className="px-3 pb-3">{children}</div>
    </div>
  );
}

export function VerifyDetailShell({
  open,
  onOpenChange,
  title,
  width,
  height,
  position,
  onNavigate,
  toolbar,
  imagePanelProps,
  image,
  details,
  actions,
}: VerifyDetailShellProps) {
  const { ref: imagePanelRef, ...imagePanelRest } = imagePanelProps ?? {};

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent
        className="flex flex-col p-3 pr-0 gap-0 overflow-hidden [&>button.absolute]:hidden"
        style={{ width, height, maxWidth: "95vw", maxHeight: "95vh" }}
        onOpenAutoFocus={(e) => e.preventDefault()}
        aria-describedby={undefined}
      >
        <DialogTitle className="sr-only">{title}</DialogTitle>

        <div className="flex flex-1 min-h-0 overflow-hidden">
          {toolbar && (
            <div className="flex flex-col items-center gap-1 px-1.5 py-2 bg-white shrink-0">
              {toolbar}
            </div>
          )}

          <div
            ref={imagePanelRef}
            className="flex-1 flex select-none items-center justify-center overflow-hidden bg-black/95 min-h-0 p-2 rounded-lg"
            {...imagePanelRest}
          >
            {image}
          </div>

          <div className="w-80 bg-white flex flex-col shrink-0">
            <div className="flex items-center justify-between px-3 py-1.5 shrink-0">
              <div className="flex items-center gap-0.5">
                {position && (
                  <span className="text-xs text-muted-foreground mr-1">
                    {position}
                  </span>
                )}
                {onNavigate && (
                  <>
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-7 w-7"
                      onClick={() => onNavigate("prev")}
                      title="Previous"
                    >
                      <ChevronLeft className="h-4 w-4" />
                    </Button>
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-7 w-7"
                      onClick={() => onNavigate("next")}
                      title="Next"
                    >
                      <ChevronRight className="h-4 w-4" />
                    </Button>
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-7 w-7"
                      onClick={() => onNavigate("nextUnverified")}
                      title="Next unverified"
                    >
                      <ChevronsRight className="h-4 w-4" />
                    </Button>
                  </>
                )}
              </div>
              <Button
                variant="ghost"
                size="icon"
                className="h-7 w-7"
                onClick={() => onOpenChange(false)}
                title="Close"
              >
                <X className="h-4 w-4" />
              </Button>
            </div>

            <div className="flex-1 min-h-0 overflow-y-auto">{details}</div>

            <div className="px-3 py-3 space-y-2 shrink-0">{actions}</div>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}

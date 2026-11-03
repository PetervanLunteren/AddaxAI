/**
 * The chip over a picture while a frame of a clip is on its way: a
 * spinner and a word at once, and under it, once the wait has lasted
 * `SLOW_WAIT_MS`, the one line that says why (`SlowHint`). The Files
 * viewer's still, the card modal's still and the player's seek show
 * the same chip, so a wait looks the same everywhere. Until 2026-11-03
 * the two stills showed nothing for the first five seconds: the old
 * picture lost its boxes and sat there, which on a slow drive is twenty
 * seconds of looking broken before a sentence appeared.
 */

import { Loader2 } from "lucide-react";

import { SlowHint } from "./SlowHint";

export function FrameWait({
  label = "Loading frame…",
  testId,
}: {
  label?: string;
  testId?: string;
}) {
  return (
    <div
      // `w-max`: the card modal's picture box is 0 by 0 until the picture
      // lands, and an absolute child of it shrank to one word per line.
      className="pointer-events-none absolute left-2 top-2 z-10 w-max max-w-sm rounded bg-black/60 px-2 py-1 text-xs text-white"
      data-testid={testId}
    >
      <span className="flex items-center gap-1.5">
        <Loader2 className="h-3 w-3 animate-spin" />
        {label}
      </span>
      <SlowHint className="mt-1 text-left text-white/80" />
    </div>
  );
}

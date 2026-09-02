/**
 * Shared vocabulary for the keyboard-shortcut lists.
 *
 * The lists themselves live with the tab that owns those keys, so a
 * shortcut and its description sit next to the handler that implements
 * it. Only the shape and the modifier name are shared, and they live
 * here rather than in `LabelsKeyboardPopover` so that file exports a
 * component and nothing else (fast refresh needs that).
 */

export type Shortcut = readonly [keys: string, action: string];

/** "Cmd" on a Mac, "Ctrl" everywhere else, so both tabs spell the
 *  modifier the same way. */
export const MOD = navigator.platform.includes("Mac") ? "Cmd" : "Ctrl";

/** The undo hint on a button: the platform's own spelling, shared by
 *  the Detections bar and the Files viewer so the two read the same. */
export const UNDO_KBD = MOD === "Cmd" ? "⌘Z" : "Ctrl+Z";

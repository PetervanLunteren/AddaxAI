/**
 * The backdrop behind every modal: a dark, slightly blurred layer over the
 * page. One string for all of them (Dialog, AlertDialog, Sheet and the
 * hand-rolled popovers), so a confirm and a dialog never look different.
 * The data-state classes animate the Radix overlays and do nothing on a
 * plain div.
 */
export const MODAL_OVERLAY_CLASS =
  "fixed inset-0 z-50 bg-black/80 backdrop-blur-sm data-[state=open]:animate-in data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=open]:fade-in-0";

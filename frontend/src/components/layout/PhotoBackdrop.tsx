/**
 * The photo backdrop of the two full-screen pages, home and setup.
 *
 * One place for both the photo and the scrim over it, because the two
 * pages had the same scrim string copied into each of them and a third
 * copy was not worth it.
 *
 * The photo follows the user's data type: a camera trap user gets the
 * forest, an underwater user gets a BRUV frame. It is the only thing in
 * the app that changes with the data type beyond which models are
 * offered, and it should stay that way: same logo, same colours, same
 * wording, one photo.
 *
 * Underwater gets its own scrim. The reef photo is bright and pale
 * through the middle where the forest is dark, so the forest's scrim
 * leaves white cards and white text sitting on sand.
 */

import { type DataType, useDataType } from "../../lib/data-type";

type Screen = "home" | "setup";

const PHOTO: Record<DataType, Record<Screen, string>> = {
  camera_trap: {
    home: "/home-background.webp",
    setup: "/setup-background.webp",
  },
  // One frame for both screens for now; the camera trap side has two
  // because it had two before.
  underwater: {
    home: "/underwater-background.webp",
    setup: "/underwater-background.webp",
  },
};

const SCRIM: Record<DataType, string> = {
  camera_trap:
    "radial-gradient(120% 90% at 50% 0%, rgba(10,30,28,0.25), transparent 60%)," +
    "linear-gradient(180deg, rgba(8,22,20,0.55) 0%, rgba(8,22,20,0.30) 35%, rgba(8,22,20,0.66) 100%)",
  underwater:
    "radial-gradient(120% 90% at 50% 0%, rgba(4,26,38,0.30), transparent 60%)," +
    "linear-gradient(180deg, rgba(4,26,38,0.60) 0%, rgba(4,26,38,0.45) 35%, rgba(4,26,38,0.72) 100%)",
};

export function PhotoBackdrop({ screen }: { screen: Screen }) {
  // Subscribed, not read once: the setup screen changes the data type
  // without reloading, and the photo follows the toggle immediately.
  const dataType = useDataType();

  return (
    <>
      {/* Decorative, so no alt text. */}
      <div
        className="absolute inset-0 scale-105 bg-cover bg-center"
        style={{ backgroundImage: `url('${PHOTO[dataType][screen]}')` }}
      />
      <div className="absolute inset-0" style={{ background: SCRIM[dataType] }} />
    </>
  );
}

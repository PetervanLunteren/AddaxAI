// Screenshot manifest for the landing page and docs.
//
// Screenshots of the real app are NOT committed to this repo. They are hosted
// on GitHub's user-attachments CDN and referenced by absolute URL below. The
// source PNGs live in ~/Desktop/addaxai-docs-screenshots/upload (with a README
// mapping every file to its key here) in case they need re-uploading.
//
// Each URL is opaque, so there is no shared base path: to swap an image,
// replace that entry's `src`. A value starting with "http" is used as-is;
// anything else is treated as a path under /static (see AppShot).
//
// All shots are 1440 px wide at 2x, light theme, expanded sidebar.

export interface Shot {
  src: string;
  alt: string;
}

const A = "https://github.com/user-attachments/assets";

export const shots: Record<string, Shot> = {
  appHome: {
    src: `${A}/18fc43a8-3958-42b1-bb48-214544b948e8`,
    alt: "AddaxAI start screen with the two ways to work",
  },
  projects: {
    src: `${A}/4e986292-df91-4c2d-8189-a908cd268f61`,
    alt: "Projects overview with photo thumbnails and summary counts",
  },
  dashboard: {
    src: `${A}/5e99bc79-6e11-41c7-976d-45d32e29aa56`,
    alt: "Project dashboard: species counts, a season-long trend, and day and night activity",
  },
  verify: {
    src: `${A}/5242e0b0-b716-4bce-9ab4-46c9d80843b8`,
    alt: "Review grid with similar animals grouped, several selected, and the relabel action bar",
  },
  verifyOddOneOut: {
    src: `${A}/87d025ad-cee7-4dc2-aba2-52a1022b3f86`,
    alt: "Similar animals grouped together, with two differently coloured labels standing out as likely mistakes",
  },
  verifyOddOneOutSelected: {
    src: `${A}/b6c8d3d4-0fdb-4174-982a-adfe31a7bec1`,
    alt: "The two odd labels selected, ready to be corrected in one action",
  },
  counts: {
    src: `${A}/2626c447-c9f2-44fb-bab0-d6f0c223262f`,
    alt: "Counts page: how many animals of each species were seen per event",
  },
  processQueue: {
    src: `${A}/761518a9-272f-4f51-974f-cd58a7dc756e`,
    alt: "Process page: a new deployment being added and five waiting in the queue",
  },
  sites: {
    src: `${A}/1dbd8b50-2d5c-43c6-bc09-c7b57a615760`,
    alt: "Sites table with habitat, elevation, notes, and tags",
  },
  deployments: {
    src: `${A}/612f00ae-51fc-44e8-aabf-bf093c2d81a2`,
    alt: "Deployments table with camera periods, notes, and tags",
  },
  export: {
    src: `${A}/b718f847-a7f8-40e3-94ca-498bc3049263`,
    alt: "Export page with format options including Camtrap DP",
  },
  settings: {
    src: `${A}/560b968f-d649-4fb6-b213-449b1853dec9`,
    alt: "Project settings: models, thresholds, and processing options",
  },
  map: {
    src: `${A}/44f06340-8d56-488c-86aa-ecf2a96b6e93`,
    alt: "Map of camera sites shaded by how often animals were seen",
  },
  mapSatellite: {
    src: `${A}/a05a9e40-11d0-4073-a365-11d3f06c10a0`,
    alt: "The same site map on satellite imagery, showing forest cover and terrain",
  },
  activity: {
    src: `${A}/27a5b0a5-a7ec-4cad-b1ba-ae674c3e076a`,
    alt: "Daily activity of two species compared, with the overlap coefficient",
  },
  timeline: {
    src: `${A}/a15c0618-c6ab-4ef5-8741-1e515c5ce8f8`,
    alt: "Deployment timeline showing when each camera was active",
  },
  confusionMatrix: {
    src: `${A}/9d21b67f-fce4-4847-be94-d037aa8d345d`,
    alt: "Confusion matrix comparing the AI's labels with the confirmed labels",
  },
  perClass: {
    src: `${A}/c66899e5-d0b0-45a4-aabc-9b553cc266c1`,
    alt: "Per-species accuracy of the AI against the confirmed labels",
  },
  folderRun1Setup: {
    src: `${A}/1cc84277-0fe1-4e2f-a575-bad123989a22`,
    alt: "Analyse a folder, step 1: pick the folder and the models",
  },
  folderRun2Labels: {
    src: `${A}/58e9cdae-af6c-47f3-8ea1-da0d1fb4119e`,
    alt: "Analyse a folder, step 2: check the labels the AI gave",
  },
  folderRun3Save: {
    src: `${A}/a22760e2-bc9b-4c3d-a047-69571b7f1984`,
    alt: "Analyse a folder, step 3: choose what to save",
  },
};

// True when a slot has no real image wired in yet (all are live now, but the
// AppShot component still checks so a future placeholder shows its caption).
export function isPlaceholder(key: string): boolean {
  const src = shots[key]?.src;
  return !src || src.endsWith("screenshot-placeholder.svg");
}

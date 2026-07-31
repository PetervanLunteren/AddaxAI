// Screenshot manifest for the landing page and docs.
//
// Screenshots of the real app are committed as WebP under docs/static/img and
// referenced by local path below, so the docs are self-contained with no
// external CDN dependency. The source PNGs live in
// ~/Desktop/addaxai-docs-screenshots/upload (with a README mapping every file
// to its key here). To update a shot: re-export the PNG there, convert it to
// WebP (e.g. `magick in.png -quality 85 out.webp`), and drop it into
// static/img under the same filename.
//
// A src starting with "http" is used as-is; a local path is served from
// /static (see AppShot). All shots are 1440 px wide at 2x, light theme,
// expanded sidebar.

export interface Shot {
  src: string;
  alt: string;
}

export const shots: Record<string, Shot> = {
  appHome: {
    src: "/img/app-home.webp",
    alt: "AddaxAI start screen with the two ways to work",
  },
  projects: {
    src: "/img/projects-list.webp",
    alt: "Projects overview with photo thumbnails and summary counts",
  },
  dashboard: {
    src: "/img/project-dashboard.webp",
    alt: "Project dashboard: species counts, a season-long trend, and day and night activity",
  },
  verify: {
    src: "/img/project-labels-similarity-selected.webp",
    alt: "Review grid with similar animals grouped, several selected, and the relabel action bar",
  },
  verifyOddOneOut: {
    src: "/img/project-labels-similarity-mixed.webp",
    alt: "Similar animals grouped together, with two differently coloured labels standing out as likely mistakes",
  },
  verifyOddOneOutSelected: {
    src: "/img/project-labels-similarity-mixed-selected.webp",
    alt: "The two odd labels selected, ready to be corrected in one action",
  },
  counts: {
    src: "/img/project-counts.webp",
    alt: "Counts page: how many animals of each species were seen per event",
  },
  countsEvent: {
    src: "/img/counts-event.webp",
    alt: "An event opened on the Counts page: the best frame, the film strip of the event below, and the count editor with a Confirm button",
  },
  processQueue: {
    src: "/img/project-process-queue.webp",
    alt: "Process page: a new deployment being added and five waiting in the queue",
  },
  sites: {
    src: "/img/project-sites.webp",
    alt: "Sites table with habitat, elevation, notes, and tags",
  },
  deployments: {
    src: "/img/project-deployments.webp",
    alt: "Deployments table with camera periods, notes, and tags",
  },
  export: {
    src: "/img/project-export.webp",
    alt: "Export page with format options including Camtrap DP",
  },
  map: {
    src: "/img/insights-map.webp",
    alt: "Map of camera sites shaded by how often animals were seen",
  },
  mapSatellite: {
    src: "/img/insights-map-satellite.webp",
    alt: "The same site map on satellite imagery, showing forest cover and terrain",
  },
  activity: {
    src: "/img/insights-activity-overlap.webp",
    alt: "Daily activity of two species compared, with the overlap coefficient",
  },
  timeline: {
    src: "/img/insights-timeline.webp",
    alt: "Deployment timeline showing when each camera was active",
  },
  confusionMatrix: {
    src: "/img/insights-confusion-matrix.webp",
    alt: "Confusion matrix comparing the AI's labels with the confirmed labels",
  },
  perClass: {
    src: "/img/insights-per-class-performance.webp",
    alt: "Per-species accuracy of the AI against the confirmed labels",
  },
  folderRun1Setup: {
    src: "/img/folder-run-1-setup.webp",
    alt: "Analyse a folder, step 1: pick the folder and the models",
  },
  folderRun2Labels: {
    src: "/img/folder-run-2-labels.webp",
    alt: "Analyse a folder, step 2: check the labels the AI gave",
  },
  folderRun3Save: {
    src: "/img/folder-run-3-save.webp",
    alt: "Analyse a folder, step 3: choose what to save",
  },
  folderRunSaved: {
    src: "/img/folder-run-saved.webp",
    alt: "Folder run finished: open the output folder, or turn it into a project",
  },
};

// True when a slot has no real image wired in yet (all are live now, but the
// AppShot component still checks so a future placeholder shows its caption).
export function isPlaceholder(key: string): boolean {
  const src = shots[key]?.src;
  return !src || src.endsWith("screenshot-placeholder.svg");
}

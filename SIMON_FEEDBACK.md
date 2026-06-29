# Simon Kravis beta 4 feedback

Notes from "AddaxAI v 17 Notes part 2.docx" plus the email. Each point has a status and a place for our findings.

Status legend: `todo` / `investigating` / `verdict` (assessed, awaiting go) / `done` / `wontfix` / `upstream`.

## Bugs / robustness

### B1 - Upgrade install failed
New version would not install over the old one (error, no log kept). Had to uninstall, wipe `C:/Simon/AddaxAI`, and re-download models on the new install.
- status: todo
- notes:

### B2 - Large dirty dataset failed late
139k-image deployment ran 9h, then failed in the last phase at 16h. Folder had wild EXIF date anomalies (e.g. consecutive images 8 months apart). Suggests a time-anomaly check (notes EXIF reads are slow on Windows).
- status: todo
- notes:

### B3 - Cancel does not stop the process
Cancel button did not stop processing. After killing all tasks, restarting, and deleting the deployment, the app hung.
- status: todo
- notes:

### B4 - Error adding multiple deployments to queue
Adding multiple camera deployments: error when adding to queue after the first was added. Worked after a restart.
- status: todo
- notes:

### B5 - Wrong reason in rejection warning
Rejection warning says "invalid time stamps found" even though EXIF is valid. Real reason: time range was only ~1.8h. Warning text should reflect the real cause.
- status: already solved (predates current code)
- notes: Both halves are fixed in the current build. (1) The "invalid time stamps found" string
  no longer exists anywhere. The datetime warning is now the accurate, non-blocking "Some files
  have no capture date" (`FolderSelector.tsx:269-302`). (2) The old 3-hour minimum-span rejection
  was removed (commit b9f2373); `folder_scanner.py:385-390` documents that dates come from EXIF
  DateTimeOriginal not mtime, so a narrow span like Simon's 1.8h is accepted, not rejected. No
  code change needed.

### B6 - Event spans longer than independence interval
An event sometimes covers a time range larger than the independence interval set in Settings. First images in the event have no detections.
- status: verdict (working as designed, matches the field standard; recommend caption tweak only)
- notes: Not a bug, and not a divergence from the field. The interval is the max gap between
  two consecutive captures, not the max total event duration
  (`app/services/event_clustering.py`). A chain of closely spaced captures (each less than the
  interval apart) cumulates into one event spanning hours. "First images have no detections"
  is also expected: an event holds every file in the time window, not only files with
  detections.

  This is exactly how the de facto standard and every major platform define it:
  - Camtrap-DP / GBIF best-practice guide: "A sequence not only combines images resulting from
    a single trigger, but also consecutive triggers that fall within a preset independence
    interval (e.g. 120s) ... continued activity is captured in a single sequence/event." No
    total-duration cap; the guide notes a sequence can extend as long as activity stays within
    successive intervals.
  - Camtrap-DP `sequenceInterval`: "Maximum number of seconds between timestamps of successive
    media files to be considered part of a single sequence." Gap-based, no cap.
  - Wildlife Insights: groups images taken within 60s of each other into one sequence.
  - Agouti: groups media into sequences by a time interval (e.g. 2 min).
  - Camelot: temporal discretisation default 600s between independent triggers.
  - WildTrax: "series gap" (default 5 min); detections separated by more than the gap are
    separate series. Gap-based chaining.
  - Research convention (the "independence interval" / "quiet period" / 30-min rule): the
    threshold is the gap between consecutive detections, typically 1-60 min depending on
    species; no cap on how long a continuous run of activity lasts.

  Two distinct concepts share this one number, worth knowing: (a) sequence/event grouping for
  data structuring (species-agnostic, what AddaxAI's events are, what Camtrap-DP calls a
  sequence), and (b) the per-species independence filter used in activity/abundance analysis.
  AddaxAI uses gap-grouped events as the unit of observation, which is the standard data-layer
  behaviour. A hard cap on total event duration would diverge from the standard and hurt
  comparability with other datasets, so do not add one.

  Recommendation: leave the logic as is. Only adjust the interval caption to say it is the max
  gap between consecutive captures (not the total event length), and that an event holds all
  captures in the window including empty ones. Optional: consider renaming "independence
  interval" to "sequence interval" to match Camtrap-DP, but the current term is also widely
  used, so this is a judgement call (parked for Peter).

### B7 - Site created before deployment not shown
Creating a site before any deployment: the sites list shows empty, but the site appears in the deployment "Camera site" picker.
- status: todo
- notes:

### B8 - Bad lat/long blurs the screen
Entering an incorrect lat or long in the site form just blurs the screen instead of explaining what is wrong.
- status: todo
- notes:

### B9 - Blue cast on filmstrip thumbnails
Some filmstrip thumbnails in Verify, Events tab have a blue cast. Clicking a detection clears the cast.
- status: todo
- notes:

### B10 - Empty-classification filtering inconsistent
In Verify Captures, selecting "All" hides Empty-classified images (only "Show only empty" reveals them), yet events still render showing only-empty images.
- status: todo
- notes:

### B11 - Label tree counts ignore site filter
Detection counts in the Labels tree are not filtered by the selected site.
- status: todo
- notes:

### B12 - Remapped drive letter shows no images
If data moves to a drive with a different letter, no images show because absolute paths are stored in `files`. Wants a check that the mapped drive is present, with a warning.
- status: todo
- notes:

## UX / wording

### U1 - "New project" in the app menu
Add a "New..." menu option to create a project. Currently only reachable via the "Back to project" link at the bottom.
- status: todo
- notes:

### U2 - Smooth the detection ETA
Image-detection remaining time fluctuates a lot (6h to 8h within seconds on 190k files). Wants smoothing.
- status: todo
- notes:

### U3 - Warn when no classification model
Warn the user when no classification model is selected, since it is the default and users expect species to be identified.
- status: done
- notes: Discussed the default first. A warning on the untouched default is an anti-pattern
  (don't cry wolf), and `null` is a deliberate first-run default (defaulting to SpeciesNet would
  force a heavy `pytorch` env build + weights download on every new user, since setup only
  pre-installs MD5A + DINOv2 and the run is gated on all models ready). Kept `null` default
  (option C) and added a neutral info note instead of a warning. New shared component
  `components/models/NoClassifierNotice.tsx` (info Callout, compact): "Without a classification
  model, AddaxAI detects animals but does not identify the species. You can label them yourself
  in the Labels section." Shown when "none" is selected on all three selection surfaces: folder-run step,
  create-project dialog, project settings. Typecheck clean.

### U4 - Map key terminology
Map key calls the rate "Observations" but they look like events; should be labelled as such. Also questions per trap-night vs per trap-day (or per trap-week).
- status: todo
- notes:

### U5 - Select all / clear all should be buttons
"Filter by label" Select all is always checked and Clear all always unchecked; clicking still performs the action. Should be buttons.
- status: done
- notes: They were already real Buttons in `TreeSelector.tsx` (the tree rendered by
  LabelFilterModal "Filter by label"), which is why clicking performed the action. What made
  them read as checkboxes was the icon choice: CheckSquare (ticked box) on Select all, Square
  (empty box) on Clear all. Swapped to CheckCheck (Select all / Include all) and X (Clear all /
  Exclude all) so they read as actions, not toggles. Typecheck clean.

### U6 - Relabel confirmation wording
After relabelling multiple observations, the message's last line would be clearer as `Switch Verified to "All" to see them`.
- status: done
- notes: Empty-state in `LabelsTab.tsx:1294`. Changed "Set the verification filter to "All" to
  see them." to "Switch the Verified filter to "All" to see them." so it names the actual
  control on the bar (the "Verified" select). Typecheck clean.

### U7 - Squamata appears twice in labels tree
Squamata shows twice in the labels tree. Simon notes it is a SpeciesNet issue; may still want display dedup.
- status: todo
- notes:

## Feature requests

### F1 - Drill down from dashboard to verify
Click or right-click a species in the dashboard summary to open Verify pre-filtered to that species. Quick way to fix frequent misidentifications.
- status: todo
- notes:

### F2 - User-defined label mapping
Map scientific names onto custom (common) names, many-to-one, to also fix misidentifications (e.g. deer recognized as cattle). Events would use the mapped values. Overlaps F3, F4.
- status: todo
- notes:

### F3 - Add user-defined labels
Allow adding user-defined labels (e.g. Tiliqua rugosa / shingleback, missing from the Australian set). Overlaps F2.
- status: todo
- notes:

### F4 - Common vs scientific name toggle
Global toggle to show common vs scientific names. Common names live in `label_taxonomy.display_name`, some with apostrophes. Already on the TODO nice-to-haves.
- status: todo
- notes:

### F5 - Copy sites/deployments into a new project
Copy sites and deployments from an existing project so a new project (e.g. to compare models) does not need full re-entry.
- status: todo
- notes:

## Questions / upstream

### Q1 - How are SpeciesNet country filters applied?
Email question: he saw no country field in the taxonomy table. Explain where geofencing lives.
- status: done (answer for email)
- notes: There is no country column because the geofence is not in the DB. It ships
  inside the SpeciesNet model directory as `geofence_release.*.json` (taxonomy key ->
  allowed `{country: [states]}`) plus a `*.labels.txt` file. When a project sets a country
  (and optional state), `compute_excluded_classes()` in `app/ml/geofence.py` reads those
  files and produces the list of labels not allowed there. That list is stored on the project
  as `project.excluded_classes` (`routers/projects.py:140-157`). Postprocessing
  (`app/ml/postprocessing.py`) then filters/rolls up those excluded labels. So the country
  filter is applied at postprocessing time via `excluded_classes`, derived from the model's
  geofence JSON, not from a taxonomy column.

### Q2 - How is the mostly-empty 270-image event generated?
A 270-image event was mostly no-detection images, with a few low-confidence detections (baits misread as M. gallopavo). Asks whether events are built by image similarity. May tie into B6 and B10.
- status: done (answer)
- notes: Events are not image similarity. They are pure time-gap clustering
  (`app/services/event_clustering.py`): inside one folder, a new event starts whenever the
  gap between two consecutive files exceeds the independence interval. Every file in that
  window joins the event, including empty / no-detection images. So the 270-image event is a
  continuous burst (baits triggering near-constant captures): each image sits less than the
  interval after the previous one, so they chain into one long event even though most are
  empty. This is the same mechanism behind B6.

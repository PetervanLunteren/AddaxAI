# Simon Kravis beta 4 feedback

Notes from "AddaxAI v 17 Notes part 2.docx" plus the email. Each point has a status and a place for our findings.

Status legend: `todo` / `investigating` / `verdict` (assessed, awaiting go) / `done` / `wontfix` / `upstream`.

## Bugs / robustness

### B1 - Upgrade install failed
New version would not install over the old one (error, no log kept). Had to uninstall, wipe `C:/Simon/AddaxAI`, and re-download models on the new install.
- status: likely fixed (Peter)
- notes: Peter: this was probably a DB migration issue on upgrade, and should be fixed now. No log
  was kept so it can't be reproduced precisely. Migration robustness has since been hardened (see
  DEVELOPERS.md "Database migrations": idempotent column drops guarded by presence checks, legacy
  DBs stamped at head, pre-upgrade backups). Treat as resolved unless it recurs with a log.

### B2 - Large dirty dataset failed late
139k-image deployment ran 9h, then failed in the last phase at 16h. Folder had wild EXIF date anomalies (e.g. consecutive images 8 months apart). Suggests a time-anomaly check (notes EXIF reads are slow on Windows).
- status: diagnosed (fix scoped, awaiting decision); shares root cause with B3, B4
- notes: Deep dive. The screenshots show the run stuck at Saving -> "Merging results... 0%" with
  6 AddaxAI.exe processes (image1/2). Findings:
  1. Detection worker is a coroutine on the MAIN event loop (asyncio.create_task in ws_manager
     handle_ready); the save-phase calls are synchronous and not offloaded, so they block the loop
     for their whole duration -> backend unresponsive (this is what ties in B3/B4).
  2. Prime cost: load_json_to_database does `absolute_path.exists()` + `.stat()` per file
     (json_pipeline.py:286, for size_bytes) -> 2 filesystem syscalls x 139k on a slow external
     drive. Linear but the constant explodes on network/external storage; matches the ~7h stuck
     save. Everything else in the per-image loop is from the in-memory JSON.
  3. Whole merged JSON is re-loaded from disk 2-3x (trim at :553, taxonomy resolve at :591) ->
     memory churn on a huge JSON.
  4. No progress updates across merge -> trim -> taxonomy -> DB load, so it shows frozen
     "Merging results 0%".
  5. No O(n^2): DB load is one pass + one commit, merge is linear, event clustering is O(n log n);
     the 8-month date jumps just make more events, not a hang.
  Scoped fix (4 parts): offload save phase to a thread w/ dedicated DB session (unblocks B3/B4);
  drop/cheapen the per-file stat (removes the multi-hour bottleneck); pass the in-memory JSON
  between trim/taxonomy/load instead of re-reading 3x; add intermediate save-phase progress.

  IMPLEMENTED (core 2 parts):
  - Per-file stat: new `_safe_file_size()` (json_pipeline.py) does a single stat() in try/except
    instead of exists()+stat(), halving per-file filesystem syscalls during DB load. Used at the
    File() build.
  - Offload: merge_json_files and the 139k-file DB load now run via `asyncio.to_thread` in the
    detection worker, so the event loop stays responsive (delete / add-to-queue no longer hang).
    The load runs through new `load_json_to_database_owned_session()` which creates its OWN session
    inside the thread (SQLite check_same_thread=True forbids reusing the loop's session). Safe
    because taxonomy is committed on the main session first and passed in as plain dicts; the
    `set_sqlite_pragma` connect listener is on the Engine base class so the thread engine also gets
    WAL/FK/seeded_hash; deployment is committed before load so FKs hold; WAL makes the thread's
    commit visible; `db.expire_all()` after the thread so the main session's later queries (taxonomy
    link, postprocessing) see the new rows.
  Tests: 430 integration+ml + 107 pipeline/worker tests pass; ruff clean. Correctness testable on a
  small folder run; the perf win (and B3/B4 responsiveness) wants validation on a large/slow-drive
  set like Simon's 139k.
  ACTION: ask Simon to re-run his large external-drive dataset (the 139k Ginini set) and confirm
  the save phase no longer hangs and the app stays responsive (delete / add-to-queue work) during
  it. This is the real validation we can't do locally.
  NOT done (deferred polish): offload trim + the taxonomy JSON re-read; pass the in-memory JSON
  between steps instead of 3 reads; fine-grained progress inside the load; a cooperative cancel
  check inside the load loop (would let Cancel interrupt the in-process save, B3 part b).

### B3 - Cancel does not stop the process
Cancel button did not stop processing. After killing all tasks, restarting, and deleting the deployment, the app hung.
- status: largely fixed via B2 offload (delete no longer blocked); cancel-during-save still in-process
- before-status: diagnosed (same root cause as B2); see B2 notes
- notes: Two parts. (a) Cancel during the save phase can't take effect because cancellation kills
  tracked subprocesses, but the save phase is in-process synchronous Python on the event loop (no
  subprocess to kill, and the loop is blocked so the cancel request isn't even processed). (b) The
  delete-deployment "Deleting..." hang is the same event-loop block plus SQLite single-writer lock
  contention from the worker's open transaction. Fixing B2 (offload save to a thread) restores
  responsiveness so cancel/delete are served; a cooperative cancel check inside the save loop would
  also let cancel interrupt the in-process work. Tie to B2's fix.

### B4 - Error adding multiple deployments to queue
Adding multiple camera deployments: error when adding to queue after the first was added. Worked after a restart.
- status: fixed via B2 offload (backend stays responsive during the save phase)
- before-status: diagnosed (same root cause as B2); see B2 notes
- notes: "Failed to add to queue: API request failed: Failed to fetch" (image5) is the frontend
  fetch timing out because the backend event loop was blocked by the in-process save phase of the
  first deployment's run (see B2). Not a queue-logic bug; it's backend unresponsiveness during the
  heavy synchronous save. Offloading the save phase (B2 fix) resolves it. "Worked after restart"
  fits: once the stuck run was gone, the loop was free again.

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
- status: done
- notes: Not a backend issue (get_sites_with_stats uses an outer join, so 0-deployment sites are
  returned). Cache bug: the blanket `invalidateProjectData` (used by the create/edit site
  mutation) invalidated `["sites", projectId]` (the deployment picker's key) but not
  `["sites-with-stats", projectId]` (the SitesPage table's key). React Query matches by prefix and
  "sites" != "sites-with-stats", so the table never refreshed after a create, hence "shows in the
  deployment picker but not in the sites list". Fix: added `["sites-with-stats", projectId]` to
  `invalidate-project.ts`. Correct for all callers (deployment delete also changes the table's
  deployment_count). Typecheck clean.

### B8 - Bad lat/long blurs the screen
Entering an incorrect lat or long in the site form just blurs the screen instead of explaining what is wrong.
- status: already solved (no change)
- notes: Data integrity is covered: the site schema is `latitude z.number().min(-90).max(90)` and
  `longitude min(-180).max(180)`, so an out-of-range value blocks the save and renders a red error
  under the field (AddSiteModal.tsx:311,325). Wrong coords can't be saved and the form does say
  what's wrong. The "blur" Simon saw is just the standard modal backdrop (`backdrop-blur-sm` in
  ui/dialog.tsx), not a failure. Validation post-dates beta 4. Decided not to gold-plate the
  message wording / on-blur timing. Left as is.

  Follow-up (Peter's call): also block 0,0. Coords are required and the form defaults to 0,0, so a
  forgotten location saved a site at null island. Added a 0,0 reject on both sides: frontend zod
  `.refine` on the site schema (message "0, 0 is not allowed. This is probably an error. Enter the
  real coordinates.", shown under latitude) and a backend `model_validator` on SiteBase + SiteUpdate
  via a shared `_reject_null_island` helper. Valid coords and partial updates still pass; 45 site
  tests green, frontend typecheck + ruff clean.
  Regression fix: the validator was first put on SiteBase, but SiteResponse/SiteWithStats also
  extend SiteBase, so reading back an existing 0,0 site (projects created before the validation)
  failed with a 500 on GET /api/sites. Moved the check onto the INPUT schemas only (SiteCreate +
  SiteUpdate); SiteResponse serializes 0,0 fine. 47 site tests pass.

  Follow-up 2: the form defaulted lat/long to a literal 0, which invited saving null island.
  Changed defaults to empty (undefined) so the inputs show their placeholders, and added friendly
  zod messages so an empty field on save reads "Enter a latitude/longitude between ..." instead of
  the raw NaN error. Validation still runs only on save (RHF default onSubmit). Edit mode and the
  map-click prefill still set real coords. Zod v4 uses `{ error: ... }`, not `invalid_type_error`.

### B9 - Blue cast on filmstrip thumbnails
Some filmstrip thumbnails in Verify, Events tab have a blue cast. Clicking a detection clears the cast.
- status: done (screenshot confirmed)
- notes: The "blue" is the `bg-muted` fallback in FrameThumbnail.tsx:54. In this theme `--muted`
  is HSL hue 210 (a blue-grey), and it shows through any tile whose thumbnail image hasn't
  painted (slow load, or the onError handler setting display:none). Clicking a detection
  re-renders the tile, which resets the img's inline style and reloads it, repainting over the
  blue, hence "clears on click". Fix: container background `bg-muted` -> `bg-neutral-200
  dark:bg-neutral-800` (true grey, no blue hue). Covers both the event-card collage and the
  Counts-modal filmstrip (same component). Typecheck clean.
  Parked secondary issue (not chased, KISS): the onError handler hides the img permanently until
  a re-render, so a transiently-failed thumbnail stays a blank placeholder until interaction. The
  root cause (why a thumbnail occasionally fails to paint) is separate and fuzzier; left alone.

### B10 - Empty-classification filtering inconsistent
In Verify Captures, selecting "All" hides Empty-classified images (only "Show only empty" reveals them), yet events still render showing only-empty images.
- status: already solved (predates current code)
- notes: The core bug (Empty filter = "All" still hid blanks) is fixed in the current
  `get_files_for_verify` (file.py:438-457), with a comment documenting exactly this: blank files
  have no detections so they fail the implicit confidence-floor EXISTS gate; when no user conf/
  label gate is set and the user has not chosen "hide", blanks are explicitly let through
  (`or_(exists(conf_subq), File.observation_type == "blank")`), "otherwise Empty: All would
  silently drop every blank tile". So now: all -> shows empties, show_only -> only blanks, hide
  (default) -> no blanks. The events path (event.py empty show_only/hide) is consistent: a
  fully-empty event is hidden by default and shown with "show only empty"; an event with some
  detections shows and includes its in-between empty frames by design (time cluster). The beta-4
  "Captures" tab itself is gone (verify restructured to Observations + Labels). No change needed.

### B11 - Label tree counts ignore site filter
Detection counts in the Labels tree are not filtered by the selected site.
- status: done
- notes: Real bug: `/label-tree` only took project_id + count_by, so tree counts were always
  project-wide even with a site (or date) selected in the Verify / Map filter bar. Scope decided
  as site + date only (KISS/DRY/YAGNI): those are the "what slice exists" filters a label
  inventory should reflect; verified/confidence/flagged/favorited/empty are review-state filters
  and would mean dragging the whole event-filter pipeline (which filters by label) into the tree
  builder, plus jumpy counts. Fix: `build_label_filter_tree` gained site_ids/date_from/date_to,
  applied via one `_apply_scope` helper to all 3 count queries (Deployment.site_id +
  File.captured_at_local, date semantics matching crud/statistics.py). Endpoint exposes the
  params; frontend `getLabelTree(projectId, countBy, scope)` passes them from both the Verify and
  Map filter bars, with site+date in the query key so the tree recounts on change. Typecheck,
  ruff, import sanity, and 13 label-tree tests all pass.

### B12 - Remapped drive letter shows no images
If data moves to a drive with a different letter, no images show because absolute paths are stored in `files`. Wants a check that the mapped drive is present, with a warning.
- status: already solved (existing relink feature)
- notes: Already implemented end to end, and goes beyond "just warn". (1) Detection: a startup
  background task `check_all_deployment_folders` (main.py:431) verifies every deployment's folder
  (path existence + sample-size check, deployment.py:717); a remapped/unmounted drive flips
  folder_status to "needs_relink". (2) Warning: DeploymentHealthToast fires a startup sonner
  warning ("N deployment folders couldn't be found ... moved, renamed, or unmounted") with a View
  action, plus RelinkGroupBanner on the deployments page. (3) Fix: suggest-relink-target walks up
  the broken path to find the moved location (resolves a new drive letter), and bulk-relink
  rewrites the absolute file_path values in the files table. So the mapped-drive case is detected,
  warned, and reconnectable. No change needed.

## UX / wording

### U1 - "New project" in the app menu
Add a "New..." menu option to create a project. Currently only reachable via the "Back to project" link at the bottom.
- status: done
- notes: Simon's feedback predates the HomePage, which already largely solved discoverability (two
  launcher cards + a File > Home item). Peter chose to also add the File > New convention.
  Added two items at the top of the File menu (electron main.ts): "New project…" (new-project) and
  "Analyse a folder…" (new-folder-run), then a separator before Home. Both gated in
  SETUP_GATED_MENU_IDS so they're disabled until first-run setup finishes. Renderer
  (MenuCommands.tsx): new-folder-run -> navigate /folder-runs/new; new-project -> navigate
  /projects?new=1. ProjectsPage reads ?new=1 on arrival, opens the CreateProjectDialog, and clears
  the param (replace) so a refresh doesn't reopen it. Frontend + electron typecheck clean.

### U2 - Smooth the detection ETA
Image-detection remaining time fluctuates a lot (6h to 8h within seconds on 190k files). Wants smoothing.
- status: done
- notes: The ETA was tqdm's own `<MM:SS` field, a short rolling-window estimate that swings on
  heterogeneous batches; the frontend buckets it but 6h vs 8h are different buckets so it still
  flips. Fix (overall-average, chosen over EMA for KISS): in `_parse_tqdm_metrics`
  (megadetector.py) replace tqdm's remaining with `elapsed_s * (total - current) / current`,
  computed from the current/total/elapsed already parsed off the same line. Stateless, output is
  the same tqdm time-string so the frontend is untouched. Two module helpers added
  (`_tqdm_time_to_seconds`, `_seconds_to_tqdm_time`). Verified: two consecutive lines where tqdm
  swung 6:04:00 -> 8:00:00 now give 4:20:20 -> 4:20:11 (stable). Trade-off accepted: an average
  since start, so the first minute or two can read a bit high until it settles. Ruff clean, 23
  detector/progress tests pass.

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
- status: done (clarified, not renamed)
- notes: Both labels are kept on purpose. "Observation" = sum of MaxN per event (an abundance
  count, not raw events and not images), the same metric and word the dashboard uses as a
  headline stat ("Observations", "Observations per 100 trap nights", "Top 10 by total
  observations"). "Trap night" is the field-standard term for one camera active 24h (confirmed
  via research; "per 100 trap-nights" is the standard rate phrasing). Renaming to "events" would
  be inaccurate; renaming to "trap-day" would be non-standard and would also diverge from the
  dashboard vocabulary. Simon's confusion was that neither term was defined where he was looking
  (the legend). Fix: added a plain-language definition to the map's "About this view" section
  (`MapPage.tsx` PlotExplainer `what`), mirroring the dashboard's wording: "An observation counts
  individuals: each event's confirmed count, or the AI's count where not yet confirmed, which is
  the most individuals visible in a single frame, so the same animals aren't counted twice. A
  trap night is one day a camera was active." Legend unchanged (matches the dashboard). Removed
  the now-redundant footer line "Rate is observations (MaxN per event) per 100 trap nights" since
  the About section covers it. Typecheck clean.

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
  Screenshot review (image18): the exact message Simon flagged was the beta-4 Observations-tab
  empty state "All 144 detections in this view are verified. Switch to "All" to see them." That
  message no longer exists: Verify was restructured since beta 4 (no more "Captures" tab; no
  per-detection all-verified message). The current Observations/Events empty state is the generic
  "No events match your filters" + a "Clear all filters" button (VerifyView.tsx:320-345), which is
  functional. The surviving analogous message (LabelsTab) is the one improved above. No further
  code needed unless we want the generic empty state to special-case the all-verified scenario.

### U7 - Squamata appears twice in labels tree
Squamata shows twice in the labels tree. Simon notes it is a SpeciesNet issue; may still want display dedup.
- status: wontfix (working as designed; confirmed by Simon's DB screenshot)
- notes: Confirmed with the label_taxonomy screenshot (image17). SpeciesNet (SPECIESNET-v4-0-2-A)
  ships TWO distinct classes that both sit at order = Squamata with no family:
    - name = "squamata"            (taxon_class reptilia, taxon_order squamata) -> 1 detection
    - name = "lizards and snakes"  (taxon_class reptilia, taxon_order squamata) -> 8 detections
  These are two different model outputs, not a duplicate row: "squamata" is the bare order-level
  prediction SpeciesNet returns when it cannot go finer, and "lizards and snakes" is a separate
  common-name class that also resolves to the order Squamata. (Same shape as "bird" and "raptor"
  both being class Aves.) The unique constraint is (model_id, name), so they are legitimately two
  separate rows with separate detection counts.

  In the tree (image16) they both hang under one "Squamata" order node and render as:
    - "Squamata (unspecified)"        -> the "squamata" row (1 detection)
    - "Squamata (lizards and snakes)" -> the "lizards and snakes" row (8 detections)
  Why those names: for a non-species leaf the display is the rank-derived "Squamata"
  (label_tree.py), and the annotation in brackets is the underlying model label. "lizards and
  snakes" differs from the rank name so it shows as the annotation; "squamata" equals its rank
  name so the annotation falls back to the literal "unspecified" (TreeSelector.tsx:475-478).

  So it is not a bug and not a dedup miss: two genuinely different SpeciesNet classes, kept
  separate so counts stay correct, already disambiguated by the bracketed annotation. Merging
  them would hide a real distinction and double-count. Left as is.

  Message for Simon: those are two separate SpeciesNet classes that both resolve to the order
  Squamata, a bare "squamata" prediction (1 detection) and a "lizards and snakes" class (8
  detections), much like "bird" and "raptor" both being class Aves. AddaxAI keeps them apart so
  the counts stay correct and shows the model label in brackets to tell them apart; the
  "(unspecified)" one is the generic order-level prediction.

## Feature requests

### F1 - Drill down from dashboard to verify
Click or right-click a species in the dashboard summary to open Verify pre-filtered to that species. Quick way to fix frequent misidentifications.
- status: done
- notes: Target is the Labels page (the per-detection label-cleanup workspace; "Verify" was
  restructured into Labels + Counts/Observations). Clicking a bar in the dashboard "Top taxa" chart
  opens /projects/:id/labels?lbl_labels=<ids>, which LabelsTab hydrates into its label filter on
  arrival (lblFiltersFromSearchParams, no extra click). Mapping: the SpeciesCount rows now carry
  `label_taxonomy_ids` (backend group_concat(distinct EventObservation.label_taxonomy_id) per bar;
  group_concat skips NULLs so non-taxonomy bars come back empty). This handles every rank: a
  species bar -> its ids, a family/"Higher-level taxa" bar -> all ids it covers. Frontend: onClick
  on the chart navigates with those ids; onHover shows a pointer only on clickable bars; bars with
  no taxonomy ids are no-ops. Gated to project mode (folder-run is a linear wizard with its own
  labels step). Files: schemas/statistics.py + crud/statistics.py (group_concat + field),
  api/statistics.ts (type), DashboardView.tsx (drillToLabels + onClick/onHover). Backend ruff +
  101 statistics tests pass; frontend typecheck clean.

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
- status: already solved
- notes: Fully implemented. `lib/species-name-mode.ts` holds a global per-user preference (common
  default / scientific) in localStorage; the native menu has View > Species names > Common /
  Scientific (species-common / species-scientific commands). `resolveSpeciesName` is consumed across
  ~25 files including verify (LabelsTab, VerifyFilterBar, VerifyView, CropGrid, EventCountPanel) and
  dashboard (DashboardView + charts), so the toggle flips names app-wide (setSpeciesNameMode reloads
  to guarantee consistency). Simon's "if not available, show something" is handled by the fallback
  chain: common -> scientific -> label -> category (and the reverse in scientific mode), so a missing
  common name shows the scientific/label name, never blank. Apostrophe concern is a non-issue: DB
  access is parameterized SQLAlchemy and names render as React text. Common names formatted via
  `format_common_name` (taxonomy_db.py). No change needed.

### F5 - Copy sites/deployments into a new project
Copy sites and deployments from an existing project so a new project (e.g. to compare models) does not need full re-entry.
- status: done (built as "Duplicate project")
- notes: Reframed (Peter): not "copy data into parallel projects" but a general "Duplicate project"
  in the project card kebab. Opens a Duplicate modal that mirrors the create fields prefilled from
  the source (name "<name> (copy)", description "Duplicate of <name>", classification model, label
  selection, empty image) plus 3 checkboxes: Settings, Sites, Deployments. Deployments re-queues the
  source's deployments' folders for reprocessing (results can't move across projects; honest caption
  says so) and auto-ticks Sites so the queued folders keep their site assignment (site IDs remapped).
  Settings off = fresh defaults. Backend: ProjectDuplicate schema, crud.duplicate_project (copies
  settings cols when flagged else model defaults; copies sites with old->new id map; re-queues
  deployments as pending DeploymentQueue entries), POST /api/projects/{id}/duplicate (404 missing
  source, 409 dup name). Frontend: projectsApi.duplicate + ProjectDuplicatePayload,
  DuplicateProjectDialog, kebab item + dialog in ProjectsPage; on success navigates to the new
  project's dashboard. Name uses "<name> (copy)" default + inline 409 handling (didn't wire the
  name-suggestion endpoint for guaranteed-unique auto-gen; could later). Tests: 4 new duplicate
  tests + 35 project tests pass; frontend typecheck clean.
  Fix after first run: ModelSelect hardcodes <FormControl>, which needs a react-hook-form context;
  the first DuplicateProjectDialog used plain useState and crashed (useFormContext null). Rewrote it
  with react-hook-form (Form provider + FormField around ModelSelect), matching every other call
  site. Checkboxes/image/label-selection stay outside RHF like CreateProjectDialog.

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

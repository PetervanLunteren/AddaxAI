## Priority 1
- [ ] TEST replace-representatives-with-maxn-frames -> Now that we have implemented the MaxN mechanism, should we rethink our "representative" frame selection in the events verification? If we have two species detected in a single event, we have two MaxNs (one for each species). It would make sense to verify both of them, as that influences the counts. The rest do not influence the counts since they are not MaxN anyway. Agree? Maybe we can rename the representative frame to "MaxN", which is understandable by the users as it is a well known concept in ecology. What do you think? So the seleciton is not based on the confidence, size, sharpness as it is done now, but only on the maxNs of the labels above the project.detection.confidence. That means an event can have several "MaxN" frames. How would this affect the current project? What needs to be done? What am I forgetting? Are there any complications? What do you think? Do a full audit. 


- [ ] (SEE DANS EMAIL) bug: when doing an analysis, the 'image classification' pbar goes from 0 to 100 without showing any stats like the other pbars. Then the 'finalizing...' part takes long. I get the sense that eveything happens in the 'finalizing...' phase. Could that be true? Investigate. 

- [ ] INVESTIGATE REFACTOR TO RUN SPECIESNET AS A NORMAL CLASSIFACTION MODEL - at the moment SpeciesNet uses its own inference code, whilest the other classification models all share their inference code. That seems like extra complexity. What if we just run SpeciesNet as a 'normal' clasisfication model like all the others? That save a lot of conplexity and if/else statements. Do a full audit on how this would affect the current code base, what needs to be changed and what features would not work then. What are the things that are hard, what pros and cons, etc. I want a full report and everything thought of. I know the current way of running SpeciesNet is by using its internal country + state geofencing, but can we mimick that ourselves by just reading the SpeciesNet sepecific country data and then allowing users to select / deselect labels just like any other classifciation model does? I know this is a great refactor, but I believe it should be thoroughly investigated, since it will make our lives a lot easier in the end. 







- [ ] ENSURE EVERY LABEL HAS FULL TAXONOMY - Currently not all detection labels have corresponding entries in the `label_taxonomy` table, which means they lack taxonomy breadcrumbs, display names, and will have incomplete export data. There are several gaps:

  **Gap 1: Exclusion rollup labels.** When `filter_and_rollup_classifications()` in `label_exclusion.py` creates ancestor labels (e.g., "suidae", "artiodactyla"), these are added to `classification_categories` in-memory but NOT persisted to `label_taxonomy`. The existing postprocessing rollup (`apply_taxonomic_rollup_to_results` in `taxonomic_rollup.py`) does persist via `add_rollup_taxonomy_entry()`, but the exclusion rollup path skips this. Fix: after DB loading in `json_pipeline.py`, iterate any new rollup labels created by exclusion and call `add_rollup_taxonomy_entry()` for each.

  **Gap 2: Person and vehicle categories.** These are always-available detection categories but never have `label_taxonomy` entries. They show up as "No taxonomy" in the label tree. Fix: seed "person" and "vehicle" as built-in taxonomy entries (with empty taxonomy fields, level="category" or similar) during DB init.

  **Gap 3: Labels without taxonomy.csv match.** If a model class name doesn't match any row in taxonomy.csv (e.g., custom labels added by the user before taxonomy was populated), it won't have taxonomy. Fix: when a custom label is created via GBIF, `add_rollup_taxonomy_entry()` or similar should be called to create the taxonomy row immediately.

  **Gap 4: display_name not set for all detections.** The `display_name` column on Detection is only populated during DB loading (when taxonomy_lookup is available) and during postprocessing label updates. Detections from older analyses or manual relabeling without taxonomy may have NULL display_name. Fix: add a one-time migration or backfill script that computes `display_name` for all detections with labels using `format_latin_display_name()`.

  **Why this matters:** Export formats (Camtrap DP, CSV, Darwin Core) require full taxonomy for every observation. The label filter tree needs taxonomy to display labels under the correct branch. The dashboard needs taxonomy for rank-based grouping. Without complete taxonomy, features degrade silently.

  **Approach:** The cleanest solution is to make `label_taxonomy` population a mandatory step that runs after every label assignment (ML classification, rollup, manual relabel, custom label creation). Every code path that sets `Detection.label` should also ensure the label has a corresponding `label_taxonomy` entry. Key files: `json_pipeline.py` (DB loading), `postprocessing.py` (smoothing/rollup), `label_exclusion.py` (exclusion rollup), `detection.py` CRUD (manual relabel), `projects.py` router (custom label creation).


- [ ] CONSOLIDATE TAXONOMY DISPLAY INTO SINGLE SOURCE OF TRUTH - We've standardized on Latin taxonomy names as the primary display format (G. camelopardalis, Felidae, etc.) across detection chips, overlays, dashboard, and taxonomy trees. But the display name is currently computed in at least 6 different places using different methods, which will cause inconsistencies as the codebase grows.

  **Current sources of truth / computation methods:**
  1. `Detection.display_name` column: stored at DB load time via `format_latin_display_name()`. This is the primary source for detection-level display. Set in `json_pipeline.py` during DB loading and `postprocessing.py` during smoothing/rollup updates.
  2. `format_latin_display_name(label, taxonomy_lookup)` in `taxonomic_rollup.py`: backend helper using a dict lookup. Used during DB loading.
  3. `format_display_name_from_taxonomy_row(label, genus, species, ...)` in `taxonomic_rollup.py`: backend helper using individual fields. Used during manual relabeling in `detection.py` CRUD and `detections.py` router.
  4. `formatLatinName(rawLabel, taxonomyEntry)` in `useLabelOptions.ts`: frontend helper mirroring the backend logic. Used by the label picker to show Latin names for label options.
  5. SQL CASE expressions in `statistics.py`: inline SQL that concatenates `upper(substr(genus,1,1)) || '. ' || species` for dashboard charts. Duplicates the formatting logic in SQL.
  6. `label_tree.py` line 215 and `taxonomy_parser.py` line 184: tree builders format species as `G. species` with their own inline logic.
  7. `EventSummary.display_labels` dict: computed in `crud/event.py` from `Detection.display_name` at query time.

  **The problem:** If we change the display format (e.g., switch from "G. camelopardalis" to "Giraffa camelopardalis"), we'd need to update all 6+ locations. The frontend and backend each have their own formatter that could drift. The SQL expressions are the hardest to maintain.

  **Proposed solution:**
  - Store `display_name` on `label_taxonomy` table (one per label, not per detection). Compute it once when taxonomy is populated.
  - `Detection.display_name` becomes a denormalized copy set from `label_taxonomy.display_name` at assignment time.
  - Frontend gets display names from the API (already does via `Detection.display_name` and `EventSummary.display_labels`). Remove `formatLatinName()` from frontend.
  - Dashboard statistics queries use `label_taxonomy.display_name` via join instead of inline SQL formatting.
  - Label tree builders read `display_name` from taxonomy rows instead of formatting inline.
  - One formatter function on the backend (`format_latin_display_name`), called only during taxonomy population. Everything else reads the stored result.

  **Files involved:** `taxonomic_rollup.py` (keep one formatter), `taxonomy_db.py` (populate display_name on taxonomy rows), `label_taxonomy.py` model (add display_name column), `json_pipeline.py` (read from taxonomy instead of computing), `statistics.py` (join instead of SQL CASE), `label_tree.py` + `taxonomy_parser.py` (read instead of format), `useLabelOptions.ts` (remove formatter, read from API), `detection.py` CRUD (read from taxonomy row).

  **Cost estimate:** Medium. The column addition and population are straightforward. The main work is updating all the read sites to use the stored value and removing the duplicate formatters. Should be done alongside the "ensure every label has full taxonomy" TODO above since both involve `label_taxonomy` changes.


## Priority 2
- [ ] dashboard verification vard, explenation text "Event representatives are one file per event, used for quick review." explain a bit more how that representative is chosen. See event verification guide for more info. 

- [ ] If we do taxonomic rollup, we might get to taxa without common names or model-class-names like "cow" and "equid". What happens then? What do we show the user in the chips and in the UI? Investigate. I want to know the current way of dealing with that and all its fallbacks. 


## Priority 3
- [ ] REMOVE LEGACY ML PIPELINE CODE - Several legacy abstractions have been superseded by the direct phase handling in `_process_batch_job()` in `backend/app/workers/detection_worker.py`. The batch path (which the UI always uses) calls standalone functions directly, while these legacy classes duplicate the same work. Removal involves three pieces: (1) `JSONBasedMLPipeline` class in `backend/app/ml/json_pipeline.py` (lines ~42-710): has its own `_load_to_database()`, `_run_detection()`, `_run_classification()`, and `process_deployment()`. Only reachable through the non-batch branch in `process_deployment_analysis()` (detection_worker.py ~line 766), which the frontend never triggers. Delete the class, the non-batch branch, and its import. (2) `MLPipeline` class in `backend/app/ml/pipeline.py`: completely dead code, never imported or called by anything. It uses the `detect()` method on MegaDetector which is also only called from this dead class. Delete the entire file. (3) The `detect()` method in `backend/app/ml/inference/megadetector.py` (lines ~95-210): only called by the dead `MLPipeline` class. The active code path uses `detect_to_json()` instead. Delete the method. Together this is roughly 500 lines of duplicated or dead code.


## Features
- [ ] TIME OFFSET - add a feature that allows datetime offset. This should happen at the "new deployment" options. Perhaps something that says "your data spans X days/weeks, etc. " Click here to see the burned in pixel dates (show a few images / frames) and show the extracted datetime next to it. Then users can add an offset to all data in the deployment. Add fast options to switch from AM to PM etc. +12:00 and -12:00. 
- [ ] METADATA MANAGEMENT - add pages for deployments and sites, where they are all visible in a table format with filter options that make sense for the data. Each row is a unit (deployment or site) and the user can filter, sort, view, and then have actions as a three dot. The actions three dot will be "edit" for now only, where the user can edit the name of the unit, the time, etc. These are not defined yet, but need to be defined in this plan. The idea is that it offers a page where users have more room to customise their metadata, and edit flexibly. We will add actions to the three dot later on. 
- [ ] TIMELAPSE STANDALONE APP
- [ ] DEPTH ESTIMATION
- [ ] PROCESS BATCH RESULTS - https://github.com/agentmorris/MegaDetector/blob/main/megadetector/postprocessing/postprocess_batch_results.py
- [ ] BATCH PROCESSING OPTION - a completely separate option to do batch processing, where it just runs det+cls on all data recursively, and is agnostic of its contents (doestn need to know if it is a deployment, site, full project, etc.) It just runs everything at once. Users should be able to set settings before running the analysis. Then, after, it should give the user a few options, like export to CSV, XLSX, maps & graphs, separate into subfolders, etc. The bulk / management choice should be the first page users see when opening AddaxAI. 
- [ ] IN DEPTH PLOTS, have a header in the menu "plots", and add page wide full plots that are interactive and with a bunch of filters and settings above. The dashboard is meant as a quick glance of the project, and these are more in depth to find out patterns etc. We will be adding in depth plots as we progress with the project, but the first one is "Comparison of the activity time" (improve wording, make it short and memorable), where the user can select up to 5 labels and compare the activity. One option should be to add the suntimes to the graph (sun hours, sunset, etc.), another option would be to have the actual time on the x axis, or the UTC times (based on the suntimes), do webqueries on how other platforms do this, and what the standard is, and what is usually reported in terms of metrics. Research scientific papers etc. 
- [ ] IN DEPTH PLOT - add new in depth plot: Gantt-style timeline — one horizontal bar per deployment (or per site), showing the active period. Immediately shows gaps, overlaps, and total survey effort. Group by site with one bar per deployment within each site row. 
- [ ] MAP - check AddaxAI Connect and copy the map from there. 
- [ ] EXPORT OPTIONS - check AddaxAI Connect and copy from there. 
- [ ]





## Installer
- [ ] merge all alembic/versions/ into one. We do not have any users yet, so we can make it just the start DB. 
- [ ] make sure on app istall it installs the default models and their environments (MDv5A and DINOv2-B). 

## IMPROVE LABEL STUFF
We've standardized on Latin taxonomy names as the primary display format (G.
camelopardalis, Felidae, etc.) across detection chips, overlays, dashboard, and
taxonomy trees. I get the feeling that we currently have differnet approaches on how to lookup taxonomy, and caluclate the display names etc. Would it make sense to just move everything to a standard with a new endpoint that just returns all the info you need from the backend ot SQL table so that there is one source of truth. I get the feeling that now we have many sources of truth.... Investigate where all the truths and the differnt methods are. And what it would cost to make a standard and merge into one source of truth. 

### COUNTRY DROPDOWN
Add a simple country dropdown if goefencing file exists. Perhaps we can add a tab like structure like "simple" / "full control". Or quick / full control or what would you propose? Most people will just want to say: I'm in the Netherlands. Or is this just adding UI cpomplexity? They can now just clikc the button and select Netherlands, then "OK". Otherwise we add a tab control, which adds complexity. But showing a full list with all 2000 species is also ceomplex! What do you think.

### IMPROVE UI ON CARDS
Improve the verification checkmarks in the grid view of events verification. Should we show pbars for the MaxN files and the all files? SOmething like that? Also make the Verification status filter explicit. Add options for all scenarios, one or more MaxNs verified, etc, etc. 


### CHECK JSON PERSISTENCE
Wen doing a different session, this came up.
"When an excluded species rolls up to an ancestor (e.g., lion -> felidae), this creates a NEW classification_categories entry in the JSON. Should we also persist    
this to the JSON file on disk (like the existing postprocessing rollup does)" I'm referring to this part: "like the existing postprocessing rollup does". What do you mean? Does it alter the JSON on disk? It shouldn't. The JSON is created during analysis and should never change afterwards. Its the ground truth. 


### Have the "+ Add label" option always at the end of the labelpicker list, not condintionally. Now I search for somrthing like "Bee", and since there are still classes showing up, there is no "+ Add label" option, while i want to add a new label for "Bee". I guess this is simple. Just make it unconditional. 
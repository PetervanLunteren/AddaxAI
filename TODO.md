## Priority 1
- [ ] TEST replace-representatives-with-maxn-frames -> Now that we have implemented the MaxN mechanism, should we rethink our "representative" frame selection in the events verification? If we have two species detected in a single event, we have two MaxNs (one for each species). It would make sense to verify both of them, as that influences the counts. The rest do not influence the counts since they are not MaxN anyway. Agree? Maybe we can rename the representative frame to "MaxN", which is understandable by the users as it is a well known concept in ecology. What do you think? So the seleciton is not based on the confidence, size, sharpness as it is done now, but only on the maxNs of the labels above the project.detection.confidence. That means an event can have several "MaxN" frames. How would this affect the current project? What needs to be done? What am I forgetting? Are there any complications? What do you think? Do a full audit. 


- [ ] (SEE DANS EMAIL) bug: when doing an analysis, the 'image classification' pbar goes from 0 to 100 without showing any stats like the other pbars. Then the 'finalizing...' part takes long. I get the sense that eveything happens in the 'finalizing...' phase. Could that be true? Investigate. 

- [ ] INVESTIGATE REFACTOR TO RUN SPECIESNET AS A NORMAL CLASSIFACTION MODEL - at the moment SpeciesNet uses its own inference code, whilest the other classification models all share their inference code. That seems like extra complexity. What if we just run SpeciesNet as a 'normal' clasisfication model like all the others? That save a lot of conplexity and if/else statements. Do a full audit on how this would affect the current code base, what needs to be changed and what features would not work then. What are the things that are hard, what pros and cons, etc. I want a full report and everything thought of. I know the current way of running SpeciesNet is by using its internal country + state geofencing, but can we mimick that ourselves by just reading the SpeciesNet sepecific country data and then allowing users to select / deselect labels just like any other classifciation model does? I know this is a great refactor, but I believe it should be thoroughly investigated, since it will make our lives a lot easier in the end. 







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















-------------


Here's the full comparison:

  How SpeciesNet does geofencing vs how AddaxAI does it

  SpeciesNet (internal)

  Geofencing is applied after a complex ensemble step that combines MegaDetector detections with SpeciesNet classifications. The ensemble has ~8 threshold-based decision branches:

  1. Human/vehicle detections bypass geofencing entirely (MD confidence thresholds at 0.2 and 0.7)
  2. Blank predictions bypass geofencing (classifier > 0.99 or classifier > 0.5 + MD conf < 0.2)
  3. Only animal predictions are geofenced, and only when confidence is high enough (> 0.65 or > 0.8)
  4. When geofenced: the species is NOT removed. Instead, it rolls up to the nearest allowed ancestor in the taxonomy tree (species → genus → family → order → class → kingdom), accumulating
  scores from all child labels at that level. E.g., lion (0.85) in USA → rolled up to felidae family (0.85 + other felidae scores)
  5. If all rollup levels are also geofenced: returns "unknown"

  AddaxAI (our approach)

  Geofencing is applied before DB loading via excluded_classes:

  1. Excluded species are completely removed from the classification list
  2. Remaining confidences are renormalized to sum to 1.0
  3. The new top-1 becomes the label
  4. No taxonomic rollup to ancestors: if a lion is excluded, its confidence doesn't flow up to felidae. Whatever was second-best (could be an unrelated species) becomes the prediction
  5. No confidence thresholds: exclusion applies regardless of how confident the model was
  6. Blank is also excluded (as a non-label class), which can promote garbage predictions

  Key differences

  ┌───────────────────┬───────────────────────────────────────────────────────────────┬──────────────────────────────────────────────────────────┐
  │      Aspect       │                          SpeciesNet                           │                         AddaxAI                          │
  ├───────────────────┼───────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────┤
  │ Geofenced species │ Rolled up to nearest allowed ancestor with accumulated scores │ Completely removed, second-best promoted                 │
  ├───────────────────┼───────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────┤
  │ Score handling    │ Scores from geofenced species flow to parent taxon            │ Scores redistributed evenly across all remaining classes │
  ├───────────────────┼───────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────┤
  │ Blank handling    │ Kept as valid prediction (if confident)                       │ Always removed as non-label, promoting next class        │
  ├───────────────────┼───────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────┤
  │ Human/vehicle     │ Special ensemble logic with MD confidence thresholds          │ No special handling, just classifier output              │
  ├───────────────────┼───────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────┤
  │ Confidence gates  │ Only geofences if classifier > 0.65                           │ Always applies regardless of confidence                  │
  ├───────────────────┼───────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────┤
  │ Rollup failure    │ Returns "unknown"                                             │ Returns whatever random class is highest after filtering │
  └───────────────────┴───────────────────────────────────────────────────────────────┴──────────────────────────────────────────────────────────┘

  Impact

  The biggest practical difference is the blank + renormalization problem you saw earlier: a detection that's 99% blank and 0.1% human gets blank removed, human renormalized to ~50%, and loaded
  as "human". SpeciesNet would have kept it as "blank" because blank at 99% passes the ensemble's blank threshold.

  The second difference is taxonomic rollup: SpeciesNet would show "felidae family" for a geofenced lion prediction, while AddaxAI would show whatever unrelated species had the second-highest
  score.








please show me some examples of your intended behaviour. 

What i want in siple terms is that it checks the excluded classes first: If cow is not supposed to be there, normalise, etc. This works the same for any model class, regardless of it being in the NON_ANIMAL_CLASSES. Some models might be shipped with classes like "false detection", "vide", or like speciesnet "blank". If the user specified that he does not want any predictions for blank (so specified "blank" as not present), it should get ri of those and renormalise. Then, when we have our filtered top-1 prediction, we move to the step that excludes certaoin detections from entering the DB. 

The JSON file remains unchanged, but the way the filtering works changes the DB. makes sense?



the raw labels (top-1 only), and if it is blank, bait, vide, etc (the NON_ANIMAL_CLASSES), skip that detection entirely for all other processes. It is left untouched, but excluded from further filtering or ingestion into the DB. If a detection is not excluded, then we continue with the excluded_classes. If cow is not supposed ot be there, normalise, etc. This works the same for   










  ┌─────────────────┬────────────────────────────────────────────┬──────────────────────────────────────┬────────────────────────────────────────────────────────────────────────────────────┐
  │     Aspect      │                 SpeciesNet                 │         AddaxAI (after fix)          │                                  Still different?                                  │
  ├─────────────────┼────────────────────────────────────────────┼──────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────┤
  │ Blank handling  │ Kept as valid prediction if confident      │ Top-1 blank → skip detection (not    │ Different but both correct. SpeciesNet keeps blank as a label, AddaxAI skips the   │
  │                 │                                            │ loaded)                              │ bbox entirely. Both prevent garbage labels.                                        │
  ├─────────────────┼────────────────────────────────────────────┼──────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────┤
  │ Geofenced       │ Rolled up to nearest allowed ancestor with │ Removed, second-best promoted via    │ Yes, still different                                                               │
  │ species         │  accumulated scores                        │ renormalization                      │                                                                                    │
  ├─────────────────┼────────────────────────────────────────────┼──────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────┤
  │ Score handling  │ Scores flow to parent taxon                │ Redistributed across all remaining   │ Yes, still different                                                               │
  │                 │                                            │ classes                              │                                                                                    │
  ├─────────────────┼────────────────────────────────────────────┼──────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────┤
  │ Human/vehicle   │ Special ensemble with MD confidence        │ No special handling                  │ Yes, still different                                                               │
  │                 │ thresholds                                 │                                      │                                                                                    │
  ├─────────────────┼────────────────────────────────────────────┼──────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────┤
  │ Confidence      │ Only geofences if classifier > 0.65        │ Always applies regardless of         │ Yes, still different                                                               │
  │ gates           │                                            │ confidence                           │                                                                                    │
  ├─────────────────┼────────────────────────────────────────────┼──────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────┤
  │ Rollup failure  │ Returns "unknown"                          │ Returns whatever is highest after    │ Yes, still different                                                               │
  │                 │                                            │ filtering                            │                                                                                    │
  └─────────────────┴────────────────────────────────────────────┴──────────────────────────────────────┴────────────────────────────────────────────────────────────────────────────────────┘



  1. Exclusion rollup (new): "this species can't be here, redirect its score to the nearest valid ancestor" (e.g., lion excluded in USA → felidae family)


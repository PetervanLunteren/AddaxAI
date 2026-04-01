## TEMPORARY
- [ ] RE-ENABLE NON-LABEL SKIP - The `should_skip_detection()` calls in `backend/app/ml/json_pipeline.py` (lines ~587 and ~976) are temporarily commented out for SpeciesNet comparison testing. Uncomment both blocks after comparison is complete. Search for "TEMPORARY: disabled non-label skip" to find them.

## Priority 1
- [ ] Since we now do the exlusion of animals not by renormalizing the the predicitons based on excluded spces like we used to do (lion 70%, bobcat 20%, deer 10% -> lion excluded -> bobcat 667% deer 33%, lion 0%), but by walking up the taxonomy ladder until we find a common anchestor that is included. We now do not need all predictions in the JSON file for all detections anymore. We could do with keeping only the top-1. Agree? If so, would it make sense to trim the JSONs at the end of the analysis step to leave only top-1 (or perhaps top-5?) and then also remove all the labels in the category dicts that are never called? That saves a LOT of disk space for things we would never use. Agree? Does that make sense?


- [ ] since we only need the top-1 predictions, we can regenerate the DB after a species slecetion change, right? 

- [ ] we should have the exact saem JSON format as SPECIESNET

# REDUCE JSON SIZE
Can we redice the JSON size so that it does not store all prediction confidences of all detections? And also not all category mappings at the end? Basically, just copy what SpeciesNet outputs. The JSON needs to be EXACTLY the same. every byte should be the same. 

# ROLLDOWN 
exlusion rollup currently works like this (corect me if i'm wrong!):
raw: wolf 60%, dog 20%, bear 10%, cat 10%.
exluded wolf, included dog, cat, bear: dog 20%, bear 10%, cat 10%, wolf 0%. 
Detection was wolf top-1, which is excluded, so it check the parent taxon to see if that is included. Canidea is included (via dog), so the deteciton gets the prediction (canidae 80%), am I correct? 
What if we also allow rolldown if there is only one child taxon present? The above example would then go to canidae 80% and see that there is only one canidae possible, so it must be that one, so it rollsdown to dog 80%. What do you think? If the raw prediction was wolf 60%, dog 20%, fox 10%, cat 10%, the rolldown wouldn't have worked since then there are two childs of the canidae, and hence the prediciton would remain canidae 80%. Agree? What do yout think of this appraoch? And what would it take to implement it? 




## Priority 2
- [ ] dashboard verification vard, explenation text "Event representatives are one file per event, used for quick review." explain a bit more how that representative is chosen. See event verification guide for more info. 

- [ ] would it make sense to upgrade the app to use DINOv3 instead of DINOv2?


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
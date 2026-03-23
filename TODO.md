## Priority 1
- [ ] TEST replace-representatives-with-maxn-frames -> Now that we have implemented the MaxN mechanism, should we rethink our "representative" frame selection in the events verification? If we have two species detected in a single event, we have two MaxNs (one for each species). It would make sense to verify both of them, as that influences the counts. The rest do not influence the counts since they are not MaxN anyway. Agree? Maybe we can rename the representative frame to "MaxN", which is understandable by the users as it is a well known concept in ecology. What do you think? So the seleciton is not based on the confidence, size, sharpness as it is done now, but only on the maxNs of the labels above the project.detection.confidence. That means an event can have several "MaxN" frames. How would this affect the current project? What needs to be done? What am I forgetting? Are there any complications? What do you think? Do a full audit. 


- [ ] (SEE DANS EMAIL) bug: when doing an analysis, the 'image classification' pbar goes from 0 to 100 without showing any stats like the other pbars. Then the 'finalizing...' part takes long. I get the sense that eveything happens in the 'finalizing...' phase. Could that be true? Investigate. 

## Priority 2
- [ ] dashboard verification vard, explenation text "Event representatives are one file per event, used for quick review." explain a bit more how that representative is chosen. See event verification guide for more info. 

## Priority 3
- [ ] REMOVE LEGACY JSONBasedMLPipeline CLASS - The `JSONBasedMLPipeline` class in `backend/app/ml/json_pipeline.py` (lines ~42-710) is a legacy abstraction that has been superseded by the direct phase handling in `_process_batch_job()` in `backend/app/workers/detection_worker.py`. The batch path (which the UI always uses) calls the standalone `load_json_to_database()` function directly, while the legacy class has its own `_load_to_database()` method, `_run_detection()`, `_run_classification()`, and `process_deployment()`. This creates two parallel code paths for the same work, which means every change (like the non-label skip logic) needs to be applied in two places. The class is only reachable through the non-batch branch in `process_deployment_analysis()` (detection_worker.py ~line 766), which the frontend never triggers (the deployment queue router always sets `is_batch_job: True`). Removal involves: (1) delete the `JSONBasedMLPipeline` class and its methods from `json_pipeline.py`, (2) remove the non-batch branch from `process_deployment_analysis()` in `detection_worker.py` (everything after the `if is_batch` check at ~line 758), (3) remove the `JSONBasedMLPipeline` import from `detection_worker.py`. This is roughly 300 lines of code that duplicates what the batch path already does better.


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
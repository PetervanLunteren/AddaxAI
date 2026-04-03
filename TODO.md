## TEMPORARY
- [ ] RE-ENABLE NON-LABEL SKIP - The `should_skip_detection()` calls in `backend/app/ml/json_pipeline.py` (lines ~587 and ~976) are temporarily commented out for SpeciesNet comparison testing. Uncomment both blocks after comparison is complete. Search for "TEMPORARY: disabled non-label skip" to find them.

## Priority 1
- [ ] since we only need the top-5 predictions in the JSON for reprocessing, we can regenerate the DB after a species slecetion change, right? Is that already implemented? If not, how would it look like? Remeber that verified predictions should NEVER be overwritten by automatic reprocessing methods or other rewrites. 

# TEST 1. Unify exclusion/rollup code paths: First-time DB load uses filter_and_rollup_classifications() (one step), reprocessing uses apply_label_exclusion_to_results() + apply_taxonomic_rollup_to_results() (two steps). User wants these unified.
## It should be exactly the same when running analysis and when doing reprocessing. So test analysis with reprocess change and back to normal, then compare again.

  - Elaborate plan in repo root: /Users/peter/Documents/Repos/AddaxAI-WebUI/UUID_LABEL_MIGRATION_PLAN.md 
  - Session plan document: /Users/peter/.claude/plans/polymorphic-sauteeing-pebble.md       

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
- [ ] 

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
This looks great! A few notes:
- icon for deteciton should be "maximize"
- icon for files should be "file-image"
- icon for Events should be "images"
- Should we do the "Detection trend" and "Species detected" and "Activity pattern" and "Detection categories" as raw counts, or adjusted per camera trap night (i.e., per 100 camera trap nights)? Or should this be configurable in the settings? I would actualy not make it configurable, since it just adds complexity for the user. Just all per camera trap night is the standard. 
- Should we add some kiund of setting popover like the filter popover where we can define the taxonomic rank the dashboard should adhere to (class/order/family/genus/species/raw)? I understand that some will not have all the way to species level, but it should be clear in the UI UX that it groups to that level where possible, or just shows the finest possible. What do you think?
- "Species comparison" shows days in the x axis. It should be times of day. 

Instructions:
* Read all MD file in root to get a understanding of the project. 
* If something is unclear at any point, stop and ask before continuing.
* Prioritize simplicity and clarity over perfection. The code must be clean, easy to read, and understandable for collaborators. Avoid unnecessary complexity.
* I'm not in a rush. Please be precise and do the task thoroughly. 
* Please ask me any question for clarification. I would rather that you ask too many questions than assume certain details. 
* Ask at least 10 clarifying questions before beginning. Based on the conventions set out in CONVENTIONS.md and your knowledge, give your recommended solution to each questions you ask me. 




## Priority 1
- [ ] The tests on github keep on failing. Perhaps just remove the tests for now. We might restore later, but for now just not test on github actions. We'll just test tem locally ourselves. 

## Priority 2
- [ ] Make sure a classification model is not required for a project. User can also just go with a megadetector version, and then do the identificaiton themselves. 

## Priority 3
- [ ] 

## Installer
- [ ] merge all alembic/versions/ into one. We do not have any users yet, so we can make it just the start DB. 
- [ ] make sure on app istall it installs the default models and their environments (MDv5A and DINOv2-B). 

## Features
- [ ] TIME OFFSET - add a feature that allows datetime offset. This should happen at the "new deployment" options. Perhaps something that says "your data spans X days/weeks, etc. " Click here to see the burned in pixel dates (show a few images / frames) and show the extracted datetime next to it. Then users can add an offset to all data in the deployment. Add fast options to switch from AM to PM etc. +12:00 and -12:00. 
- [ ] METADATA MANAGEMENT - add pages for deployments and sites, where they are all visible in a table format with filter options that make sense for the data. Each row is a unit (deployment or site) and the user can filter, sort, view, and then have actions as a three dot. The actions three dot will be "edit" for now only, where the user can edit the name of the unit, the time, etc. These are not defined yet, but need to be defined in this plan. The idea is that it offers a page where users have more room to customise their metadata, and edit flexibly. We will add actions to the three dot later on. 
- [ ] TIMELAPSE STANDALONE APP
- [ ] DEPTH ESTIMATION
- [ ] PROCESS BATCH RESULTS - https://github.com/agentmorris/MegaDetector/blob/main/megadetector/postprocessing/postprocess_batch_results.py
- [ ] BATCH PROCESSING OPTION - a completely separate option to do batch processing, where it just runs det+cls on all data recursively, and is agnostic of its contents (doestn need to know if it is a deployment, site, full project, etc.) It just runs everything at once. Users should be able to set settings before running the analysis. Then, after, it should give the user a few options, like export to CSV, XLSX, maps & graphs, separate into subfolders, etc. The bulk / management choice should be the first page users see when opening AddaxAI. 
- [ ] IN DEPTH PLOTS, have a header in the menu "plots", and add page wide full plots that are interactive and with a bunch of filters and settings above. The dashboard is meant as a quick glance of the project, and these are more in depth to find out patterns etc. We will be adding in depth plots as we progress with the project, but the first one is "Comparison of the activity time" (improve wording, make it short and memorable), where the user can select up to 5 labels and compare the activity. One option should be to add the suntimes to the graph (sun hours, sunset, etc.), another option would be to have the actual time on the x axis, or the UTC times (based on the suntimes), do webqueries on how other platforms do this, and what the standard is, and what is usually reported in terms of metrics. Research scientific papers etc. 
- [ ] MAP - check AddaxAI Connect and copy the map from there. 
- [ ] EXPORT OPTIONS - check AddaxAI Connect and copy from there. 
- [ ]
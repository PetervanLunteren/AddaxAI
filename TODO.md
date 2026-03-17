## Priority 1
- [ ] 

## Priority 2
- [ ] when adding a label, it should also finish what you were trying to do. SO for example if I relabel 10 detections and say "Add label for .... tiger", i add the label, and click OK, then it should relabel them to Tiger. Now i have to relabel them again ans select the just created label. 
- [ ] in the verification dropdown where you see all the labels of the project, it should show the taxonomy in small caption below the common names (class>order> etc). 
- [ ] Add dividers to the pbar modal for analysis between the pbars for more vertical breathing room.
- [ ] We used to have texts like "Starting up" and "Finalising" with small teal spinner for when a process started but was not further than 0%, or if already 100% but not finsihed yet. Can we get these back? They were very helpful, good UX UI. Please check the git history for how that worked, or implement it frpo sratch again, whatever is easiest. 
- [ ] add stats to the project cards in the project grid view. Stats like n_files, n_detecitons, trap nights, etc. 
- [ ] how are trap nights calculated? If I have 10 cameras in paralel from 1 to 10 march, then nothing until we deploy 10 cameras in paralel again from 1 april to 10 april. How many nights do we have? explain the calculation.  
- [ ] In the slideout for adding a custom model, should we make it clear that they can also save without taxonomy? Perhaps by making the button "Save without taxonomy" if not set, and otherwise "Save with taxonomy"? Or somehting like that? What do you propose? Also explain in the caption. 
- [ ] In the dashboard, lets rethink the "Taxonomic rank" filter a bit more. If we choose a rank, we exclude all other classes, right? Is that what people want? Maybe yes, what do you think? But if we choose "raw label", it should also show persons, vehicles, bait, custom labels without taxonomy, etc. Right? Or should we add a checkbox for this behaviour, something like "Include labels without taxonomy"? What do you think?
- [ ] dashboard verification vard, explenation text "Event representatives are one file per event, used for quick review." explain a bit more how that representative is chosen. See event verification guide for more info. 
- [ ] dashboard verification vard, explenation text "Detections are individual animal, person, or vehicle bounding boxes within files." bounding box is jargon. Make it "observations" or somehting like that. Same with "bounding boxes " in the lines after that. 
- [ ] the pbar of "Image classification" and "video classifciation" drops breifly to 0% after finishing to 100%. Try to find out if you can find out why, and otherwise add some debug lines so I can run a test deployment and I can copy paste the console.log back to you. 

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
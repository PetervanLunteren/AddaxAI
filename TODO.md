## Priority 1
- [ ] Is there currently a mechanis that calculates MaxN in events? Or how do we count occurance? If we have an event with 10 images, each with 10 cows, and one with 20, do we count 110 cows? Or do we count maxN=20, so 20 cows? How does this work currently.  And how do the settings "Independence interval" and "Event smoothing" affect that? Investigate. Also, how do other platforms do this? 

How do similar camera trap management systems like Agouti, Camelot, Wildlife insights, TrapTagger, WildTrax, TRAPPER AI, eMammal do it? See below stadards to keep in mind. Please thoroughly investigate. I'm not in a rush. You can webquery to find the most up to date information. 
- GBIF camera trap best practices - https://docs.gbif.org/camera-trap-guide/en/ - Guidance on managing, structuring, validating, and publishing camera trap data at scale.
- Camtrap-DP (TDWG camera trap data package) - https://camtrap-dp.tdwg.org/ - The de facto data standard for camera trap datasets, defining tables, fields, relationships, and controlled vocabularies.
- Darwin Core (TDWG) - https://dwc.tdwg.org/ - A widely used biodiversity data standard enabling interoperability with GBIF and other biodiversity infrastructures.
- FAIR data principles - https://www.go-fair.org/fair-principles/ - Principles for making data findable, accessible, interoperable, and reusable.
- MegaDetector documentation (Microsoft AI for Earth) - https://github.com/microsoft/CameraTraps - Standards and conventions for animal detection models commonly used in camera trap workflows.
- eMammal camera trap protocols - https://emammal.si.edu/protocols - Best practices for camera deployment, metadata capture, QA/QC, and long-term monitoring.
- WCAG accessibility standards - https://www.w3.org/WAI/standards-guidelines/wcag/ - Accessibility guidelines applicable to research dashboards and annotation tools.
- Nielsen Norman Group usability heuristics - https://www.nngroup.com/articles/ten-usability-heuristics/ - Core UX principles for evaluating interface and workflow usability.
- OCI (Operational Camera Trap Metadata Standard) - https://github.com/tdwg/camtrap-dp/blob/main/metadata/README.md - Guidance for consistent camera trap metadata capture across projects.
- Open Geospatial Consortium standards (OGC) - https://www.ogc.org/standards - Standards for spatial metadata and georeferencing, relevant when publishing precise camera trap locations.
- Snapshot Safari / Zooniverse project design guidelines - https://help.zooniverse.org/kb/ - Guidance on annotation UI/UX, workflow design, and volunteer engagement for large-scale projects.

Instructions:
* Switch to plan mode, I want this task to be done with "plan mode on"
* Read all MD file in root to get a understanding of the project. 
* If something is unclear at any point, stop and ask before continuing.
* Prioritize simplicity and clarity over perfection. The code must be clean, easy to read, and understandable for collaborators. Avoid unnecessary complexity.
* I'm not in a rush. Please be precise and do the task thoroughly. 
* Please ask me any question for clarification. I would rather that you ask too many questions than assume certain details. 
* Ask me clarifying questions before beginning. Based on the conventions set out in CONVENTIONS.md and your knowledge, give your recommended solution to each questions you ask me. The minimum number of questions to ask me is 10




- [ ] bug: similarity verification: if I double click a verifed detection, clicking the "unverify" button doesnt do anything. It doesnt update the modal. But if I esc the modal, it does show up as non verified, so it obviously does something, but just not to the modal itself. 

- [ ] bug: when doing an analysis, the 'image classification' pbar goes from 0 to 100 without showing any stats like the other pbars. Then the 'finalizing...' part takes long. I get the sense that eveything happens in the 'finalizing...' phase. Could that be true? Investigate. 

- [ ] There is something weird going on. If I vcerify all detections as "Spotted bird" via similarity verification, i still see other labels in the dashboard. Why? Are you not taking the settings detection threshold into account? Check the DB for http://localhost:5173/projects/dc6f3a78-5a8b-4b8c-a959-cd1ade3c481a/dashboard
 

## Priority 2
- [ ] dashboard verification vard, explenation text "Event representatives are one file per event, used for quick review." explain a bit more how that representative is chosen. See event verification guide for more info. 
- [ ] dashboard verification vard, explenation text "Detections are individual animal, person, or vehicle bounding boxes within files." bounding box is jargon. Make it "observations" or somehting like that. Same with "bounding boxes " in the lines after that. 
- [ ] the dropdown widgets in the project settings page are not the same width as the other widgets. This makes it look off. Make it easy on the eyes. How? What would you porpose for UI? 

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
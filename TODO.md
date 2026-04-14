## Priority 1
- [ ] 

## Priority 2
- [ ] 

## Priority 3
- [ ] 

## New features
- [ ] TIMELAPSE STANDALONE APP
- [ ] DEPTH ESTIMATION
- [ ] PROCESS BATCH RESULTS - https://github.com/agentmorris/MegaDetector/blob/main/megadetector/postprocessing/postprocess_batch_results.py
- [ ] BATCH PROCESSING OPTION - a completely separate option to do batch processing, where it just runs det+cls on all data recursively, and is agnostic of its contents (doestn need to know if it is a deployment, site, full project, etc.) It just runs everything at once. Users should be able to set settings before running the analysis. Then, after, it should give the user a few options, like export to CSV, XLSX, maps & graphs, separate into subfolders, etc. The bulk / management choice should be the first page users see when opening AddaxAI. 
- [ ] IN DEPTH PLOTS, have a header in the menu "plots", and add page wide full plots that are interactive and with a bunch of filters and settings above. The dashboard is meant as a quick glance of the project, and these are more in depth to find out patterns etc. We will be adding in depth plots as we progress with the project, but the first one is "Comparison of the activity time" (improve wording, make it short and memorable), where the user can select up to 5 labels and compare the activity. One option should be to add the suntimes to the graph (sun hours, sunset, etc.), another option would be to have the actual time on the x axis, or the UTC times (based on the suntimes), do webqueries on how other platforms do this, and what the standard is, and what is usually reported in terms of metrics. Research scientific papers etc. 
- [ ] IN DEPTH PLOT - add new in depth plot: Gantt-style timeline — one horizontal bar per deployment (or per site), showing the active period. Immediately shows gaps, overlaps, and total survey effort. Group by site with one bar per deployment within each site row. 
- [ ] EXPORT OPTIONS - check AddaxAI Connect and copy from there. 
- [ ] TIMEZONE SETTING - make a timezone setting in the settings page, check how Connect does it. What should be the default? UTC? I dont know. Perhaps it should be a required setting when creating a project, what do you think? That determines the suncalc in the Activity patterns. When we have that, we can make a plot with activity patterns and sun hour overlays. 
- [ ] make caption or title of setting timezone more explicit. "Whatever the cameras were set to."

Based on your understanding of the project, what do you propose as the  
  optimum solution in regarding the DB sotrage of timestamps as read from the data? Local time as rtead from the data stored as is, or stored as UTC and then convertedf back for UI purposes? I do not like the idea of having two different conventions. I also have the 
   idea that the best prctises would be to store UTC in the DB and show UTC+timezone in the UI. 
   Do a full audit and provide a recommendation to me. It doesnt matter if we need to adjust    
  loads of code. Better now than in a year or so. 



## Installer
- [ ] merge all alembic/versions/ into one. We do not have any users yet, so we can make it just the start DB. 
- [ ] make sure on app istall it installs the default models and their environments (MDv5A and DINOv2-B). 

## Nice to haves
- [ ] Add a geolocation file for the Sub Saharan model too, like SpeciesNet, so users of the SSmodel can also prefil by country. 
- [ ] would it make sense to upgrade the app to use DINOv3 instead of DINOv2?
- [ ] ROLLDOWN - exlusion rollup currently works like this (corect me if i'm wrong!):
    > raw: wolf 60%, dog 20%, bear 10%, cat 10%.
    > exluded wolf, included dog, cat, bear: dog 20%, bear 10%, cat 10%, wolf 0%. 
    > Detection was wolf top-1, which is excluded, so it check the parent taxon to see if that is included. Canidea is included (via dog), so the deteciton gets the prediction (canidae 80%), am I correct? 
    > What if we also allow rolldown if there is only one child taxon present? The above example would then go to canidae 80% and see that there is only one canidae possible, so it must be that one, so it rollsdown to dog 80%. What do you think? If the raw prediction was wolf 60%, dog 20%, fox 10%, cat 10%, the rolldown wouldn't have worked since then there are two childs of the canidae, and hence the prediciton would remain canidae 80%. Agree? What do yout think of this appraoch? And what would it take to implement it? 
- [ ]
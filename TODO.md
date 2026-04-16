## Priority 1
- [ ] 

## Priority 2
- [ ] should we have the save settings button as a hover or a footer bar so that it is always visible? The page has quite a lot of settings and the save button is not very clear. what do you propose in terms of UX UI? 

## Priority 3
- [ ] In the deployments page, there should be an "info" option that opens a model that shows the path, the number of files (img/vid), events, observations, average confidences, etc. Just some insights into the deployment for investigation purposes. 
- [ ] Add the country or region to the TZ dropdown too and make sure the search also does that. So not only "Nairobi" but something like Kenya, Nairobi. What do you think? How would users search? By country, city, or continent? What is good UI UX? I've already implemented this fix in the other project. Check how that is implemented there: /Users/peter/Documents/Repos/AddaxAI-Connect/. It has a few mapping files etc, so make sure you copy it like its being used there. 

## New features
- [ ] TIMELAPSE STANDALONE APP
- [ ] DEPTH ESTIMATION
- [ ] PROCESS BATCH RESULTS - https://github.com/agentmorris/MegaDetector/blob/main/megadetector/postprocessing/postprocess_batch_results.py
- [ ] BATCH PROCESSING OPTION - a completely separate option to do batch processing, where it just runs det+cls on all data recursively, and is agnostic of its contents (doestn need to know if it is a deployment, site, full project, etc.) It just runs everything at once. Users should be able to set settings before running the analysis. Then, after, it should give the user a few options, like export to CSV, XLSX, maps & graphs, separate into subfolders, etc. The bulk / management choice should be the first page users see when opening AddaxAI. 
- [ ] IN DEPTH PLOT - add new in depth plot: Gantt-style timeline — one horizontal bar per deployment (or per site), showing the active period. Immediately shows gaps, overlaps, and total survey effort. Group by site with one bar per deployment within each site row. do webqueries on how other platforms do this, and what the standard is, and what is usually reported in terms of metrics. Research scientific papers etc. 
- [ ] EXPORT OPTIONS - check AddaxAI Connect and copy from there. 
- [ ] TIMEZONE SETTING - make a timezone setting in the settings page, check how Connect does it. What should be the default? UTC? I dont know. Perhaps it should be a required setting when creating a project, what do you think? That determines the suncalc in the Activity patterns. When we have that, we can make a plot with activity patterns and sun hour overlays. 
- [ ] make caption or title of setting timezone more explicit. "Whatever the cameras were set to."





  - json_pipeline.load_json_to_database now pre-flights every file's capture timestamp   
  before touching the DB. If any image has no extractable EXIF DateTimeOriginal (or      
  exiftool date for videos), it raises MissingTimestampError with the file list — no     
  silent mtime/utcnow fallback. Rollback is free because nothing has been written yet. 
  -> But MD extracts the datetime, right? It puts it in the JSON. Dont we use that for our DB? 

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
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

TASK: 

add a completely separate option to do batch processing, where it just runs det+cls on all data recursively, and is agnostic of its contents (doestn need to know if it is a deployment, site, full project, etc.) It just runs everything at once. Users should be able to set settings before running the analysis. Then, after, it should give the user a few options, like export to CSV, XLSX, maps & graphs, separate into subfolders, etc. 

IN terms of UX, where and how should this feature live? A separate page where users can navigate to "bulk" or "project based / management feature"? Or something else? I'm envisioning more options later, like projects for fish, projects page for drones, etc. So we need to think ahead of how we want to do this UX UI wise. What is your best guess on how it should go? Try to think like an AddaxAI user. 



- [ ] IN DEPTH PLOT - add new in depth plot: Gantt-style timeline — one horizontal bar per deployment (or per site), showing the active period. Immediately shows gaps, overlaps, and total survey effort. Group by site with one bar per deployment within each site row. do webqueries on how other platforms do this, and what the standard is, and what is usually reported in terms of metrics. Research scientific papers etc. 

## Installer
- [ ] merge all alembic/versions/ into one. We do not have any users yet, so we can make it just the start DB. 
- [ ] make sure on app istall it installs the default models and their environments (MDv5A and DINOv2-B). 

## Nice to haves
- [ ] SUBSAHARA GEOFILE - Add a geolocation file for the Sub Saharan model too, like SpeciesNet, so users of the SSmodel can also prefil by country. 
- [ ] CLS THRESH - add a classification threshold and a per species override. Check how that is one in AddaxAI-Connect. I want something like that.  
- [ ] DINOv3 - would it make sense to upgrade the app to use DINOv3 instead of DINOv2?
- [ ] ROLLDOWN - exlusion rollup currently works like this (corect me if i'm wrong!):
    > raw: wolf 60%, dog 20%, bear 10%, cat 10%.
    > exluded wolf, included dog, cat, bear: dog 20%, bear 10%, cat 10%, wolf 0%. 
    > Detection was wolf top-1, which is excluded, so it check the parent taxon to see if that is included. Canidea is included (via dog), so the deteciton gets the prediction (canidae 80%), am I correct? 
    > What if we also allow rolldown if there is only one child taxon present? The above example would then go to canidae 80% and see that there is only one canidae possible, so it must be that one, so it rollsdown to dog 80%. What do you think? If the raw prediction was wolf 60%, dog 20%, fox 10%, cat 10%, the rolldown wouldn't have worked since then there are two childs of the canidae, and hence the prediciton would remain canidae 80%. Agree? What do yout think of this appraoch? And what would it take to implement it? 
- [ ]




- [x] I want to add an option to avoid linking a deployment to a site. The site dropdown should be optional, with the placeholder "Leave blank if unknown or from multiple locations". That means the site can be nullable. Then, if a deployment has no site (site = NULL), show a "-" in the deployments page. The features that need GPS coordinates must show a warning banner with something like, X deployment(s) have no camera site filled in. Skipping...." (improve wording) and then a button that shows the deployments in question in the deployments page. 

- [x] it might be better for the mind flow UX UI to not placeholder the site selection with "Leave blank if unknown or from multiple locations" because it then suggests that you can run it over several sites at the same time (which is true of course, but better to do it one deployment at the time). Perhaps we just want it so show "Optionally select a location". That suggests nothing and the user chooses to do nothing with the site, so they know they need to fix it later if they want analyses with loications. Makes sense? 

- [x] Should we just keep calling the unit a 'deployment', even though it might actually not be a deployment (if user runs multiple deployments at once). We could rename the unit to something more generic like 'runs' or something like that, but i feel like users know about deployments from other platforms and dont want to learn more jargon. It also tries to motivate people to work in deployments, and if it has mixed data, its the users fault. They can merge/split later if they want (feature to be added - dont have to implement now).

- [x] should we add a info text to the folder selection? It is there now already, but update it with the knowledge we have now. Explain that you can run any folder and it will recursively analyse all images and videos. But if you want research grade analyses and camtrapDP export, you'll need to run them one deployment at a time (but they can add them in the queue so they can still run it in batches). If they just want to run the whole backlog of data in one go, thats fine too. Then they just select the whole project folder. They can always splot them into smaller chunks later and add locations if they decide to do more analayises. Makes sense? Then the user knows about it, but its not in your face directly. Only if the user hovers the info icon. Or would you advise against this since it just complicates things? Just ask for deployment, and if they want to add locations, merge split etc, they can find out about it later. Progressive disclusure UX. Dont make it overly complex if they dont need to know about it. What do you think?

- [x] should we add the "(optional)" text to all the input fields of the new deployment analysis form? To all except the folder intpu of course. Or would you advise against that in terms if UX UI / cluttering / screen real estate. 

- [ ] Some users might have run analysis over a bunch of deployments at the saem time (backlog of data with 3 sites and 10 deployments). So if he want to show them all in the map and in the exports etc, he needs a way of splitting the deployment over several smaller ones that all have a single site. SO we need to add this feature. Investigate. Same for splitting deployments, we might need to add a feature that merges them. Same as splitting, but exactly the opposite.

- [ ] After that we can think about how we want to make sure the CamtrapDP export checks for deployments. I think a fully automatic check is error prone. WOuldnt it just be easiest if it just is a text model where it is described and refered to the deploymetns page where users can merge and split deployments to make a single deployment per unit. Do you understand what I'm saying? 

- [ ] Add the "(no site)" option to all site filters trhoughout the app. That way users can select the data that has no site attached to them. Only show if there actually are data with no sites attached. Makes sense? 

- [ ] Right now we have key:value tags and notes for both sites and deployments, which is good. But might also be confusing... "Didnt I alreay filled this in?". SHould we rename them to "Site notes" / "Deployment notes" (might be bad idae as we duplicate words and add visual clutter). Perhaps better to update the placeholder to match the site and deployment specific inpupts better. What do you thnink? 

- [ ] currently there is a check at the end of analysis (see below). Should we check all timestamps before analysis and warn user upfront (might add extra processing time before startiong analyis), or we check it after detection has been done (like now is implemented, but we show it as a warning instead of an error). "These did not have any tiemstamps and are therefore excluded in analysis." Agree? Now if there is a single image with currupted metadata, it blocks everything. We should bascially just make a log file structure or something like that. "There were some warnings, see ... for more info." And then just continue with the ones that are in order. What do you think? 
                                                                                                                       
            No extractable capture timestamp for 10 file(s):                                                                     
            /Users/peter/Downloads/example-data/project_Ukraine/loc_SIMON03/dep001/01_dan_IMG_0004.AVI,                          
            /Users/peter/Downloads/example-data/project_Ukraine/loc_SIMON03/dep001/02_bobcat_IMG_0180.AVI,                       
            /Users/peter/Downloads/example-data/project_Ukraine/loc_SIMON03/dep001/03_bobcat_IMG_0256.AVI,                       
            /Users/peter/Downloads/example-data/project_Ukraine/loc_SIMON03/dep001/04_coyote_IMG_0096.AVI,                       
            /Users/peter/Downloads/example-data/project_Ukraine/loc_SIMON03/dep001/05_owl_IMG_0108.AVI (+5 more)                 
                                                                                                                                
            Does this mean no data was added to the DB? Or that only these were skipped? 


- [ ] If you select "no embedding model" in the settings, then you obviously dont have any embeddings to verify similarity. Currently it says the below. It should say something like, to use this feature, you need to slelect an embedding model in the settings and save. It will then run embedding on all your data. (related but other bug: when doing this, i embedde everything with dinov2, success, but the similarity verification is still empty after hard reload. ALso no placeholder text. Just empty - missing grid, but filters are visible. Is that because they are all attached to '(no site)' and this one is not selected in the site selector? Probabaly, haha.)

              No embeddings yet

              Run an analysis with an embedding model selected to use similarity features. Embeddings are computed from detection crops using DINOv2.

- [ ] wehn running MD only (no cls model), and adding a box via event verification, accepting auto label "animal". It shows up as two labels with two colors, both "Animal". I have a hiunch that it is about capitalisation. The MD produced ones are "Animal" and the bbox added ones are "animal", right? Fix this. 

- [ ] Almost everything depends on having the analysis done per deployment (event creation - if overlapping DateTime stamps in different fodler - merged into one), the trap nights (affects all statistics, maps, exports, etc). Should we not add a warning at the analysis step? Maybe a note like (should be one deployment per analysis. Affects all statistics, events, plots, smoothing, settings, and exports. If analysing a backlog and you dont want to care about structure, that is fine, but please take this into account. You can always split your backlog folder into proper deployments... )
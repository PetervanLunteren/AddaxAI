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





- [ ] After that we can think about how we want to make sure the CamtrapDP export checks for deployments. I think a fully automatic check is error prone. WOuldnt it just be easiest if it just is a text model where it is described and refered to the deploymetns page where users can merge and split deployments to make a single deployment per unit. Do you understand what I'm saying? 
- [ ] Almost everything depends on having the analysis done per deployment (event creation - if overlapping DateTime stamps in different fodler - merged into one), the trap nights (affects all statistics, maps, exports, etc). Should we not add a warning at the analysis step? Maybe a note like (should be one deployment per analysis. Affects all statistics, events, plots, smoothing, settings, and exports. If analysing a backlog and you dont want to care about structure, that is fine, but please take this into account. You can always split your backlog folder into proper deployments... )

- [ ] reevaluate the trap nights calculation. If we can make that one also mixed-deployment proof, that would save us a lot of headaches later when analysing backlogs. 

# other

- [ ] the Detection trend card in the dashboard reports only the ticks it finds, not 0 days or months. Do you see what I mean? A observation on 1 jan, and one on 1 juli would show as a two tick straight line. It should also shjow the empty ticks. 

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


## Priority 1
- [ ] 

## Priority 2
- [ ]

## Priority 3 
- [ ] 

## New features
- [ ] TIMELAPSE STANDALONE APP
- [ ] MULTI LANGUAGE SUPPORT
- [ ] DEPTH ESTIMATION
- [ ] PROCESS BATCH RESULTS - https://github.com/agentmorris/MegaDetector/blob/main/megadetector/postprocessing/postprocess_batch_results.py
- [ ] IN DEPTH PLOT - add new in depth plot: Gantt-style timeline — one horizontal bar per deployment (or per site), showing the active period. Immediately shows gaps, overlaps, and total survey effort. Group by site with one bar per deployment within each site row. do webqueries on how other platforms do this, and what the standard is, and what is usually reported in terms of metrics. Research scientific papers etc. I want it research grade analysis. 


- in the deployment edit from, there is a site selection. SHould we add a note/caption below it saying: "If your deployment contains data from multiple sites, please [split](link-that-goes-to-the-current-deployment-split=feature) them first, and then assign the proper site to each."

- Same for the analysis from at the begining. We might add "If the selected folder conatins multiple sites, consider splitten them add adding them to the queue separately." ot is this too much noise on the front page form? At the deployment edit modal it is not every day that a user comes there, so we can add some more noise/texts/explenations. What do you think? Keep it clean? Or add notes/captions to the analysis form at the main page of the project. 

- In the about this view, you put down a bunch of citations. Please use only the ones you actually refer to in the text. So if you can ref to papers in the text, do so, and add only those to the ref section. Do this task for the other insight pages too, to double check and add where possible. 

- how does it look when a single site has more than one concurrent trap night interval? How should we handle that in the deployments timeline plot?


- in the Deployment info slideout, we should have a deployment ID with a copy button. Same for the site info slideout. The ID is important for people if they want to share info with collegues. 



## Installer
- [ ] merge all alembic/versions/ into one. We do not have any users yet, so we can make it just the start DB. 
- [ ] make sure on app istall it installs the default models and their environments (MDv5A and DINOv2-B). 

## Nice to haves
- [ ] SUBSAHARA GEOFILE - Add a geolocation file for the Sub Saharan model too, like SpeciesNet, so users of the SSmodel can also prefil by country. 
- [ ] CLS THRESH - add a classification threshold and a per species override. Check how that is one in AddaxAI-Connect. I want something like that.
- [ ] 


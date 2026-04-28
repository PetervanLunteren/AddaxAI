## Priority 1
- [ ] 

## Priority 2
- [ ] remove the "today" legend item of the timeline plot, no need for that. 

## Priority 3 
- [ ] 

- [ ] I'm a bit confused as how the navigation works in the events and images details modals. We have left and right for just normal navigation (regardless of verification status), and we have >> for next unverified. Is that correct? Should we add a << for previous unverifed? Should we do the arrows for the simple navigation (just prev next regardless oof verificatin status)? Should we do default navigation per maxN in the events tab, and something like SHIFT+navigation for frame navigation...? There is also a "Navigate by maxN / file" dropdown in the events modal. Perhaps redundant now that we have the images tab? Anyways, list all the options and think about the UX UI. What are the options, possibilites, user needs, and how can we make this simple. Its confusing now. 

- [ ] we renamed the Files tab to Images. Do we need to change the underlaying parameters in the code too to keep clean? 

- [ ] the "All labels" button should not have an icon (inconsistent with the other filters). remoce icon from that button in all three tabs of the verificaiton flow. 

- [ ] Should we improve the caption of the verify page now that we have three options? Do a few suggestions. 

- [ ] if all is done and fixed, focus on the texts. The (?) icon in the files Modal and files page, it reidrects to the event verification slideout. Make one specifically for the files. Propose a few options so I can select one.




## New features
- [ ] TIMELAPSE STANDALONE APP
- [ ] MULTI LANGUAGE SUPPORT
- [ ] DEPTH ESTIMATION
- [ ] PROCESS BATCH RESULTS - https://github.com/agentmorris/MegaDetector/blob/main/megadetector/postprocessing/postprocess_batch_results.py
- [ ] DOCUMENTATION

## Installer
- [ ] merge all alembic/versions/ into one. We do not have any users yet, so we can make it just the start DB. 
- [ ] make sure on app istall it installs the default models and their environments (MDv5A and DINOv2-B). 
- [ ] 

## Nice to haves
- [ ] SUBSAHARA GEOFILE - Add a geolocation file for the Sub Saharan model too, like SpeciesNet, so users of the SSmodel can also prefil by country. 
- [ ] CLS THRESH - add a classification threshold and a per species override. Check how that is one in AddaxAI-Connect. I want something like that.
- [ ] 


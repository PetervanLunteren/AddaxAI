## Priority 1
- [ ] 

## Priority 2
- [ ]

## Priority 3 
- [ ] FILE VERIFICATION - add a section for image verification. Or at least thinik about it. You have something for events and detections (called similarity now). Should we make a thrid tab, files? exactly the same as events, but then on the file level. if you search fot wolf, you get all the images or frames with a wolf. Now in events you still have to search trhough the event to find it. events verify MaxN (and files if you want), files verify files (and if lucky its a maxN too), decections (or did we choose to call it observations? I think so), verify on the instance level. Here we do the embedding too. SO basically, just leave events and similarity as they are (perhaps rename similarity), and add a new one for files (which is almost the same as events, just not grouping for events). And while we're at it, should we make these their separate pages? Then we have all levels: sites, deployments, events, files, observations. Or would you advise against that and keep it all three in verify page as tabs? What is you recommendation in terms of UX UI? Be honest, dont sugar coat. 
- [ ] 

## New features
- [ ] TIMELAPSE STANDALONE APP
- [ ] MULTI LANGUAGE SUPPORT
- [ ] DEPTH ESTIMATION
- [ ] PROCESS BATCH RESULTS - https://github.com/agentmorris/MegaDetector/blob/main/megadetector/postprocessing/postprocess_batch_results.py
- [ ] 

## Installer
- [ ] merge all alembic/versions/ into one. We do not have any users yet, so we can make it just the start DB. 
- [ ] make sure on app istall it installs the default models and their environments (MDv5A and DINOv2-B). 
- [ ] 

## Nice to haves
- [ ] SUBSAHARA GEOFILE - Add a geolocation file for the Sub Saharan model too, like SpeciesNet, so users of the SSmodel can also prefil by country. 
- [ ] CLS THRESH - add a classification threshold and a per species override. Check how that is one in AddaxAI-Connect. I want something like that.
- [ ] 


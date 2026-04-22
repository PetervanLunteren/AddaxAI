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
- [ ] IN DEPTH PLOT - confusion matrix
- [ ] IN DEPTH PLOT - confusion table

## Installer
- [ ] merge all alembic/versions/ into one. We do not have any users yet, so we can make it just the start DB. 
- [ ] make sure on app istall it installs the default models and their environments (MDv5A and DINOv2-B). 

## Nice to haves
- [ ] SUBSAHARA GEOFILE - Add a geolocation file for the Sub Saharan model too, like SpeciesNet, so users of the SSmodel can also prefil by country. 
- [ ] CLS THRESH - add a classification threshold and a per species override. Check how that is one in AddaxAI-Connect. I want something like that.  
- [ ]


# other

- [ ] currently there is a check at the end of analysis (see below). Should we check all timestamps before analysis and warn user upfront (might add extra processing time before startiong analyis), or we check it after detection has been done (like now is implemented, but we show it as a warning instead of an error). "These did not have any tiemstamps and are therefore excluded in analysis." Agree? Now if there is a single image with currupted metadata, it blocks everything. We should bascially just make a log file structure or something like that. "There were some warnings, see ... for more info." And then just continue with the ones that are in order. What do you think? 
                                                                                                                       
            No extractable capture timestamp for 10 file(s):                                                                     
            /Users/peter/Downloads/example-data/project_Ukraine/loc_SIMON03/dep001/01_dan_IMG_0004.AVI,                          
            /Users/peter/Downloads/example-data/project_Ukraine/loc_SIMON03/dep001/02_bobcat_IMG_0180.AVI,                       
            /Users/peter/Downloads/example-data/project_Ukraine/loc_SIMON03/dep001/03_bobcat_IMG_0256.AVI,                       
            /Users/peter/Downloads/example-data/project_Ukraine/loc_SIMON03/dep001/04_coyote_IMG_0096.AVI,                       
            /Users/peter/Downloads/example-data/project_Ukraine/loc_SIMON03/dep001/05_owl_IMG_0108.AVI (+5 more)        


- [ ] If you select "no embedding model" in the settings, then you obviously dont have any embeddings to verify similarity. Currently it says the below. It should say something like, to use this feature, you need to slelect an embedding model in the settings and save. It will then run embedding on all your data. (related but other bug: when doing this, i embedde everything with dinov2, success, but the similarity verification is still empty after hard reload. ALso no placeholder text. Just empty - missing grid, but filters are visible. Is that because they are all attached to '(no site)' and this one is not selected in the site selector? Probabaly, haha.)

              No embeddings yet

              Run an analysis with an embedding model selected to use similarity features. Embeddings are computed from detection crops using DINOv2.

- [ ] wehn running MD only (no cls model), and adding a box via event verification, accepting auto label "animal". It shows up as two labels with two colors, both "Animal". I have a hiunch that it is about capitalisation. The MD produced ones are "Animal" and the bbox added ones are "animal", right? Fix this. 

- [ ] There is currently no cancel button when running ML (proicessing deployments). Is it difficult to add that? And would it be difficult to add a pauze button? The cancel button is pretty improtant, but the pauze button not so much. Dont make it too complex. KISS. 
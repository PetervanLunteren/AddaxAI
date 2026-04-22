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

- [ ] wehn running MD only (no cls model), and adding a box via event verification, accepting auto label "animal". It shows up as two labels with two colors, both "Animal". I have a hiunch that it is about capitalisation. The MD produced ones are "Animal" and the bbox added ones are "animal", right? Fix this. 

- [ ] There is currently no cancel button when running ML (proicessing deployments). Is it difficult to add that? And would it be difficult to add a pauze button? The cancel button is pretty improtant, but the pauze button not so much. Dont make it too complex. KISS. 



Processed 1 deployment.
Successfully analysed 46 images.

should me one line “Successfully processed 3 deployments with 46 images and 23 videos.”
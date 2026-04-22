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
- [ ] IN DEPTH PLOT - add new in depth plot: Gantt-style timeline — one horizontal bar per deployment (or per site), showing the active period. Immediately shows gaps, overlaps, and total survey effort. Group by site with one bar per deployment within each site row. do webqueries on how other platforms do this, and what the standard is, and what is usually reported in terms of metrics. Research scientific papers etc. I want it research grade analysis. 
- [ ] IN DEPTH PLOT - confusion matrix
- [ ] IN DEPTH PLOT - classification report

MATRIX
- make normalised yes/no option on the matix
- Why are there not the same rules with the taxonomic rank filter in the dashboard as in the matrix? So if not species level but higher, they get aggregated to "Higher level taxa". Model classes with no taxonomy get "No taxonomy". Also, its called "Taxonomic rank" and it should also have the option "Most specific" (which is the default). This is all already done correctly in the taxonomic rank filter in the dashboard. Check how it works there and investigate how it should work in the matrix. And while were at it, also make it work like that in the classification report. And while were at it, should we make a single source of truth here? How would that work? I can imagine that this filter would come back in many analysis plots later on, so better now think of the most efficient solution. Agree?  


## Installer
- [ ] merge all alembic/versions/ into one. We do not have any users yet, so we can make it just the start DB. 
- [ ] make sure on app istall it installs the default models and their environments (MDv5A and DINOv2-B). 

## Nice to haves
- [ ] SUBSAHARA GEOFILE - Add a geolocation file for the Sub Saharan model too, like SpeciesNet, so users of the SSmodel can also prefil by country. 
- [ ] CLS THRESH - add a classification threshold and a per species override. Check how that is one in AddaxAI-Connect. I want something like that.  
- [ ] 



## Priority 1
- [ ] 

## Priority 2
- [ ] 

## Priority 3 
- [ ] 


## Future stuff
- [ ] TIMELAPSE STANDALONE APP
- [ ] MULTI LANGUAGE SUPPORT
- [ ] DEPTH ESTIMATION
- [ ] PROCESS BATCH RESULTS - https://github.com/agentmorris/MegaDetector/blob/main/megadetector/postprocessing/postprocess_batch_results.py
- [ ] DOCUMENTATION
- [ ] REPEAT DETECTION ELIMINATION
- [ ] WLIDBOOKS INTEGRATION

## Installer
- [ ] merge all alembic/versions/ into one. We do not have any users yet, so we can make it just the start DB. 
- [ ] make sure on app istall it installs the default models and their environments (MDv5A and DINOv2-B). 
- [ ] Update the release tag

## Nice to haves
- [ ] SUBSAHARA GEOFILE - Add a geolocation file for the Sub Saharan model too, like SpeciesNet, so users of the SSmodel can also prefil by country. 
- [ ] CLS THRESH - add a classification threshold and a per species override. Check how that is one in AddaxAI-Connect. I want something like that.
- [ ] 



1: when I release with notes in GitHub online
2: build arm64 only
3: hard fail
4: yes, but please double check in the action.yml
5: wire up all three at the same time using a matrix, but make Windows and Linux non-blocking for now
6: matrix-strategy single workflow
7: 3.11 + Node 20
8: yes, sync the version from the git tag
9: Not sure what this means. Please elaborate. I thought the assets (installer files, etc) are stored there indefinately? Or will they be gone after a number of days?
10: yes to both.


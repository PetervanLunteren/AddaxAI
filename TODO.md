## Priority 1
- [x] TIMELAPSE INTEGRATION (in-app window + --timelapse CLI + open.bat shim)

- [ ] Test with Timelapse itself
- [ ] Test with a clean install without the setup done in the main AddaxAI app
- [ ] 
- [ ] 
- [ ] 







- [ ]
- [ ] TIMELAPSE STANDALONE APP
- [ ] The app shows "AddaxAI didn't shut down cleanly last time. If this is unexpected, export a diagnostic report and email it to support." quite often. Not sure why, but it seems to also show it there wasnt much going on. IS it a little to tight or is there really something going on?
- [ ] ALLOW FULL IMAGE CLS MODELs TOO (AHDRIFT-v1)
- [ ] ADD ALL MODELS 
- [ ] ask to Saul to add AddaxAI.exe --timelapse "<folder>" to Timelapse's command list as the long-term path.
- [ ] If I check Apps > Installed apps in Windows, I see its created by "AddaxAI Team". Should we make this "Peter van Lunteren"?
- [ ] Show me the popup text for the Windows uninstall "Also delete your user data?". I want to fine tune it. First show it to me, then I'll give you tasks.  

## Priority 2
- [ ] Activity overlap. The default pick for label A is the most common one, right (max samples)? Should we auto pick the second most common one for label B? Then they see directly what it does. 

## Priority 3 
- [ ] 

## AFter the Beta phase
- [ ] If everything works and all models are verified, please double check if there are any stale environment.ymls that are never used by any of the models. If so, remove them. 
- [ ] Any other non used imports or requirements in the environments YMLS? 
- [ ] Bump addaxai-base from cu118 to cu128 so RTX 50-series (sm_120, Blackwell) gets native kernels instead of the 4-5 min PTX JIT fallback. Suggested pins: torch==2.8.0+cu128, torchvision==0.23.0+cu128, --extra-index-url https://download.pytorch.org/whl/cu128. Both windows and linux YAMLs. Adds ~700 MB to the install but fixes the GPU warning reported at https://forum.addaxai.com/t/model-warning-on-running-with-gpu/202. Requires NVIDIA driver >= 555.x, mention in the beta-tester readme. 
- [ ] Bump the pytorch env from Python 3.8 to 3.11 (3.8 is EOL since Oct 2024 and recent torch wheels are starting to drop py38 builds). Also bump torch alongside the python jump. SpeciesNet-fine-tuned classifiers (.pt files with pickled onnx2torch operator classes) need a smoke test after the bump: load NAM-ADS-v1 or similar and confirm torch.load() succeeds across the major version jump. 
- [ ] Do we want a custom minimal menu (just Reload / Force Reload / DevTools / About / Quit) with our own styling? Or keep the electron built in? We can put the hamburger menu in the electron menu row? And the bug report etc. Then we can also add video tutorials etc. 
- [ ] Would it be a good idea to add a extra level for the smoothing "Very aggresive" (or something similar), that does not run the MD utils smoothing script at all, but just flattens out the entire event to a single label. We'll need to think about which label of course (the max cls conf label for all? or some kind of average label for all?). If we decide to do this, we might also want to add captions in the dropdown that try to explain the tiers off / mild / aggresive / etc. Make it a tall dropdown just like the models dropdown with captions, use the same format. What do you think? 



## Future stuff
- [ ] MULTI LANGUAGE SUPPORT
- [ ] DEPTH ESTIMATION
- [ ] POSTPROCESS BATCH RESULTS MEGADETECTOR
- [ ] DOCUMENTATION
- [ ] REPEAT DETECTION ELIMINATION
- [ ] WLIDBOOKS INTEGRATION



## Nice to haves
- [ ] SUBSAHARA GEOFILE - Add a geolocation file for the Sub Saharan model too, like SpeciesNet, so users of the SSmodel can also prefil by country. 
- [ ] CLS THRESH - add a classification threshold and a per species override. Check how that is one in AddaxAI-Connect. I want something like that.
- [ ] 


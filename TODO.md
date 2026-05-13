## Priority 1
- [x] The app shows "AddaxAI didn't shut down cleanly last time. If this is unexpected, export a diagnostic report and email it to support." quite often. Not sure why, but it seems to also show it there wasnt much going on. IS it a little to tight or is there really something going on?
- [x] Add backups options. 
- [x] Make embedding count configurable and trackable. 
- [ ] Think about how to solve the frames issue. Check dans email. 
- [ ] Add different modes at the home screen. Something like
    > Simple mode (Point at folder, select model, get CSV)
    > .... [IDK propose name] (Point at folder, select model, write JSON, show postprocesing options like visualise, separte into subfolders, etc. )
    > Timelapse mode (Create JSON for timelapse) (Already written, just needs a card a t home page). 


    > Lets think about it a bit more before we go directly into planning. What is good UX, UI, what are good names? How do we want to define the modes? Can we merge modes? Have a look at how the legacy AddaxAI worked (that is how users are used to use AddaxAI: /Users/peter/Documents/Repos/AddaxAI). The legacy AddaxAI had two modes (simple and advanced). I dont think we need to recreate those 

    > I think it would be great to have a "simple mode" that includes things like folder separation and doesn't include things like projects/deployments, and I would probably make that the default.  Even going through it a second time, where I knew what to expect, I was still somewhat overwhelmed by the project/deployment structure that seems like it's only maybe relevant to AddaxAI (i.e., it sounds like this kind of integrated analysis is still more of a future goal?).  You should ignore this comment if you think 80% of AddaxAI users are going to be doing lots of spatial analysis in AddaxAI, but I would guess that 80+% of users will be using AddaxAI for the AI (it's literally in the name :) ), and the project structure may be confusing.  

    > It's true 100% of users have a concept projects/deployments, and 100% of users do spatiotemporal analyses, but I would guess that very few of them will do those in AddaxAI unless you write lots and lots and lots and lots and lots of population analysis code that I don't think you want to write, to entirely replace what people do in, e.g., camtrapR, spOccupancy, etc.  Assuming that most this functionality is going to be done in other packages, tracking deployments/projects in AddaxAI is asking users to track all of this in two places, without any direct way of connecting them.

    > Along the same lines, I don't see a common scenario where someone would want to export from AddaxAI directly to CamtrapDP... is that a scenario you've seen come up?  That seems outside the scope of what CamtrapDP is intended for; it's a format for (correct) observations, rather than (sometimes-correct) AI predictions.  FWIW I really really really really don't want people to use CamtrapDP to publish datasets that haven't been human-reviewed.

- [ ] Add variant rank. See future-plans/add-variant-rank.md
- [ ] ADD ALL MODELS 
- [?] ALLOW FULL IMAGE CLS MODELs TOO (AHDRIFT-v1)
- [?] ALLOW CANADA sex-age-classes GRANT MODEL 
- [ ] ask to Saul to add AddaxAI.exe --timelapse "<folder>" to Timelapse's command list as the long-term path.


## Priority 2
- [ ] 


Open a project with completed deployments → should land on Dashboard. - DOES NOT LAND ON DASHBOARD. LANDS ON PROCESS. 

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
- [ ] DOCUMENTATION - in text and in video tutorials
- [ ] REPEAT DETECTION ELIMINATION
- [ ] WLIDBOOKS INTEGRATION



## Nice to haves
- [ ] SUBSAHARA GEOFILE - Add a geolocation file for the Sub Saharan model too, like SpeciesNet, so users of the SSmodel can also prefil by country. 
- [ ] CLS THRESH - add a classification threshold and a per species override. Check how that is one in AddaxAI-Connect. I want something like that.
- [ ] Bulk-create sites and deployments from a text file (CSV). Columns: name, lat, lon, optional habitat, deployment folder, altitude. Requested by Simon for a 34-camera survey where adding each site/deployment by hand is laborious.
- [ ] Follow Windows .lnk shortcuts when walking deployment folders. Currently Python's os.walk treats .lnk as regular files and skips them. Workaround for users: NTFS junctions (mklink /J) are followed natively. Requested by Simon for a per-camera-per-week shortcut layout.
- [ ] Global toggle to show labels as common name vs scientific name. Labels already carry both via label_taxonomy.display_name; just need a per-user or per-project UI switch. Requested by Simon.
- [ ] Ranked flat label filter (descending count) as an alternative to the hierarchical tree on Verify and Insights. Requested by Simon.
- [ ] Show time delta between adjacent images / videos in EventDetailModal and FileDetailModal (e.g. "+12s since previous"). Helps verifiers spot independent events vs. continuous bursts. Requested by Simon.
- [ ] Verification-status layer on the Map insights page (e.g. dot color = % verified, or filter to deployments still unverified). Requested by Simon for tracking field-work progress across many cameras.
- [ ] Take the heatmap for deploment timeline feature from AddaxAI Connect
- [ ] Naïve occupancy insights page. See AddaxAI-Connect. 


# TODO

we made the env update into the toast, but why not open it in a blocking Modal (perhaps reuse the preparation modal)? no users can so other stuff, with the app and this happens in the background. Isnt that complex and error prone? Also, what happens if the user presses the X? I like KISS. If you cnacelled it didnt work. You have to wait, not parallel processes, no backrgound stuff. What do you think? 

                Environment update available
                The analysis environment ships a newer version than the one installed on this machine. Rebuild it to match this app version.

                env-pytorch
                Installing packages...



how do we decide which label to choose? the most aparent one? If so, add it to the caption. 

                Keep events together
                All photos and videos from one event go to the same folder


There seems to be some exclusions in the detections.csv in folder run (perhaps also in projects mode?). If I have a run of 6 coyotes and 4 AI suggested coyotes but verified birds, it only lists the coyotes in the detections.csv. Why? Investigate. Is this the only bug, or are there more like this?



## Priority 1
- [ ] Do all the other feedback received via email
- [ ] What does the label cleanup step / page really yeild for the user? From the user persective, what does the label cleanup really do? Do we want to leave it in the folder mode? It corects labels in the CSVs, JSONs, and it makes sugestions for events better, right? What else? It makes the confusion matix, and the future train function dataset creation. What else? DO we want to leave it? Or perhaps add something like this to the caption? Or something else? What do you think? TYhink as a user. Thiunk UX UI.
- [ ] How are the filmstrip thumbs sorted? I got an event where it wasnt oin chronoligcal order. The two images that were swapped had the same HH:MM:SS timestamp, but the order was wrong. How can we avoid that? If exact same timestamp order alphabitcally (given that camtrap images are generally ityerative IMG001.jpg etc.)?
- [ ] Should we add a option to dismiss info bars? They are info bars and non intrisive, but i can imagine that a user will annoy themselves if they have to look at the info mars at every progress modal. What do you think? Which ones should have a dismiss option, and whisch ones not? Or all? 
- [ ] It says "You have unsaved changes" every time i open the settings page. As a test: click "reset changes" the "You have unsaved changes" is not shown anymore, no apparent changes visible. Move different page, move back to settings, it shows "You have unsaved changes" again. Bug. Investigate. 
- [ ] Do we need to do anything with this finding from a user? For the current Electron build I mean. https://forum.addaxai.com/t/ssl-certificate-issue-in-addaxai-bundled-python-windows/306
- [ ] Do we need to set PYTHONNOUSERSITE=1 to stop users' global Python user-site packages from leaking into AddaxAI's envs? Is that a real risk? Or not worth it?


## Priority 2
- [ ] 

## Priority 3 
- [ ] 

## AFter the Beta phase
- [ ] ADD ALL MODELS (Also Caras model)
- [?] ALLOW FULL IMAGE CLS MODELs TOO (AHDRIFT-v1)
- [?] ALLOW CANADA sex-age-classes GRANT MODEL 
- [ ] ask to Saul to add AddaxAI.exe --timelapse "<folder>" to Timelapse's command list as the long-term path.
- [ ] If everything works and all models are verified, please double check if there are any stale environment.ymls that are never used by any of the models. If so, remove them. 
- [ ] Any other non used imports or requirements in the environments YMLS? 
- [ ] Bump addaxai-base from cu118 to cu128 so RTX 50-series (sm_120, Blackwell) gets native kernels instead of the 4-5 min PTX JIT fallback. Suggested pins: torch==2.8.0+cu128, torchvision==0.23.0+cu128, --extra-index-url https://download.pytorch.org/whl/cu128. Both windows and linux YAMLs. Adds ~700 MB to the install but fixes the GPU warning reported at https://forum.addaxai.com/t/model-warning-on-running-with-gpu/202. Requires NVIDIA driver >= 555.x, mention in the beta-tester readme. 
- [ ] Bump the pytorch env from Python 3.8 to 3.11 (3.8 is EOL since Oct 2024 and recent torch wheels are starting to drop py38 builds). Also bump torch alongside the python jump. SpeciesNet-fine-tuned classifiers (.pt files with pickled onnx2torch operator classes) need a smoke test after the bump: load NAM-ADS-v1 or similar and confirm torch.load() succeeds across the major version jump. 
- [ ] If we have bumped pytorch env to 3.11, we can implement https://github.com/MNHN-OFVI/DeepForestVisionV2. See email from Hugo at Tue, Jun 16, 4:01 PM for more info. 


## Future stuff
- [ ] MULTI LANGUAGE SUPPORT
- [ ] DEPTH ESTIMATION
- [ ] TRAIN FEATURE - https://agentmorris.github.io/speciesnet-fine-tuning/ - must do something to have it accept bounding boxes with verified labels from projects mode (should allow data from multiple projects, so where does thi live? Not inside a single project, right? perhaps better in the projects overview page?). The tutorial also focusses on SpeciesNet only, but for marine workflows it would be good if we could have it accept generic pretrained models too. If we could adjust the script so it can train a pretrained Efficientnet model too, that would be great!
- [ ] Add variant rank. See future-plans/add-variant-rank.md
- [ ] POSTPROCESS BATCH RESULTS MEGADETECTOR
- [ ] DOCUMENTATION (see items below)
- [ ] REPEAT DETECTION ELIMINATION
- [ ] WLIDBOOKS INTEGRATION

## Documentation
- [ ] Make a tutorial on how to move data between computers. "The difficulty is that AddaxAI uses three data sources, and all are required. The raw images and videos (to show you while doing verification)The internal JSON files hidden in the processed folders (to reprocess after settings are changed)The internal AddaxAI database (stores all detections, verification statuses, etc)If we want to move everything to a new computer, we must move all three of these components. Luckily, components 1 and 2 are together, so if you have the images on an external drive, you can just plug it into a new computer. Then, you also need to move the DB, which means you must back it up manually, move the DB file to the new computer, and then restore from the there. "
- [ ] in text and in video tutorials
- [ ] also all the models avaiulable with species etc 
- [ ] also include the fallback date reader from filename (...addaxai-YYYYMMDD-HHMMSS.ext)

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
- [ ] Update the exports page to match the 2 col format. Check Connect. 







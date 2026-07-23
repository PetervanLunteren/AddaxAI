# TODO

## Priority 1
- [ ] 

## Priority 2
- [ ]  

## Priority 3 
- [ ] 

## After the Beta phase




- [x] I want all models from the legacy AddaxAI also in the new AddaxAI (this repo). Here is the legacy AddaxAI: /Applications/AddaxAI_files. figure out how it works. This is a large task, as it also includes converting the taxonomy CSV formats, etc. SO please investigate hoe the old format looks like, and the new one. And how to convert (I believe there are scripts already written that do this conversion, so please investgate and search for this). Perhaps I need to updaload some files to HF, and if so, please tell me exactly which file, to which HF repo. I want to be intructed clearly by you. Make sure to ask me clarifying questions if uncertain, that is very important. I want you to do everything (except the HF stuff, and perhaps other things that are just easier manually), but I'm here to help. Use me for clarification. You do not have to do this alone. Also add a test script to see whether or not all models work, and if they all take GPU/MPS correctly. It can be a standalone test that I run manually on all OS (no need to run this on every push or release - just if a dev wants to double check). Just to see if they run, if they take GPU/MPS, and if they have the expected outcome (probabaly best to download a camera trap image from LIAL BC to check the results with what is expected). How can we know what is expected? We need to double check eveything with the outcome pof these models on the legacy AddaxAI. So lets build a plan that does this all for us. Here is the legacy AddaxAI: /Applications/AddaxAI_files. figure out how it works, and then run commands to get all the test results etc. Lets make a plan here. I want you to do the full e2e test, and set up the tests for other OS too, which I can then manually run. Ask me questions for clarification. KISS, DRY, YAGNI. Be honest, no sugar coating. First do a full audit of both the current and the legacy AddaxAI, and then lets make a plan together. 
- [x] Bump addaxai-base from cu118 to cu128 so RTX 50-series (sm_120, Blackwell) gets native kernels instead of the 4-5 min PTX JIT fallback. Suggested pins: torch==2.8.0+cu128, torchvision==0.23.0+cu128, --extra-index-url https://download.pytorch.org/whl/cu128. Both windows and linux YAMLs. Adds ~700 MB to the install but fixes the GPU warning reported at https://forum.addaxai.com/t/model-warning-on-running-with-gpu/202. Requires NVIDIA driver >= 555.x, mention in the beta-tester readme. 
- [x] Bump the pytorch env from Python 3.8 to 3.11 (or higher, what do you recommend, whats the latest stable python, what is best to bump it too?) (3.8 is EOL since Oct 2024 and recent torch wheels are starting to drop py38 builds). Also bump torch alongside the python jump. SpeciesNet-fine-tuned classifiers (.pt files with pickled onnx2torch operator classes) need a smoke test after the bump: load NAM-ADS-v1 or similar and confirm torch.load() succeeds across the major version jump. 
- [x] If we have bumped pytorch env to 3.11, we can implement https://github.com/MNHN-OFVI/DeepForestVisionV2. See email from Hugo at Tue, Jun 16, 4:01 PM for more info. 
- [x] Test all models on all OS'ses. (Run python scripts/test_models.py on Windows and Ubuntu.)




- [?] [ALREADY IMPLEMENTED - NEEDS TESTING] ALLOW FULL IMAGE CLS MODELs TOO (AHDRIFT-v1)
- [ ] Investigate WHY full-image classifiers cannot process videos at all. `detection_worker.py` refuses the whole folder when a full-image classifier (e.g. AHDRIFT-v1) is selected and the folder contains any video ("cannot process videos. Folder contains N video file(s); use a folder with only images"). A full-image classifier labels the whole frame, and a video's best frame IS a frame, so it is not obvious why it cannot be classified like an image. If the limitation is real, the error should say why; if it is historical, the refusal can go. Note: setting "Media to analyse" to "Only images" now sidesteps the refusal (no videos reach the check), so this is no longer a dead end, just unexplained.
- [?] [ALREADY IMPLEMENTED - NEEDS TESTING] ALLOW CANADA sex-age-classes GRANT MODEL 
- [ ] ask to Saul to add AddaxAI.exe --timelapse "<folder>" to Timelapse's command list as the long-term path.
- [ ] If everything works and all models are verified, please double check if there are any stale environment.ymls that are never used by any of the models. If so, remove them. 
- [ ] Any other non used imports or requirements in the environments YMLS? 





- [ ] ADD Caras model

- [ ] On installing an env or modal (preparing model) the modal can close when clicked outside. That is not what you want. It should close only when clicked "Close". The cross on the top right should also not be there. 


## Future stuff
- [ ] MULTI LANGUAGE SUPPORT
- [ ] DEPTH ESTIMATION
- [ ] TRAIN FEATURE - https://agentmorris.github.io/speciesnet-fine-tuning/ - must do something to have it accept bounding boxes with verified labels from projects mode (should allow data from multiple projects, so where does thi live? Not inside a single project, right? perhaps better in the projects overview page?). The tutorial also focusses on SpeciesNet only, but for marine workflows it would be good if we could have it accept generic pretrained models too. If we could adjust the script so it can train a pretrained Efficientnet model too, that would be great!
- [ ] Add variant rank. See future-plans/add-variant-rank.md
- [ ] POSTPROCESS BATCH RESULTS MEGADETECTOR
- [ ] DOCUMENTATION (see items below)
- [ ] REPEAT DETECTION ELIMINATION
- [ ] WLIDBOOKS INTEGRATION
- [ ] EARTHRANGER/GUNDI integration
- [ ] CONSUME restrict_to_taxa_list - Get the most out of “vanilla SpeciesNet”. In particular, many issues that might make fine-tuning seem like a good option can be resolved by just remapping SpeciesNet’s outputs differently, instead of using the standard SpeciesNet geofence (a list of taxa that are allowed in each country (or US state)). You can do this with the restrict_to_taxa_list function, which takes a list of SpeciesNet taxa in a .csv file, and maps them to whatever labels you want. In addition to mapping one species to another, you could, for example, map all birds that aren’t otherwise mapped to an “other bird” label. This skill or this app can help you make those .csv files. More generally, I have some “pro tips” for getting the most out of MegaDetector and SpeciesNet here.
            https://megadetector.readthedocs.io/en/latest/postprocessing.html#megadetector.postprocessing.classification_postprocessing.restrict_to_taxa_list
            https://github.com/agentmorris/agentmorrispublic/blob/main/skills/speciesnet-taxonomy-mapping/SKILL.md
            http://dmorris.net/speciesnet-taxonomy-mapper
            http://lila.science/speciesnet-pro-tips

## Documentation
- [ ] Make a tutorial on how to move data between computers. "The difficulty is that AddaxAI uses three data sources, and all are required. The raw images and videos (to show you while doing verification)The internal JSON files hidden in the processed folders (to reprocess after settings are changed)The internal AddaxAI database (stores all detections, verification statuses, etc)If we want to move everything to a new computer, we must move all three of these components. Luckily, components 1 and 2 are together, so if you have the images on an external drive, you can just plug it into a new computer. Then, you also need to move the DB, which means you must back it up manually, move the DB file to the new computer, and then restore from the there. "
- [ ] in text and in video tutorials - proposed workflow: record MP4 locally with ScreenKite, host on HF (tutorial-videos repo), stream in-app, bundle nothing
- [ ] also all the models avaiulable with species etc 
- [ ] also include the fallback date reader from filename (...addaxai-YYYYMMDD-HHMMSS.ext)

## Nice to haves
- [ ] If there are new detections there is a mechanism that lets the user re-embed them so they are added later in the process. Can we do that for classification too? Imagine this scenario: User runs folder run at all the defaults with csl model SpeciesNet. IIn labels page it sets the det thresh to 0.01-1 to include all detections. there is an option to add embeddings for the ones that did not get embeddings right away, but the ones without a cls label will always remain null ( or effetively "animal"). How difficult would it be to have a similar mechanism as the embedding, but then for classifications? Is it hard to do? Can we reuse the embedding logic? Dont bother if its hard to do, since its a pretty nice use case, so KISS. ALso DRY YAGNI. Be honest, no sugar coating. 
- [ ] Take a look at https://huggingface.co/conservationxlabs/miewid-msv3. IS that a better embedding model than DINOv2? 
- [ ] RELABEL PICKER RESPECTS SPECIES SELECTION. The relabel spotlight (R on the labels page) is built from the model's full `taxonomy.all_classes` in `frontend/src/hooks/useLabelOptions.ts`, with no `excluded_classes` filter, so it offers every model class (for SpeciesNet ~3000) even when the user curated a short species list. Don't hard-restrict it (a human override is a stronger signal than the AI, and may legitimately need an excluded species, e.g. a vagrant or a slightly-wrong geofence), but for large-vocabulary models prioritise the selected species at the top of the picker with the rest still reachable below. Needs the project's `excluded_classes` passed into the picker for sorting/sectioning, so not a one-liner. Nice-to-have for curated / SpeciesNet workflows.
- [ ] APT REPO FOR LINUX UPDATES - host a small signed apt repository so Linux users add it once and then get AddaxAI updates through Ubuntu's normal Software Updater. Currently a new deb opened in the App Center shows a greyed-out "Installed" button with no update path (App Center limitation for sideloaded debs), so BETA.md tells users to run `sudo apt install ./AddaxAI-amd64.deb` by hand. An apt repo also removes the "Unknown publisher / potentially unsafe" warning on the install page. Only worth it if Linux uptake grows beyond the current handful of users.
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


# TODO

## Priority 1
- [ ] CHECK IF THIS IS RESOLVED: It says "You have unsaved changes" every time i open the settings page. As a test: click "reset changes" the "You have unsaved changes" is not shown anymore, no apparent changes visible. Move different page, move back to settings, it shows "You have unsaved changes" again. Bug. Investigate. The thing is, we investigated previous already but could not find anything. We could not reproduce it, but now its back, and perhaps it has to do with the electron build as opposed to the localhost dev version. That the bug is only in electron, but not in the dev version. Could that be? 

- [ ] THe output preview that we worked on the scroll area before. Can we have it expand ot the available page height? So the scroll inside should be like it is now, but the card height is limited by the page hieght. Does that make sense? And is that easy? Then we accomodate large and small screens. 

- [ ] LINUX DEB PACKAGE - decision (2026-07-05): ship the Linux beta as a .deb instead of the AppImage. Goal: zero terminal for the user. Double-click the .deb, install via the software center, launch AddaxAI from the app menu like any other app. Background: the AppImage aborts on launch on Ubuntu 23.10 and newer because AppArmor restricts unprivileged user namespaces and Electron's SUID chrome-sandbox fallback cannot work on a nosuid FUSE mount (confirmed on Ubuntu 26.04 in VirtualBox; --no-sandbox works but drops the sandbox). The deb solves both the crash and the chmod +x UX in one go. Implementation sketch:
    - add "deb" to the linux targets in electron/package.json (electron-builder generates the desktop entry and icons, so it appears in the app menu)
    - add a deb afterInstall script that installs an AppArmor profile granting userns (the standard Ubuntu 24.04+ electron fix) and runs apparmor_parser; afterRemove cleans it up
    - keep the AppImage as a secondary download for non-deb distros, with the --no-sandbox relaunch fallback in the main process so it at least starts
    - CI: build-electron.yml linux job already runs --linux, so it picks up the new target; check the artifact name pattern
    - update BETA.md with the Linux download + install steps once it works
    - test on the clean-install VirtualBox snapshot: double-click install, menu launch, model download, folder run, uninstall

- [ ] SHould we add an option to opt out? Just to be clear: opting out is canceling, not doing it in the backgound. NO cleverness here. Or better yet, a lot simpler. Just add to the cpation that the user can quit the app and try it again later. Some users might want to know that it is safe to quit. 
            Updating analysis environment
            The environment is wiped and rebuilt to match this app version. This can take several minutes and cannot be cancelled. Keep the app open until it finishes.

- [ ] There is quite a lot of whitespace here ion the project create moidal info. card for no cls model. Can we make it less? Wihtout messing up the format if the bar is not there. '/Users/peter/Desktop/Screenshot 2026-07-06 at 12.29.39.png' '/Users/peter/Desktop/Screenshot 2026-07-06 at 12.29.48.png'

- [ ] Set up a scheduled GitHub Action to fetch download counts for all release assets via the GitHub API and store daily or weekly snapshots in a CSV for tracking downloads over time. Where to nsave the CSV? I dont know. What is customary here? What is best practises? 

- [ ] The very first time a user opens AddaxAI after a fresh install, it takes quite long to open. Can we do anything about that? Perhaps do a warmup open during the actual installation part? Or how would that work? And what would it save in terms of time the first time? KISS DRY YAGNI

- [ ] folder run save step. option : "Keep events together - The whole event goes to the folder of its most confident species". Is this most confident species referring to the cls confidence? Because that if set to 100% if verified, right? Or to the det conf? Worth knowing which one it is, and if that is the desired behaviour. No action point yet, just checking and see if we want to do anything with it. 


## Priority 2
- [ ] 

## Priority 3 
- [ ] would it make sense to show a list of previously analysed folders at the folder run step 1? SO that they do not have to drag and drop the folder or browse trough them, but just scroll and click... ? If so, how and where?

## After the Beta phase
- [ ] ADD ALL MODELS (Also Caras model)
- [?] [ALREADY IMPLEMENTED - NEEDS TESTING] ALLOW FULL IMAGE CLS MODELs TOO (AHDRIFT-v1)
- [?] [ALREADY IMPLEMENTED - NEEDS TESTING] ALLOW CANADA sex-age-classes GRANT MODEL 
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


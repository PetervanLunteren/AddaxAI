# TODO

## Priority 1
- [ ] 

## Priority 2
- [ ]  

## Priority 3 
- [ ] 

## After the Beta phase

- [ ] DOCUMENTATION (see items below)



- [ ] I just installed it from scratch again, and it told me I needed to setup the DINOv2 ViT-B model. This is some old stuff, and hasent been udpated yet. It used to be defaulted to DINOv2 ViT-B, but now we default to DINOv2 ViT-S. It goes well in the install, but aparently, it is still the default for new projects and new fodler runs. Is that correct? That is the wrong UX. If the user installs the app, it shjould not be greeted with a setup error for DINOv2 ViT-B while DINOv2 ViT-S is silently installed. SO make sure the defaults all point to DINOv2 ViT-S, so that the user has a smooth experience. For both folder runs, and project modes. 


## Future stuff
- [ ] MULTI LANGUAGE SUPPORT
- [ ] MARINE RESEARCH - BRUV MODE - Sharktrack and Community Fish Detector
- [ ] DEPTH ESTIMATION
- [ ] TRAIN FEATURE - https://agentmorris.github.io/speciesnet-fine-tuning/ - must do something to have it accept bounding boxes with verified labels from projects mode (should allow data from multiple projects, so where does thi live? Not inside a single project, right? perhaps better in the projects overview page?). The tutorial also focusses on SpeciesNet only, but for marine workflows it would be good if we could have it accept generic pretrained models too. If we could adjust the script so it can train a pretrained Efficientnet model too, that would be great!
- [ ] POSTPROCESS BATCH RESULTS MEGADETECTOR
- [ ] REPEAT DETECTION ELIMINATION
- [ ] WLIDBOOKS INTEGRATION
- [ ] [ALREADY IMPLEMENTED - NEEDS TESTING] ALLOW CANADA sex-age-classes GRANT MODEL (possible the same or in line with todo item: - [ ] Add variant rank. See future-plans/add-variant-rank.md)? 
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

## Nice to haves
- [ ] If there are new detections there is a mechanism that lets the user re-embed them so they are added later in the process. Can we do that for classification too? Imagine this scenario: User runs folder run at all the defaults with csl model SpeciesNet. IIn labels page it sets the det thresh to 0.01-1 to include all detections. there is an option to add embeddings for the ones that did not get embeddings right away, but the ones without a cls label will always remain null ( or effetively "animal"). How difficult would it be to have a similar mechanism as the embedding, but then for classifications? Is it hard to do? Can we reuse the embedding logic? Dont bother if its hard to do, since its a pretty nice use case, so KISS. ALso DRY YAGNI. Be honest, no sugar coating. 
- [ ] Take a look at https://huggingface.co/conservationxlabs/miewid-msv3. IS that a better embedding model than DINOv2? 
- [ ] RELABEL PICKER RESPECTS SPECIES SELECTION. The relabel spotlight (R on the labels page) is built from the model's full `taxonomy.all_classes` in `frontend/src/hooks/useLabelOptions.ts`, with no `excluded_classes` filter, so it offers every model class (for SpeciesNet ~3000) even when the user curated a short species list. Don't hard-restrict it (a human override is a stronger signal than the AI, and may legitimately need an excluded species, e.g. a vagrant or a slightly-wrong geofence), but for large-vocabulary models prioritise the selected species at the top of the picker with the rest still reachable below. Needs the project's `excluded_classes` passed into the picker for sorting/sectioning, so not a one-liner. Nice-to-have for curated / SpeciesNet workflows.
- [ ] APT REPO FOR LINUX UPDATES - host a small signed apt repository so Linux users add it once and then get AddaxAI updates through Ubuntu's normal Software Updater. Currently a new deb opened in the App Center shows a greyed-out "Installed" button with no update path (App Center limitation for sideloaded debs), so BETA.md tells users to run `sudo apt install ./AddaxAI-amd64.deb` by hand. An apt repo also removes the "Unknown publisher / potentially unsafe" warning on the install page. Only worth it if Linux uptake grows beyond the current handful of users.
- [ ] SUBSAHARA GEOFILE - Add a geolocation file for the Sub Saharan model too, like SpeciesNet, so users of the SSmodel can also prefil by country. 
- [ ] Bulk-create sites and deployments from a text file (CSV). Columns: name, lat, lon, optional habitat, deployment folder, altitude. Requested by Simon for a 34-camera survey where adding each site/deployment by hand is laborious.
- [ ] Follow Windows .lnk shortcuts when walking deployment folders. Currently Python's os.walk treats .lnk as regular files and skips them. Workaround for users: NTFS junctions (mklink /J) are followed natively. Requested by Simon for a per-camera-per-week shortcut layout.
- [ ] Ranked flat label filter (descending count) as an alternative to the hierarchical tree on Verify and Insights. Requested by Simon.
- [ ] Verification-status layer on the Map insights page (e.g. dot color = % verified, or filter to deployments still unverified). Requested by Simon for tracking field-work progress across many cameras.
- [ ] Take the heatmap for deploment timeline feature from AddaxAI Connect
- [ ] Naïve occupancy insights page. See AddaxAI-Connect. 
- [ ] Update the exports page to match the 2 col format. Check AddaxAI Connect. 



# DAN and SAULs feedback
- [x] make sure the verified / edited / corrected stuff is in the output files (JSON, CSV / XLSX). Extra bounding boxes, add verified flag (or something like "AI", "Human", etc). So the database should refelct the in the JSON, all the stuff that is in the DB should be in the JSON and CSV/XLSX. 
- [x] Rename the Verification step of the folder to something like Edit, or Change. Make sure it feels like its optional. Optionally change AI predictions. Optionally edit predictions. 
- [x] Perhaps we should hide the edit / verify / change step, and the summary step by default. Since it is optionally. Anyways, I received feedback that it felt like it was required to do something with it. That is not the case. 
- [x] make the edit / verify step not have tabs, but do this (observation / media / events) in a dropdown in the filter bar under View as or something like that. So that its clear that the user is seeing the same dataset under the different groupings. 
- [x] add view options to the events and media views too. So S / M / L to increase the number of columns one can scroll through. 
- [x] on the save stap, default to only outputting CSV and JSON. Leave the rest non selected. 
- [x] Nest blurring and visualisations under separation. Or better yet, rethink the all the options. Now we have 3 options (separate, blur, visualise) that kind of are dependent on each other. They all do something with the source data, they should somehow be grouped together. Agree? And then we have the export results option, with options to CSV, XLSX, and JSON, which are doing stuff with the results and not the source data. 
- [x] make sure the JSON takes all the relevant info form the folder run DB and puts it into the JSON, following the format of MegaDetector. See how the projects page writes its JSONs, follow that format exactly. Including the classification_descxriptions with the taxonomy information (just like results mode does it). 
- [x] Save outpout settings to localStorage on buttoin click too. Next time the user has the same settings for saving as last time. 
- [x] Investigate how it works when we get a command from Timelapse to open AddaxAI in timelpase mode with the given directory path. How does it work? Well, the timelapse mode is now depreciated (now its integrated into the folder run mode), so I want you to do two things: 1) make sure the command given from TimeLapse that previously opened addaxai-timelpase-mode with the dir filled in, now open folder run with the dir filled in. Then 2) remove the addaxai-timelpase-mode all together. We will not use it anymore. No dead code. Rmove all the code and all the links to it. It should look like it never existed. 
- [x] SHould we make display name option toggle for common name or scientific? Not all labels will have a common name (e.g. if rollup happens), but some will. So perhaps then show common if present, else scientific? how would that work? Invesitgate. Is this a simple refactor? Or a major one? Where so we store this toggle? Or should we just default to common-if-present?
- [x] In proejcts mode, if no site is selected, but the timezone is set in the settings, it shows "Sun-time mode needs at least one camera site with GPS coordinates. Assign a location to a site or switch the time axis to clock-time." Is that by design? The seetigns TZ is silently set to browser time, shihc might not be the TZ in the camera traps. We might need to think about this. Is TZ in settings all we need, or is the a lat/lon all we need? or do we need both? 
- [x] We should probabaly have a "Decline suggestion" button that ignores the cohort of suggestions. It doesnt do anything to the cohort crops, it just removes them from the suggestions. This is good for the workflow "yes, these are indeed crows -> accept. No, these are actually a bunch of different animals. I dont want to fix them all here, and I also dont want to click and relabel here. Just ignore the suggestion and send them back to the main sort so I can relabel them there-> ignore.". 
- [x] run a folder of videos
- [x] Run a folder of edge cases

- [ ] How difficult would it be to have the timestamps of the frames being offset by the first timestamp? Currently it works by having all the frames have the same timestamp of the video, right? How difficult would it be? And what do we benefit from it from a user perspective? What are the pros and cons? Is it worth the effort? 

- [ ] the filmstrip below the event show verification flags. Why? Where is that coming from? What verified it? Is that still old code? 

- [ ] On the observations card a few comments:
        - The confirm button should be the primary action, so priamry colour a t the bottom. 
        - The "+ Add a species the AI missed" can be shorter, and the button should cover the whole text. Perhaps somethjing like "+ Add observation" or "+ Add row"
        - Why cant you remove the AI proposed species row? Curently, the [X] only appears at the manually added rows. Is that by design?
        - The "AI saw X" is noise. remove it
        - do not reorder the list depending on count. If you press + on the last one, the order keep moving and you end up nclicking on the wrong +'s.
        - Do not do the inline reset button. It is annyoing since it is at the position of the inital +. perhaps better a event wide reset button that resets it to the AI state? 
        - Thwre should be a scroll if there are too many species for the screen. (do debug month interval for videos and check)

- [ ] SHould we also add sex, age, and behaviour while were at it? I think this would be a power feature. Have a look at /Users/peter/Documents/Repos/AddaxAI-Connect/ to see how its done there. This is how it looks. '/Users/peter/Desktop/Screenshot 2026-06-09 at 14.31.44.png' Can we learn something here? It also has opptions like duplicateing if you need to split for a different behaviour etc. 

- [ ] The 73 tsc -b errors should be fixed. Investigate. 

- [ ] EventCard grid chips still show species names without "×N", and the dashboard's headline verification progress card (separate from the Observations-page pill, which is already event-level) still counts files — I can switch it to event-level if you want.. 






## Priority 1
- [ ] Add different modes at the home screen. What if we frame the two workflows as "Processing workflow / One-off analysis tool" and "Management platform"? Because that is the real distiction, right? One just processes the data and lets you handle it, the other stores it and can be revisisted later and can be added to like a project. 
- [ ] Add a country dropdown for models with geofence data (speciesnet eg), I;ve received the feedback that the country dropdown was hard to find. Perhaps a country dropdown and then a small line saying something like "X of Y labels included/excluded. Click here to refine." Or something like that. Because it is a pretty powerful feature that one can refine the geofence reules themselves. What do you think? How would that look? Give me a few options as previews, so I can visually see, we can refine, and then I can choose. We'll need to replace all the label slection pickers that are visible (fodler mode step 1, create project modal, project settings, more?). 
- [ ] DO not block data without datetime in projects mode. Just write NA, and they are excluded from the  insights etc that need times.
- [ ] Allow reproceesing of a deployment. CUrrently it says "You can't, it already in the DB, first delete". But erhn make it "Please note that this one is already in the DB, if you process it again, you'll overwrite the previous preduicxtions, including the verifications etc. Are you sure?"
- [ ] Do Sauls feedback
- [ ] Add variant rank. See future-plans/add-variant-rank.md
- [ ] ADD ALL MODELS (Also Caras model, and make TODO on the cls-pipeline to fix this properly there too)
- [?] ALLOW FULL IMAGE CLS MODELs TOO (AHDRIFT-v1)
- [?] ALLOW CANADA sex-age-classes GRANT MODEL 
- [ ] ask to Saul to add AddaxAI.exe --timelapse "<folder>" to Timelapse's command list as the long-term path.

❯ Make the caption more user friendly: "Hard limit on how many detections one similarity sort loads. Narrowing filters first is always faster than raising the
  cap." What do users want to know? This is a limit. The higher the limit, the more memory needed and the slower it gets. Lowering the number of observations 
  with the filters is generally easier and faster. 

## Priority 2
- [ ] Make projects-mode analysis blocking too, like the folder-run flow now does. Today it runs in the background queue worker with the `DeploymentHealthToast` showing progress; user can navigate everywhere while it runs, kick off other deployments, etc. For consistency with folder-run's modal-blocking pattern (and the same "ML uses all your compute, don't do other heavy stuff simultaneously" reasoning), wrap the running deployment in the same blocking modal. Keeps the queue concept (sequential deployments), just makes the active one block the UI. Reuse the JobProgressModal from folder-run.
- [ ] Fallback to read datetimeorignal from filename as a fallback. Reach out to flavio to see how to format the read exactly. example "S1_20250222_072314.mp4", regex: the last two parts separated by "_" in the format of <whatever>_<YYYYMMDD>_<HHMMSS>.<extention>

Open a project with completed deployments → should land on Dashboard. - DOES NOT LAND ON DASHBOARD. LANDS ON PROCESS. 

## Priority 3 
- [ ] Make a tutorial on how to move data between computers. "The difficulty is that AddaxAI uses three data sources, and all are required. The raw images and videos (to show you while doing verification)The internal JSON files hidden in the processed folders (to reprocess after settings are changed)The internal AddaxAI database (stores all detections, verification statuses, etc)If we want to move everything to a new computer, we must move all three of these components. Luckily, components 1 and 2 are together, so if you have the images on an external drive, you can just plug it into a new computer. Then, you also need to move the DB, which means you must back it up manually, move the DB file to the new computer, and then restore from the there. "

## AFter the Beta phase
- [ ] If everything works and all models are verified, please double check if there are any stale environment.ymls that are never used by any of the models. If so, remove them. 
- [ ] Any other non used imports or requirements in the environments YMLS? 
- [ ] Bump addaxai-base from cu118 to cu128 so RTX 50-series (sm_120, Blackwell) gets native kernels instead of the 4-5 min PTX JIT fallback. Suggested pins: torch==2.8.0+cu128, torchvision==0.23.0+cu128, --extra-index-url https://download.pytorch.org/whl/cu128. Both windows and linux YAMLs. Adds ~700 MB to the install but fixes the GPU warning reported at https://forum.addaxai.com/t/model-warning-on-running-with-gpu/202. Requires NVIDIA driver >= 555.x, mention in the beta-tester readme. 
- [ ] Bump the pytorch env from Python 3.8 to 3.11 (3.8 is EOL since Oct 2024 and recent torch wheels are starting to drop py38 builds). Also bump torch alongside the python jump. SpeciesNet-fine-tuned classifiers (.pt files with pickled onnx2torch operator classes) need a smoke test after the bump: load NAM-ADS-v1 or similar and confirm torch.load() succeeds across the major version jump. 
- [ ] Do we want a custom minimal menu (just Reload / Force Reload / DevTools / About / Quit) with our own styling? Or keep the electron built in? We can put the hamburger menu in the electron menu row? And the bug report etc. Then we can also add video tutorials etc. Also add a Check for updates option. We can put global settings here (like common-name/scientific-name toggle, or language etc). Make it look like a mature app. What else would you recommend in the menu items, and in which order, groupings? How do other mature apps do it? And what do you recommend for this app? You can web query if you want. 
- [ ] Would it be a good idea to add a extra level for the smoothing "Very aggresive" (or something similar), that does not run the MD utils smoothing script at all, but just flattens out the entire event to a single label. We'll need to think about which label of course (the max cls conf label for all? or some kind of average label for all?). If we decide to do this, we might also want to add captions in the dropdown that try to explain the tiers off / mild / aggresive / etc. Make it a tall dropdown just like the models dropdown with captions, use the same format. What do you think? 

- [ ] explain again how the 20K limit affects the search and the pbar., What happens if there are to many above the limit? Does it just take the first N observations, or does it block? 

## Future stuff
- [ ] MULTI LANGUAGE SUPPORT
- [ ] DEPTH ESTIMATION
- [ ] POSTPROCESS BATCH RESULTS MEGADETECTOR
- [ ] DOCUMENTATION - in text and in video tutorials - also all the models avaiulable with species etc. 
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
- [ ] Update the exports page to match the 2 col format. Check Connect. 










Have a look at how the app uses and saves video frames. Report back to me how it works, how the plsitting works, how the reading works, etc. I want a full audit of how this works, as we'll talk about different scenarios of improving the frame splitting and saving. The main reason for delving into this is the following user report. 

It looks like video frames were extracted to disk inside the .addaxai folder during processing (within "video_frames"), and not deleted.  The frames are around the same size as the original videos, even at the default frame sampling rate, so I think this will be problematic.  If we crank up the frame rate at all, the frames would quickly become much larger than the original videos.  If you want to extract frames to disk, I would process each image one frame at a time, then delete the frames.  But are you sure you want to extract frames to disk?  This used to be how I processed videos, but neither run_md_and_speciesnet nor process_video require this.  Best case, you become dependent on (possibly slow) hard drive write speeds, worst case, you also add a lot of storage overhead.

SO long story short, can we somehow avoid saving videos to disk? And hoq would that work? What would be affected by it? Can we run analysis on frames on the fly? Or on batches of frames? Or batches of frames per video? The real issue is that if somebody will analyse a backlog of 1TB videos, this current method will be very destructive. What do you think? How would analysis look if we not save the frames? That means extracting the frames for the detection, classification, and embedding phases. And how would the UI work? Perhaps save thumbnails to disk for UI quickness, and extract on the fly for verification of high quality frames? IDK. Anyway, just some thoughts. I would like to hear the best appraoches from you. 

Instructions:
* Claude code will review your output once you are done, so make sure you exceed his expectations
* do not sugar coat, be honest and clear
* Switch to plan mode, I want this task to be done with "plan mode on"
* Read all MD file in root to get a understanding of the project. 
* If something is unclear at any point, stop and ask before continuing.
* Prioritize simplicity and clarity over perfection. The code must be clean, easy to read, and understandable for collaborators. Avoid unnecessary complexity.
* I'm not in a rush. Please be precise and do the task thoroughly. 
* Please ask me any question for clarification. I would rather that you ask too many questions than assume certain details. 
* Ask me clarifying questions before beginning. Based on the conventions set out in CONVENTIONS.md and your knowledge, give your recommended solution to each questions you ask me. The minimum number of questions to ask me is 3

Workflow:
* Based on my answers, suggest a few general approaches. These should range from simple solutions to more sophisticated alternatives, with clear trade-offs for each. For every approach, explain:
   - Complexity (difficulty, dependencies, maintainability)
   - Readability (clarity for collaborators)
   - Effect (impact on performance, usability, flexibility)
* Give your recommendation regarding the alternatives discribed earlier, with a short reasoning. Be short and concise. Key words if possible.
* After I select an approach, draft a detailed plan for implementation.
* Only start working if I agree with the proposed plan.

# TODO


We probabaly want to change the inpedendece interval caption now that there are no events in folder mode run, it just feeds the smoothing and save "keep events together" right? Do we even want to call it "events" in folder mode? Or dependent batches? IDK. whats you take here on the caption and the rest of the wording in folder run? Now that I'm reading the caption of the smoothing, it also refers to events. It moight be confusing not to call it events. And it also shows what the user does if setting the independence interval. Perhaps just a tiny addition to the caption of the indopendence interval with "also determines the smoothing and .... (fill in what it feeds)". 

"Also used as the starting point for your next run." is not needed. Users dont need to kow that some of this also is used by the inference time. Perhaps this will suffice: "Applies to the results below without re-running the models. "

Do we need to rename it from Ananlsysis settings to something else? Perhaps Analysis preferences? IDK probabaly not that. I'm just thjinking that thjere are many settings here.... at step 1, advanced at step 1, analysis settings at step 2, settings inside projects, settings in the save steps itself etc. What do yoiu think?

If I reprocess the labels pane should auto update. Now it needs a hard refesh to see the new labels. 

In projects mode, if you reprocess, it shows you a summary of how the DB was changed. Might be good to do here too. You can use the exact same code and toasts, modals etc for this. Use shared helpers. 

"Where everything gets written. Defaults to the folder you analysed; your originals are never overwritten." -> Where everything gets written. Defaults to the folder you analysed. Your originals are never overwritten.

"Your media sorted into folders, videos as a best-frame image" -> Your media sorted into folders. Videos are written as best-frame images.

What do we think of this order? Is it logical? 
            Folder structure
            How the copies are organised

            Nested by taxonomy
            Keep events together
            The whole event goes to the folder of its most confident species

            Folder order
            Whether species or your original folders sit on top

            Species folder first
            Labels
            Which labels to copy and visualise

            All labels
            Confidence
            Detections below this score are left out of the copies. The data files always include everything.
            0.20
            Draw detection boxes
            Boxes and labels on each file

            Blur people and vehicles
            People and vehicles blurred on each file

            Also copy empty files
            Images and videos with no animals, people, or vehicles


"Detections below this score are left out of the copies. The data files always include everything." -> Detections below this score are left out

Should we make the actions all in the same row card format? With a title + caption? Now its a mix. '/Users/peter/Desktop/Screenshot 2026-07-07 at 13.30.03.png' Perhaps a bit more following this format? '/Users/peter/Desktop/Screenshot 2026-07-07 at 13.36.01.png'

in the outputs CSVs, there are redundant columns like eventID, deploymentID etc. These are not needed in folder run. Investiogate what is currently there, and what can be hidden for folder mode runs. Make sure the porojects exports remain exactly the same, so you only edit the folder run exports. BTW, the full CSV/XSLX exports are still present on projects mode exports, right? With deployments, events, trap days, etc. ? 




## Priority 1
- [ ] ORPHANED BACKEND PROCESS - the packaged app can leave its backend running after quit, and the next launch then silently talks to the stale backend. Observed on Peters mac (2026-07-07): /Applications/AddaxAI.app backend orphaned on port 8000, PPID 1. Three causes in electron/src/main.ts: (1) stopBackend() sends SIGTERM blind, never verifies exit, no SIGKILL fallback; uvicorn graceful shutdown can hang forever on open connections (an open browser tab is enough). (2) the relaunch path uses app.exit(0), which skips before-quit/will-quit entirely, so stopBackend never runs on relaunch. (3) force-quit/crash. Worst consequence: after an update, the new app fails to bind 8000 but the health check gets an answer from the STALE old-version backend and uses it -> new frontend on old backend, schema drift, unexplainable bugs. Fix ideas: SIGTERM then wait then SIGKILL; call stopBackend explicitly before app.exit in the relaunch path; on startup, verify the /health version matches the app version and kill/refuse a stale backend.
- [ ] projects menu sidebar: the "Per class performance" text is too long. Propose shorter alternative. Perhaps class performace? 
- [ ] The prcess for label verification and event comfirmation can be quick. That is what we designed it for, so it s working. That is great. But mistakes can happen. NOw the user needst to change a filter to see verified ones, then relabel them. Can we make this faster? Like an UNDO button or so? Just invetigate and report with estimated effort. KISS DRY YAGNI. Can we make it undo unlimited? Just like CNTR+Z on windows? maybe only the labels is enough for the undo, as counts confirmation you can go back to the previous one with the arrows right? 
- [ ] If ordering by event, does it order the events of a single camera first? how does it order the events? how does it currently work and how should it work? we want the most dependent detections grouped/ordered. what is possible given the contraints of folder runs with agnostic data.
- [ ] when verifying labels, it often shows a taost saying it verified them succesfully, but the user is do this for hours on end.... and it covers the floating bar wich is annoying... what do do about this? Arent toasts for infrequent things? What is best here? in terms of UX UI
- [ ] SHould we name the folder run buttons [back] [continue] to [previous step] and [next step]?
- [ ] SHould we make the folder run steps clickable? now they are only clickable if you have visited it in that run already. But why not just have users click step 5 directly after processing... ?
- [ ] in the counts confirmation modal, would it make sense to add an option to edit the label of an already present species row? for instance, sometimes you'll get an event with a deer in a garden with a bucnh of chickens. It will suggest: deer x1, and bird x5. But the user wants deer x1, and chicken x5. Now he has to remove the biurd row and add a new one, choose chicken, and add 5 counts. If the label can change, that will be much faster. It doesnt have to be a button (as little visual clutter as possible), perhaps just by clickin on the word "bird"?

- [ ] The save step pbar is like this below. But the separating files takes quite long. Why not show progress per image? Then the user gets better feedback. perhaps pbars for all stages? or just stage + percentage + ETA? or something like that. It doesnt need to be beatiful, just practical. Also, what do these stages actually mean? "Separating files" and "Writing annotated copies"?

                Saving outputs
                Separating files (0 / 5)

                Separating files
                Writing annotated copies
                Writing recognition JSON
                Writing CSV
                Writing run README

- [ ] The save step output preview is now this below. Arguably a bit too much. You dont have to visualise everything, just enough to show the user how the folders will be grouped. What do you thnk? add a cut off point somewhere? if so, how and what is good? Also, what happens if the folder name is super duper long? Lets say the species name is "superduperlongspeicesnamewithsomeextracharactersjusttotestwithlongwords"? how would that look in the preview ciurrently? and do we want to do anything with that?

                Output preview
                What the run will write into your output folder

                AddaxAI-output
                ├─ 
                mammalia
                │  ├─ 
                carnivora
                │  │  ├─ 
                canidae
                │  │  │  ├─ 
                canis
                │  │  │  │  ├─ 
                coyote
                │  │  │  │  │  ├─ 
                Pocono_North
                123
                │  │  │  │  │  ├─ 
                Ricketts_Glen
                52
                │  │  │  │  │  ├─ 
                Allegheny_National_North
                49
                │  │  │  │  │  └─ 
                …
                │  │  │  │  └─ 
                domestic_dog
                │  │  │  │     ├─ 
                Kinzua
                48
                │  │  │  │     ├─ 
                Rocky_Spring
                37
                │  │  │  │     ├─ 
                Maple_Hollow
                31
                │  │  │  │     └─ 
                …
                │  │  │  ├─ 
                vulpes
                │  │  │  │  └─ 
                red_fox
                │  │  │  │     ├─ 
                Worlds_End
                56
                │  │  │  │     ├─ 
                Cold_Gap
                34
                │  │  │  │     ├─ 
                Beaver_Marsh
                26
                │  │  │  │     └─ 
                …
                │  │  │  ├─ 
                Tuscarora_North
                133
                │  │  │  └─ 
                …
                │  │  ├─ 
                felidae
                │  │  │  ├─ 
                felis
                │  │  │  │  └─ 
                domestic_cat
                │  │  │  │     ├─ 
                Pocono_South
                120
                │  │  │  │     ├─ 
                Hyner_Run
                48
                │  │  │  │     ├─ 
                Eagle_Mill
                23
                │  │  │  │     └─ 
                …
                │  │  │  ├─ 
                lynx
                │  │  │  │  └─ 
                bobcat
                │  │  │  │     ├─ 
                Susquehannock
                49
                │  │  │  │     ├─ 
                Slippery_Ridge
                21
                │  │  │  │     ├─ 
                Cherry_Valley
                15
                │  │  │  │     └─ 
                …
                │  │  │  └─ 
                Sugar_Valley
                1
                │  │  ├─ 
                ursidae
                │  │  │  └─ 
                ursus
                │  │  │     └─ 
                american_black_bear
                │  │  │        ├─ 
                Tioga_State_Forest
                79
                │  │  │        ├─ 
                Laurel_Highlands
                41
                │  │  │        ├─ 
                Bedford_State_Forest
                40
                │  │  │        └─ 
                …
                │  │  └─ 
                …
                │  ├─ 
                rodentia
                │  │  ├─ 
                sciuridae
                │  │  │  ├─ 
                Allegheny_North
                179
                │  │  │  ├─ 
                marmota
                │  │  │  │  └─ 
                woodchuck
                │  │  │  │     ├─ 
                Otter_Cove
                36
                │  │  │  │     ├─ 
                New_Gap
                29
                │  │  │  │     ├─ 
                Slate_Mountain
                29
                │  │  │  │     └─ 
                …
                │  │  │  ├─ 
                Tuscarora
                142
                │  │  │  └─ 
                …
                │  │  └─ 
                Otter_Hill
                2
                │  ├─ 
                didelphimorphia
                │  │  └─ 
                didelphidae
                │  │     └─ 
                didelphis
                │  │        └─ 
                virginia_opossum
                │  │           ├─ 
                Allegheny_Front
                288
                │  │           ├─ 
                Allegheny_South
                158
                │  │           ├─ 
                Gallitzin
                43
                │  │           └─ 
                …
                │  └─ 
                …
                ├─ 
                aves
                │  ├─ 
                galliformes
                │  │  └─ 
                phasianidae
                │  │     ├─ 
                gallus
                │  │     │  ├─ 
                domestic_chicken
                │  │     │  │  ├─ 
                Moshannon
                68
                │  │     │  │  ├─ 
                Buchanan_State_Forest
                42
                │  │     │  │  ├─ 
                Tioga_North
                35
                │  │     │  │  └─ 
                …
                │  │     │  ├─ 
                Clear_Bend
                18
                │  │     │  ├─ 
                Laurel_Spring
                18
                │  │     │  └─ 
                …
                │  │     ├─ 
                meleagris
                │  │     │  └─ 
                wild_turkey
                │  │     │     ├─ 
                Michaux
                55
                │  │     │     ├─ 
                Tionesta
                21
                │  │     │     ├─ 
                Crow_Cove
                20
                │  │     │     └─ 
                …
                │  │     ├─ 
                Old_Valley
                6
                │  │     └─ 
                …
                │  ├─ 
                passeriformes
                │  │  ├─ 
                corvidae
                │  │  │  ├─ 
                corvus
                │  │  │  │  ├─ 
                american_crow
                │  │  │  │  │  ├─ 
                Ohiopyle
                51
                │  │  │  │  │  ├─ 
                Birch_Marsh
                37
                │  │  │  │  │  ├─ 
                Oak_Mill
                35
                │  │  │  │  │  └─ 
                …
                │  │  │  │  ├─ 
                Crow_Bend
                12
                │  │  │  │  ├─ 
                Otter_Creek
                9
                │  │  │  │  └─ 
                …
                │  │  │  └─ 
                Iron_Ridge
                5
                │  │  └─ 
                turdidae
                │  │     └─ 
                turdus
                │  │        └─ 
                american_robin
                │  │           ├─ 
                Sproul_State_Forest
                14
                │  │           └─ 
                Old_Branch
                8
                │  ├─ 
                Sproul_State_Forest
                70
                │  └─ 
                …
                ├─ 
                vehicle
                │  ├─ 
                Wolf_Hill
                35
                │  ├─ 
                Bear_Glen
                27
                │  ├─ 
                Coal_Branch
                12
                │  └─ 
                …
                ├─ 
                blank
                │  ├─ 
                Crow_Bottom
                8
                │  ├─ 
                Bald_Eagle
                6
                │  ├─ 
                Allegheny_North
                4
                │  └─ 
                …
                ├─ 
                other
                │  ├─ 
                unknown
                │  │  ├─ 
                Mossy_Ridge
                10
                │  │  ├─ 
                Tioga_State_Forest
                6
                │  │  └─ 
                Sugar_Hollow
                1
                │  └─ 
                fictional_species
                │     ├─ 
                Cherry_Cove
                4
                │     └─ 
                Michaux
                1
                ├─ 
                person
                │  ├─ 
                Buck_Branch
                3
                │  └─ 
                Bear_Glen
                1
                ├─ 
                bird
                │  ├─ 
                Sproul_State_Forest
                1
                │  └─ 
                Tumbling_Branch
                1
                ├─ 
                deployments.csv
                ├─ 
                files.csv
                ├─ 
                detections.csv
                ├─ 
                counts.csv
                ├─ 
                recognitions.json
                └─ 
                summary.txt
                6,465 source files → 6,465 written

                ~2.6 GB



- [ ] It says "You have unsaved changes" every time i open the settings page. As a test: click "reset changes" the "You have unsaved changes" is not shown anymore, no apparent changes visible. Move different page, move back to settings, it shows "You have unsaved changes" again. Bug. Investigate. The thing is, we investigated previous already but could not find anything. We could not reproduce it, but now its back, and perhaps it has to do with the electron build as opposed to the localhost dev version. That the bug is only in electron, but not in the dev version. Could that be? 

- [ ] "Initial setup - The AI models and their environment need to be installed before AddaxAI can analyse images. This is a one-time download and can take 10 to 30 minutes depending on your internet connection." -> " ... can analyse images." - its not only about analysing images, also videos. Perhaps something more generic like "before AddaxAI works" or something like that. Propose a few suggestions. 

- [ ] If a user clicks the menu item help > "export diagnotics report" there are two toasts: '/Users/peter/Desktop/Screenshot 2026-07-06 at 13.21.31.png'

- [ ] "Analyzing - AddaxAI is analysing your files." the trailing dot is not consistent with the rest of the captions, right? Please check. 

- [ ] LINUX DEB PACKAGE - decision (2026-07-05): ship the Linux beta as a .deb instead of the AppImage. Goal: zero terminal for the user. Double-click the .deb, install via the software center, launch AddaxAI from the app menu like any other app. Background: the AppImage aborts on launch on Ubuntu 23.10 and newer because AppArmor restricts unprivileged user namespaces and Electron's SUID chrome-sandbox fallback cannot work on a nosuid FUSE mount (confirmed on Ubuntu 26.04 in VirtualBox; --no-sandbox works but drops the sandbox). The deb solves both the crash and the chmod +x UX in one go. Implementation sketch:
    - add "deb" to the linux targets in electron/package.json (electron-builder generates the desktop entry and icons, so it appears in the app menu)
    - add a deb afterInstall script that installs an AppArmor profile granting userns (the standard Ubuntu 24.04+ electron fix) and runs apparmor_parser; afterRemove cleans it up
    - keep the AppImage as a secondary download for non-deb distros, with the --no-sandbox relaunch fallback in the main process so it at least starts
    - CI: build-electron.yml linux job already runs --linux, so it picks up the new target; check the artifact name pattern
    - update BETA.md with the Linux download + install steps once it works
    - test on the clean-install VirtualBox snapshot: double-click install, menu launch, model download, folder run, uninstall


- [ ] There is a bug in the new label addition slideout. At least on linux, other OS not tested. Reproduce: select crops in labels step, Relabel, type "new species", click add, add new label slidout appears, focus on the GBIF lookup text field, sometimes on first click it disapears, sometimes you have to click a few times, sometimes it doesnt at all, but it is a bug that keeps the slidoue from closing without any label being added. INvestigate.

- [ ] new label slidout '/Users/peter/Desktop/Screenshot 2026-07-06 at 14.37.33.png'. We shoduld probabaly make it ocnsistent with the rest of the app and dont put all the text in the main caption, but do the relevant things under its relevant widgets.... that keeps it managemenble. Make sure you use the shared helper for the widget cpations (same font, size, and indetation.)

- [ ] The add deployment to queue form on the projects process page, do we want to hide the metadata like description, tags, etc by default? What do you thbink? I think not a lot of users will use that. The folder and the site is important, so alsways show that, but we can hide the rest under a collapsable, what do you think? Then save the collpsabel state to LocalStorage so it is hidden / open next time too. Matches the users workflow. INvestigate and report the effort.  

- [ ] Add a back arrow to home button on the folder run view and the projects overview page, exactly like the Baout page. Or better yet, make the current logo go back to home (as it already does currently), but make it more aparent. How? IDK. What do you suggest. Hover text, perhaps. Hover arrow appears icon? hover makes it a home icon? Perhaps. wouildnt that look modern? SOmething like that. What would you suggest? be honest, no sugar coating. Whats your opinion on this from a stadpoint of UX UI. KISS DRY YAGNI. How do other wellknown applications do it? You can webquery if you want. 

- [ ] fodler run save step. it always automatically saves summary.txt. possibly confusing for people who just want one thing. Should we put this under a checkbox too? If so, which title+captipon and where in the page? 

- [ ] ABout page.... should we add the logo there too in the header to make it consistent with the other pages folder run and projects view? We can keep the ABout and the tag caption as is.  Also, should we get rid of the large logo in the page contents? I think the users know the logo by now.... what do you think? 

- [ ] ABout page.... it still references "Click the (i) button next to a model " but that is not true anymore. Its the "model info" caption below the dropdown that they need to click. Update this. Are there more stale references? 

- [ ] lables page/step - on events sorting, when pressing 'E' shotcut, it should auto scroll/focus on the first crop of the new selection. Now you have to scroll upwards every time your lower. 

- [ ] The "Restoree database from backup" option modal should be made more user friendly. There are auto backups and manual backups (more flavours?) in the app dir, and then also manual backups in user difined dirs. The ones in the auto app dir shouod show as human friendly list of cards with datetimes, the flavour, etc. OR the user can select a backup file from a custom location. That is good UX UI. NOw the user needs to find it himself every time... and figure out how the naming works... 

- [ ] SHould we add an option to opt out? Just to be clear: opting out is canceling, not doing it in the backgound. NO cleverness here. Or better yet, a lot simpler. Just add to the cpation that the user can quit the app and try it again later. Some users might want to know that it is safe to quit. 
            Updating analysis environment
            The environment is wiped and rebuilt to match this app version. This can take several minutes and cannot be cancelled. Keep the app open until it finishes.

- [ ] There is quite a lot of whitespace here ion the project create moidal info. card for no cls model. Can we make it less? Wihtout messing up the format if the bar is not there. '/Users/peter/Desktop/Screenshot 2026-07-06 at 12.29.39.png' '/Users/peter/Desktop/Screenshot 2026-07-06 at 12.29.48.png'

- [ ] Set up a scheduled GitHub Action to fetch download counts for all release assets via the GitHub API and store daily or weekly snapshots in a CSV for tracking downloads over time.

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

## Documentation
- [ ] Make a tutorial on how to move data between computers. "The difficulty is that AddaxAI uses three data sources, and all are required. The raw images and videos (to show you while doing verification)The internal JSON files hidden in the processed folders (to reprocess after settings are changed)The internal AddaxAI database (stores all detections, verification statuses, etc)If we want to move everything to a new computer, we must move all three of these components. Luckily, components 1 and 2 are together, so if you have the images on an external drive, you can just plug it into a new computer. Then, you also need to move the DB, which means you must back it up manually, move the DB file to the new computer, and then restore from the there. "
- [ ] in text and in video tutorials - proposed workflow: record MP4 locally with ScreenKite, host on HF (tutorial-videos repo), stream in-app, bundle nothing
- [ ] also all the models avaiulable with species etc 
- [ ] also include the fallback date reader from filename (...addaxai-YYYYMMDD-HHMMSS.ext)

## Nice to haves
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


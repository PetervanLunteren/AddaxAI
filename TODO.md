## Priority 1
- [ ] TEST replace-representatives-with-maxn-frames -> Now that we have implemented the MaxN mechanism, should we rethink our "representative" frame selection in the events verification? If we have two species detected in a single event, we have two MaxNs (one for each species). It would make sense to verify both of them, as that influences the counts. The rest do not influence the counts since they are not MaxN anyway. Agree? Maybe we can rename the representative frame to "MaxN", which is understandable by the users as it is a well known concept in ecology. What do you think? So the seleciton is not based on the confidence, size, sharpness as it is done now, but only on the maxNs of the labels above the project.detection.confidence. That means an event can have several "MaxN" frames. How would this affect the current project? What needs to be done? What am I forgetting? Are there any complications? What do you think? Do a full audit. 


- [ ] (SEE DANS EMAIL) bug: when doing an analysis, the 'image classification' pbar goes from 0 to 100 without showing any stats like the other pbars. Then the 'finalizing...' part takes long. I get the sense that eveything happens in the 'finalizing...' phase. Could that be true? Investigate. 

- [ ] INVESTIGATE REFACTOR TO RUN SPECIESNET AS A NORMAL CLASSIFACTION MODEL - at the moment SpeciesNet uses its own inference code, whilest the other classification models all share their inference code. That seems like extra complexity. What if we just run SpeciesNet as a 'normal' clasisfication model like all the others? That save a lot of conplexity and if/else statements. Do a full audit on how this would affect the current code base, what needs to be changed and what features would not work then. What are the things that are hard, what pros and cons, etc. I want a full report and everything thought of. I know the current way of running SpeciesNet is by using its internal country + state geofencing, but can we mimick that ourselves by just reading the SpeciesNet sepecific country data and then allowing users to select / deselect labels just like any other classifciation model does? I know this is a great refactor, but I believe it should be thoroughly investigated, since it will make our lives a lot easier in the end. 







## Priority 2
- [ ] dashboard verification vard, explenation text "Event representatives are one file per event, used for quick review." explain a bit more how that representative is chosen. See event verification guide for more info. 

- [ ] If we do taxonomic rollup, we might get to taxa without common names or model-class-names like "cow" and "equid". What happens then? What do we show the user in the chips and in the UI? Investigate. I want to know the current way of dealing with that and all its fallbacks. 


## Priority 3
- [ ] REMOVE LEGACY ML PIPELINE CODE - Several legacy abstractions have been superseded by the direct phase handling in `_process_batch_job()` in `backend/app/workers/detection_worker.py`. The batch path (which the UI always uses) calls standalone functions directly, while these legacy classes duplicate the same work. Removal involves three pieces: (1) `JSONBasedMLPipeline` class in `backend/app/ml/json_pipeline.py` (lines ~42-710): has its own `_load_to_database()`, `_run_detection()`, `_run_classification()`, and `process_deployment()`. Only reachable through the non-batch branch in `process_deployment_analysis()` (detection_worker.py ~line 766), which the frontend never triggers. Delete the class, the non-batch branch, and its import. (2) `MLPipeline` class in `backend/app/ml/pipeline.py`: completely dead code, never imported or called by anything. It uses the `detect()` method on MegaDetector which is also only called from this dead class. Delete the entire file. (3) The `detect()` method in `backend/app/ml/inference/megadetector.py` (lines ~95-210): only called by the dead `MLPipeline` class. The active code path uses `detect_to_json()` instead. Delete the method. Together this is roughly 500 lines of duplicated or dead code.


## Features
- [ ] TIME OFFSET - add a feature that allows datetime offset. This should happen at the "new deployment" options. Perhaps something that says "your data spans X days/weeks, etc. " Click here to see the burned in pixel dates (show a few images / frames) and show the extracted datetime next to it. Then users can add an offset to all data in the deployment. Add fast options to switch from AM to PM etc. +12:00 and -12:00. 
- [ ] METADATA MANAGEMENT - add pages for deployments and sites, where they are all visible in a table format with filter options that make sense for the data. Each row is a unit (deployment or site) and the user can filter, sort, view, and then have actions as a three dot. The actions three dot will be "edit" for now only, where the user can edit the name of the unit, the time, etc. These are not defined yet, but need to be defined in this plan. The idea is that it offers a page where users have more room to customise their metadata, and edit flexibly. We will add actions to the three dot later on. 
- [ ] TIMELAPSE STANDALONE APP
- [ ] DEPTH ESTIMATION
- [ ] PROCESS BATCH RESULTS - https://github.com/agentmorris/MegaDetector/blob/main/megadetector/postprocessing/postprocess_batch_results.py
- [ ] BATCH PROCESSING OPTION - a completely separate option to do batch processing, where it just runs det+cls on all data recursively, and is agnostic of its contents (doestn need to know if it is a deployment, site, full project, etc.) It just runs everything at once. Users should be able to set settings before running the analysis. Then, after, it should give the user a few options, like export to CSV, XLSX, maps & graphs, separate into subfolders, etc. The bulk / management choice should be the first page users see when opening AddaxAI. 
- [ ] IN DEPTH PLOTS, have a header in the menu "plots", and add page wide full plots that are interactive and with a bunch of filters and settings above. The dashboard is meant as a quick glance of the project, and these are more in depth to find out patterns etc. We will be adding in depth plots as we progress with the project, but the first one is "Comparison of the activity time" (improve wording, make it short and memorable), where the user can select up to 5 labels and compare the activity. One option should be to add the suntimes to the graph (sun hours, sunset, etc.), another option would be to have the actual time on the x axis, or the UTC times (based on the suntimes), do webqueries on how other platforms do this, and what the standard is, and what is usually reported in terms of metrics. Research scientific papers etc. 
- [ ] IN DEPTH PLOT - add new in depth plot: Gantt-style timeline — one horizontal bar per deployment (or per site), showing the active period. Immediately shows gaps, overlaps, and total survey effort. Group by site with one bar per deployment within each site row. 
- [ ] MAP - check AddaxAI Connect and copy the map from there. 
- [ ] EXPORT OPTIONS - check AddaxAI Connect and copy from there. 
- [ ]





## Installer
- [ ] merge all alembic/versions/ into one. We do not have any users yet, so we can make it just the start DB. 
- [ ] make sure on app istall it installs the default models and their environments (MDv5A and DINOv2-B). 


### Add a simple country dropdown if goefencing file exists. Perhaps we can add a tab like structure like "simple" / "full control". 


### Improve the verification checkmarks in the grid view of events verification. Should we show pbars for the MaxN files and the all files? SOmething like that? Also make the Verification status filter explicit. Add options for all scenarios, one or more MaxNs verified, etc, etc. 


#### EXCLUSION METHOD

See current approach and the difference with SpeciesNet approach below. I believe its better to implement the SpeciesNet approach for all models. It doesnt feel good to promote confidences by inflating them by normalising. What do you think?  I would rather have all models follow the speciesNet exclusion method with auto rollup to a taxa that is allowed. (lion in the USA gets Felidae) for all models than the current approach where garbage labels get promoted to confident predictions (lion 97%, bird 2%, in the USA becomes bird 67%). Investigate. What would need to be changed? Make a plan.

Instructions:
* Switch to plan mode, I want this task to be done with "plan mode on"
* Read all MD file in root to get a understanding of the project. 
* If something is unclear at any point, stop and ask before continuing.
* Prioritize simplicity and clarity over perfection. The code must be clean, easy to read, and understandable for collaborators. Avoid unnecessary complexity.
* I'm not in a rush. Please be precise and do the task thoroughly. 
* Please ask me any question for clarification. I would rather that you ask too many questions than assume certain details. 
* Ask me clarifying questions before beginning. Based on the conventions set out in CONVENTIONS.md and your knowledge, give your recommended solution to each questions you ask me. The minimum number of questions to ask me is 10


```
  How geofence exclusion works: SpeciesNet vs AddaxAI                                                                                                                                                                                                                                                                                                                          
   
  The scenario                                                                                                                                                                                                                                                                                                                                                                 
                                                                                                                                                                                                                                                                                                                                                                             
  A camera trap in the USA captures an image. The classifier outputs:                                                                                                                                                                                                                                                                                                          
  - Lion: 90%                                           
  - Bobcat: 4%                                                                                                                                                                                                                                                                                                                                                                 
  - Felidae family (accumulated): 1%                    
  - Other classes: 5%                    
                                                                                                                                                                                                                                                                                                                                                                               
  Lion doesn't exist in the USA, so it's geofenced/excluded.
                                                                                                                                                                                                                                                                                                                                                                               
  SpeciesNet's approach: taxonomic rollup                                                                                                                                                                                                                                                                                                                                      
                                                                                                                                                                                                                                                                                                                                                                               
  SpeciesNet sees that lion is geofenced and asks: "what's the nearest allowed ancestor in the taxonomy tree?" It walks up: lion → Panthera genus → Felidae family (allowed in USA). It then accumulates all the scores from geofenced children at that family level (lion 90% + other felidae scores) and returns:                                                            
                                                        
  Result: "felidae family" at ~91% confidence                                                                                                                                                                                                                                                                                                                                  
                                                        
  The information is preserved: the model was very confident it saw a cat. The specificity is reduced (species → family), but the prediction is honest.                                                                                                                                                                                                                        
                                                        
  AddaxAI's approach: remove and renormalize                                                                                                                                                                                                                                                                                                                                   
                                                        
  AddaxAI removes lion from the classification list entirely. The remaining confidences (bobcat 4%, felidae 1%, other 5%) are renormalized to sum to 100%:                                                                                                                                                                                                                     
   
  Result: "bobcat" at ~40% confidence                                                                                                                                                                                                                                                                                                                                          
                                                        
  The bobcat was originally at 4% (the model was NOT confident it was a bobcat), but renormalization inflates it to 40%. If it crosses the 0.65 rollup threshold, AddaxAI's separate taxonomic rollup step won't catch it. The user sees a confident-looking "bobcat" that's actually garbage.                                                                                 
                                                        
  When it matters                                                                                                                                                                                                                                                                                                                                                              
                                                        
  This only matters when a high-confidence geofenced species gets excluded. In practice:                                                                                                                                                                                                                                                                                       
  - If the model says 90% lion in the USA, lion genuinely doesn't exist there, so the model is confused (maybe it saw a mountain lion and the model's best match was African lion)
  - SpeciesNet would say "felidae" (some kind of cat) which is correct                                                                                                                                                                                                                                                                                                         
  - AddaxAI would say "bobcat" at inflated confidence, which might be wrong
                                                                                                                                                                                                                                                                                                                                                                               
  When it doesn't matter                                
                                                                                                                                                                                                                                                                                                                                                                               
  - If the excluded species had low confidence (e.g., lion at 3%), removing it barely affects renormalization                                                                                                                                                                                                                                                                  
  - If the second-best species is genuinely high confidence (e.g., lion 50%, bobcat 45%), renormalization gives bobcat ~90% which might actually be correct
                                                                                                                                                                                                                                                                                                                                                                               
  Why we decided to keep AddaxAI's approach                                                                                                                                                                                                                                                                                                                                    
                                                                                                                                                                                                                                                                                                                                                                               
  We discussed whether to implement SpeciesNet-style rollup during exclusion. The arguments against:                                                                                                                                                                                                                                                                           
                                                        
  1. It adds complexity: a second rollup mechanism alongside the existing postprocessing rollup                                                                                                                                                                                                                                                                                
  2. It's SpeciesNet-specific thinking: for smaller regional models (EUR-DF with 34 classes), species don't share taxonomy branches the same way. Excluding "fox" wouldn't roll up meaningfully to "canidae" because there might not be a canidae label
  3. The existing two-step process sometimes catches it: remove lion → bobcat at 40% → if below 0.65, the postprocessing rollup kicks in and sums felidae scores → returns "felidae". But if renormalization pushes bobcat above 0.65, the rollup is skipped                                                                                                                   
  4. Edge cases are rare: a model confidently predicting a geofenced species means the model is already confused about the geography. The wrong label would be caught during human verification                                                                                                                                                                                
                                                                                                                                                                                                                                                                                                                                                                               
  The gap that remains                                                                                                                                                                                                                                                                                                                                                         
                                                                                                                                                                                                                                                                                                                                                                               
  The case where it fails: high-confidence excluded species → renormalization inflates a wrong second-best above the rollup threshold → user sees a confident but incorrect species label. SpeciesNet handles this correctly, AddaxAI doesn't. We accepted this trade-off to keep the code simple and consistent across all models.  



>>> you mention "Excluding "fox" wouldn't roll up meaningfully to "canidae" because there might not be a canidae label" but does it need to be a model class? It can be a non model class, right? As in, it doesnt have to be a actual class the model was trained on? We can just aggregate and make a labal for canidae? Just like we do in the other rollup feature? That     
would make it a lot easier to implement the same SpeciesNet approach for all models, as we all have taxonomy.csv info for all models.


You're right. AddaxAI already creates rollup labels that the model was never trained on. The existing taxonomic rollup does exactly this: if no species crosses the threshold, it sums confidences at family level and creates a "felidae" label on the fly, adds it to classification_categories, and stores it in the taxonomy DB via add_rollup_taxonomy_entry(). The
  label doesn't need to be a model class.

  So the argument I made against implementing SpeciesNet-style exclusion rollup was wrong. The infrastructure is already there. Every model has taxonomy.csv, and the rollup machinery already handles creating new taxonomy labels at any level.

  This means we could implement exclusion rollup for all models with relatively little new code: reuse load_taxonomy_lookup() and the score accumulation logic from taxonomic_rollup.py, apply it during the exclusion step in label_exclusion.py. When a species is excluded, redirect its score to the nearest non-excluded ancestor, create the rollup label if needed.

```




When an excluded species rolls up to an ancestor (e.g., lion -> felidae), this creates a NEW classification_categories entry in the JSON. Should we also persist    
this to the JSON file on disk (like the existing postprocessing rollup does)

"like the existing postprocessing rollup does"??????
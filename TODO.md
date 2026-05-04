## Priority 1
- [ ] 

## Priority 2
- [ ]  

## Priority 3 
- [x] Add a readme file with instructions for Meta Testers. How to download, how to install, what to do with a bug, what to report back to me, whats still todo, how to reset, etc. etc. 
- [x] The map and activity pattern insight pages have a non consistent format of shoing there is no data (when there is no data). The other pages (matrix, deployment timeline, performace) all have the same format (card + center aligned bold title + greyish text center aligned). Make consistent. 
- [x] /insights/timeline -> "Drag horizontally across the date axis at the top to zoom into a specific range." there are no date ticks anymore....
- [x] In the deployments page on Windows the first col ('folder') shows the full path. Is that because of a different path seperator?
- [x] The flag emoticons dont work on windows, why? They show up as "EU", letters instead of flags. Other emojis do render like the globe and the cactus.. why? Can we fix this? The flag emojis are still not showing on Windows. ANy idea why? This was looked at before (check git logs), but that didnt solve it... 
- [ ] Add drift detection. The existing ModelCatalogUpdater.sync() in backend/app/ml/catalog_updater.py and ModelUpdateToast in frontend/src/App.tsx already do most of the plumbing for the model-revision flow, so the drift check can hang off the same machinery. 
>  - For models: store the HF revision SHA in manifest.json at download time, compare against HfApi().model_info(repo_id).sha on app startup, surface a "Update available" toast (the same path ModelUpdateToast already uses for new models).
>  - For envs: hash the bundled YAML at install time, store next to the env directory, rebuild on mismatch.

## AFter the Beta phase
- [ ] If everything works and all models are verified, please double check if there are any stale environment.ymls that are never used by any of the models. If so, remove them. 
- [ ] Any other non used imports or requirements in the environments YMLS? 
- [ ] The NSIS installer on windows shows a pbar, but it goes up and down a few times... 
- [ ] Bump addaxai-base from cu118 to cu128 so RTX 50-series (sm_120, Blackwell) gets native kernels instead of the 4-5 min PTX JIT fallback. Suggested pins: torch==2.8.0+cu128, torchvision==0.23.0+cu128, --extra-index-url https://download.pytorch.org/whl/cu128. Both windows and linux YAMLs. Adds ~700 MB to the install but fixes the GPU warning reported at https://forum.addaxai.com/t/model-warning-on-running-with-gpu/202. Requires NVIDIA driver >= 555.x, mention in the beta-tester readme. 
- [ ] Bump the pytorch env from Python 3.8 to 3.11 (3.8 is EOL since Oct 2024 and recent torch wheels are starting to drop py38 builds). Also bump torch alongside the python jump. SpeciesNet-fine-tuned classifiers (.pt files with pickled onnx2torch operator classes) need a smoke test after the bump: load NAM-ADS-v1 or similar and confirm torch.load() succeeds across the major version jump. 
- [ ] Do we want a custom minimal menu (just Reload / Force Reload / DevTools / About / Quit) with our own styling? Or keep the electron built in?


## Future stuff
- [ ] TIMELAPSE STANDALONE APP
- [ ] ALLOW FULL IMAGE CLS MODELs TOO (AHDRIFT-v1)
- [ ] MULTI LANGUAGE SUPPORT
- [ ] DEPTH ESTIMATION
- [ ] POSTPROCESS BATCH RESULTS MEGADETECTOR
- [ ] DOCUMENTATION
- [ ] REPEAT DETECTION ELIMINATION
- [ ] WLIDBOOKS INTEGRATION
- [ ] ADD ALL MODELS 


## Nice to haves
- [ ] SUBSAHARA GEOFILE - Add a geolocation file for the Sub Saharan model too, like SpeciesNet, so users of the SSmodel can also prefil by country. 
- [ ] CLS THRESH - add a classification threshold and a per species override. Check how that is one in AddaxAI-Connect. I want something like that.
- [ ] 


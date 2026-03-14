## TODO priority 1
- [ ] "Does not protect detections from reprocessing — if you rerun analysis, smoothing/postprocessing will overwrite labels on verified detections too" -> this is wrong! HUman verified is always better than ML perdictions. 
- [ ] Apparently "the classification worker avoids reloading the model for each deployment but is more complex to implement and manage". Should we make it simple and just batch process it every time again for each deployment and task (img / vid)? It adds a bit of model loading, but I would like to keep it as simple as possible. Investigate what is currently happening and report the options to me. What is possible to make it more simple and what would be the benefit? Is it a major refactor? 
- [ ] can we make the spotlight effect clearer in the grid thumbs? So outside the bbox more 
  dark, and the border more thick. Also, the overlapping bboxes whould not be dark. This is fixed in the large image view, but aparently in the thumbs bbox overlay it is not. The uniion of the bbooxes should not be dark. 
- [ ] is there a way that you can read the console.log yourself without me having to copy paste it every time?

## TODO priority 2
- [ ] make the pbars for processing more compact. Take the information that is now below it and put that in the pbar description with icons. SO it would be something like "Running... (file-icon) 18 of 46 - (elapsed time icon) 00:06 - etc. Then make the modal wider so it feels less cramped. 

## TODO priority 3
- [ ] merge all alembic/versions/ into one. We do not have any users yet, so we can make it just the start DB. 
- [ ] make the backgrounds of all modals not only overlay dark, bot also vague. 
- [ ] make sure on app istall it installs the default models and their environments (MDv5A and DINOv2-B). 

## TODO after beta version
- [ ] add a feature that allows datetime offset. This should happen at the "new deployment" options. Perhaps something that says "your data spans X days/weeks, etc. " Click here to see the burned in pixel dates (show a few images / frames) and show the extracted datetime next to it. Then users can add an offset to all data in the deployment. Add fast options to switch from AM to PM etc. +12:00 and -12:00. 
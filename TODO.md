## TODO priority 1
- [ ] "Does not protect detections from reprocessing — if you rerun analysis, smoothing/postprocessing will overwrite labels on verified detections too" -> this is wrong! HUman verified is always better than ML perdictions. 
- [ ] We should probabaly have something to tag the deteciton as a "False detection" in the similarity verification process. Right? In the event verification users would simply remove the box ("this is not an animal at all"), but in similarity verification it is a bit harder... What would you suggest? What is good UX? 
- [ ] do we need to add anythoing about this testing suite to the README.md / DEVELOPERS.md / other MD files? Or is it already there?
- [ ] Apparently "the classification worker avoids reloading the model for each deployment but is more complex to implement and manage". Should we make it simple and just batch process it every time again for each deployment and task (img / vid)? It adds a bit of model loading, but I would like to keep it as siomple as possible. Investigate what is currently happening and report the options to me. What is possible to make it more simple and what would be the benefit? Is it a major refactor? 
- [x] Can we confirm that the smoothing works for both taxonomies returned by SpeciesNet and taxonomies returned by other CLS models? 
- [x] Investiagte whether we can make the event smoothing more aggresive. And whether if wouold be translatable to a slider of some kind, or a dropdown with a few categories like mild, normal, aggresive, very aggresive, or simething like that. 
- [x] build a proper test infrascturture where we can keep adding tests. Add some basic ones to fill the test suite. 
- [ ] is there a way that you can read the console.log yourself without me having to copy paste it every time?
- [x] do not use instances of "blank", "false detection", "vide", "no cv result", (case insensitive) into the smoothing function and do not load into the DB. They should remain in the JSON as raw data, but should otherwise be ignored. 

## TODO priority 2
- [ ] make the pbars for processing more compact. Take the information that is now below it and put that in the pbar description with icons. SO it would be something like "Running... (file-icon) 18 of 46 - (elapsed time icon) 00:06 - etc. Then make the modal wider so it feels less cramped. 

## TODO priority 3
- [ ] merge all alembic/versions/ into one. We do not have any users yet, so we can make it just the start DB. 
- [ ] make the backgrounds of all modals not only overlay dark, bot also vague. 
- [ ] make sure on app istall it installs the default models and their environments (MDv5A and DINOv2-B). 

## TODO after beta version
- [ ] add a feature that allows datetime offset. This should happen at the "new deployment" options. Perhaps something that says "your data spans X days/weeks, etc. " Click here to see the burned in pixel dates (show a few images / frames) and show the extracted datetime next to it. Then users can add an offset to all data in the deployment. Add fast options to switch from AM to PM etc. +12:00 and -12:00. 
## TODO priority 1
- [ ] sometimes if you go to the a projects settings page, the dropdown manu value of "Event smoothing" is empty. Investigate whats going on there and why it is not always showing the selected value. 

## TODO priority 2
- [ ] Make sure a classification model is not required for a project. User can also just go with a megadetector version, and then do the identificaiton themselves. 

## TODO priority 3
- [ ] merge all alembic/versions/ into one. We do not have any users yet, so we can make it just the start DB. 
- [ ] make sure on app istall it installs the default models and their environments (MDv5A and DINOv2-B). 

## TODO after beta version
- [ ] add a feature that allows datetime offset. This should happen at the "new deployment" options. Perhaps something that says "your data spans X days/weeks, etc. " Click here to see the burned in pixel dates (show a few images / frames) and show the extracted datetime next to it. Then users can add an offset to all data in the deployment. Add fast options to switch from AM to PM etc. +12:00 and -12:00. 
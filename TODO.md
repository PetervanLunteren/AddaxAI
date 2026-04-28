## Priority 1
- [ ] 

## Priority 2
- [ ] Add flag button to and files. See addaxai. 
- [ ] remove the "today" legend item of the timeline plot, no need for that. 

## Priority 3 
- [ ] FILE VERIFICATION - add a section for image verification. Or at least thinik about it. You have something for events and detections (called similarity now). Should we make a thrid tab, files? exactly the same as events, but then on the file level. if you search fot wolf, you get all the images or frames with a wolf. Now in events you still have to search trhough the event to find it. events verify MaxN (and files if you want), files verify files (and if lucky its a maxN too), decections (or did we choose to call it observations? I think so), verify on the instance level. Here we do the embedding too. SO basically, just leave events and similarity as they are (perhaps rename similarity), and add a new one for files (which is almost the same as events, just not grouping for events). And while we're at it, should we make these their separate pages? Then we have all levels: sites, deployments, events, files, observations. Or would you advise against that and keep it all three in verify page as tabs? What is you recommendation in terms of UX UI? Be honest, dont sugar coat. 
- [ ] 




- [ ] Its getting pretty crowded in the filters bar.... should we move everything into a filters popover? Or just show a few and then move the more advanced filters into an "advanced" popover? What do you think in terms of UX UI? Be honest, no sugar coating. 

- [ ] How should we handle video's? Its basically just a bunch of frames. SHould we just show them as images? SO a video with 10 frames show up as 10 frames? Or should we merge them into one and then show the filmstrip like events has, since one video is still one file, but AddaxAI has splitted it into 10 frames. WHats the good mental model? Is probabaly not too much work to add the filmstip as with events, right? Since the code is already there. 

- [ ] In the files Modal, when drawing, it shows the to-be-drawn dashed box, but on release it is not visible. After heart-ing, it becomes visible. Some kind of refresh bug? 

- [ ] The (?) icon in the files Modal and files page, it reidrects to the event verification slideout. Make one specifically for the files. Propose a few options so I can select one.

- [ ] When first entering the events pages, it shows a modal with some basic explenation, right? First of all, can you show them to me? Or make sure that I can see them again now if I go to the verification pages (reset param). I'll inspect. After inspection I might have a few questions about the contetns of the existing ones, and after that, please make one for files verification too, in the same style and tone of voice. 

- [ ] should we have the pill inside the image in the observation grid too, to match the pattern of the event and files images? Or would you advise against that? 

- [ ] Should we improve the caption of the verify page now that we have three options? Do a few suggestions. 

- [ ] Are all the files shortcuts tested? 

- [ ] If I click a label, the label selection bar opens, I type "bird", get "Aves", enter, and it still shows the old label. If I then ENTER (verify the file), and go back, it shows Aves. SO the result works, but the intermidiate update of the label doesnt. 

- [ ] in the files modal, the "edit" button on already verfied files doesnt work. 

- [ ] what should 'A' add box do in the files modal? It currently doesnt do anything...

- [ ] I'm a bit confused as how the navigation works. we have left and right for just normal navigation (regardless of verification status), and we have >> for next unverified. Is that correct? Should we add a << for previous unverifed? should we do the arrows for the simple navigation (just prev next regardless of verification status)? 

- [ ] the "All labels" button should not have an icon (inconsistent with the other filters). remoce icon from that button in all three tabs of the verificaiton flow. 

- [ ] if all is done and fixed, focus on the texts




## New features
- [ ] TIMELAPSE STANDALONE APP
- [ ] MULTI LANGUAGE SUPPORT
- [ ] DEPTH ESTIMATION
- [ ] PROCESS BATCH RESULTS - https://github.com/agentmorris/MegaDetector/blob/main/megadetector/postprocessing/postprocess_batch_results.py
- [ ] DOCUMENTATION

## Installer
- [ ] merge all alembic/versions/ into one. We do not have any users yet, so we can make it just the start DB. 
- [ ] make sure on app istall it installs the default models and their environments (MDv5A and DINOv2-B). 
- [ ] 

## Nice to haves
- [ ] SUBSAHARA GEOFILE - Add a geolocation file for the Sub Saharan model too, like SpeciesNet, so users of the SSmodel can also prefil by country. 
- [ ] CLS THRESH - add a classification threshold and a per species override. Check how that is one in AddaxAI-Connect. I want something like that.
- [ ] 


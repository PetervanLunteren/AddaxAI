## Priority 1
- [x] When the "Nothing to promote" toast appears after clicking the "Add box" button in the EventsDetailsModal, the cross of the toast doesnt work. I have to wait until it disapears....
- [x] If I click "reembed" after I manually added some bboxes that need embedding, the modal shows me somethin like "Re-embedding detections - Deployment 1 of 2". Why is it showing me deployments counts? That is not relevant now, right? 
- [x] The activity pattern in the dashboard. There is not a lot of space for the dropdown next to the card title and caption. Should we stack it horizontally instead? Or make the dropdown less wide (by not showing the contents or something like that? ) What do you advise in terms of UX and UI? 
- [x] The space between these lines "Activity pattern <-> Observations by hour of day" and the others, e.g., "Detection trend <-> Observations over time" are not the same. I have the feeling because the dropdown on the activity card is not set up the saem way as the one on detection trend. Agree? Fix it. 
- [x] Is there a logical reason that the Label filter in the map is a multiselect of a flat list and not the hierarchical taxonomy tree we use for the verification filters and settings label selection? ONe could argue that it is not needed to choose all canids on a map, but I'd agrue otherwise. It makes it flexible for the user, they can navigate through the taxonomy and it makes it feel oncsistent with the rest of the app. Investigate why this is the case, and if no apparent reason, make it use the taxon tree modal. 
- [x] the filter bars in the verification tabs have the feature that the set filters show up bwloe the bar and a user can clear them one by one, or all together. The filter bars in the insights dont have that feature. Can we add that? It makes it feel consistent and also is good UX UI.
- [x] Activity overlap insight page. When selected a species A or B, a cross appears to remove that again, right? Can we make that cross with outer border button? Also, should we update the labels of these filters as it is not always a species, it can also be an order. What is better? "Label A"?
- [x] We need the logo in the app opf course. Totoally forgot about that. How and where do you want me to supply it to you? What format and what resolution? I can give you anything you want, just tell me the best way you want it, most efficient, and best results. 
- [x] Currently the overlay of bbounding boxes is spotlight, which means everything that is not the animal is darker. That is a great visual, but it also means that empty images are a lot lighter than the non-empty ones. Which is visually weird. should we darken the empties too (without any bboxes)? What do you think in terms of UX UI? 
- [x] Should we reverse the verify tabs from observations/captures/events? What do you think would need to be the default for users=? Probabaly observations is the most modern and state of the art verfication method, with embeddings and bulk verification options, right? 
- [x] The app is called "frontend" in electron. Why? Should we name it to "AddaxAI vXX.YY.ZZ"?
- [x] The Verification card in the dashboard doesnt have any caption... make consistent with the rest. 
- [x] The flag emoticons dont work on windows, why? They show up as "EU", letters instead of flags. Other emojis do render like the globe and the cactus.. why? Can we fix this? 
- [x] We dont have any environments.yml files for windows and linux yet. Build those. You can find previous working versions in this project: /Users/peter/Documents/Repos/AddaxAI. They were built in github actions: /Users/peter/Documents/Repos/AddaxAI/.github/workflows. there you'll find the requirements per OS per environment. Does that work for you? 
- [x] It would be cool if we could zoom in (read: set start and end date) by clicking and dragging on the /insights/timeline graph. Would that be difficult to do? No worries if you dont think its feasible. This is just a nice-to-have, not manadatory. KISS. 
- [x] Do a full audit to check all the title cases and em dashes. 2. **No Title Case** - Use natural English capitalisation. That means only capitalising the first word of sentences and proper nouns (like "Peter van Lunteren", "Utrecht", "MegaDetector", "SpeciesNet", "Today, I was walking in the park.",  "Things I love about Amsterdam.", "Cities visited"). Do capitalize the first letter of headers (e.g., "Detections per 100 trap-days", "Species selection", "Observations"). 3. **No em dashes** - Never use em dashes (—) or double hyphens (--) in text. Use commas, colons, semicolons, or separate sentences instead.
- [x] merge all alembic/versions/ into one. We do not have any users yet, so we can make it just the start DB. 
- [x] make sure on app istall it installs the default models and their environments (MDv5A and DINOv2-B). 
- [ ] ERROR REPORTING
- [ ] ADD ALL MODELS IN THE ZOO
- [ ] If I want to update the release tag: (1) should I bumb to which version number? What would you recommend? Its now ready for beta testing. Furthermore, what should I write in the release text? Give me about 10 bullet points of what the changes are with previous tag. ONly keywords, no lengthy texts. 

## Priority 2
- [ ] 

## Priority 3 
- [ ] If everything works and all models are verified, please double check if there are any stale environment.ymls that are never used by any of the models. If so, remove them. 
- [ ] Any other non used imports or requirements in the environments YMLS? 


## Future stuff
- [ ] TIMELAPSE STANDALONE APP
- [ ] MULTI LANGUAGE SUPPORT
- [ ] DEPTH ESTIMATION
- [ ] POSTPROCESS BATCH RESULTS MEGADETECTOR
- [ ] DOCUMENTATION
- [ ] REPEAT DETECTION ELIMINATION
- [ ] WLIDBOOKS INTEGRATION


## Nice to haves
- [ ] SUBSAHARA GEOFILE - Add a geolocation file for the Sub Saharan model too, like SpeciesNet, so users of the SSmodel can also prefil by country. 
- [ ] CLS THRESH - add a classification threshold and a per species override. Check how that is one in AddaxAI-Connect. I want something like that.
- [ ] 


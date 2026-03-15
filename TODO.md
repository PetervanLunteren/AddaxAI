## TODO priority 1
- [ ] sometimes if you go to the a projects settings page, the dropdown manu value of "Event smoothing" is empty. Investigate whats going on there and why it is not always showing the selected value. 
- [ ] For the dashboard page, have a look at a similar project (/Users/peter/Documents/Repos/AddaxAI-Connect) and how the dasboard is set up there. I want exactly the same dasboard, with the same widgets, and the same filters, etc. In terms of data source: Also use the same colors, and color codes for the species. Always prefer verified detecitons / files over non verfied detections / files. Instructions:
* Read all MD file in root to get a understanding of the project. 
* If something is unclear at any point, stop and ask before continuing.
* Prioritize simplicity and clarity over perfection. The code must be clean, easy to read, and understandable for collaborators. Avoid unnecessary complexity.
* I'm not in a rush. Please be precise and do the task thoroughly. 
* Please ask me any question for clarification. I would rather that you ask too many questions than assume certain details. 
* Ask at least 3 clarifying questions before beginning. Based on the conventions set out in CONVENTIONS.md and your knowledge, give your recommended solution to each questions you ask me. 
Workflow:
* Based on my answers, suggest a few general approaches. These should range from simple solutions to more sophisticated alternatives, with clear trade-offs for each. For every approach, explain:
   - Complexity (difficulty, dependencies, maintainability)
   - Readability (clarity for collaborators)
   - Effect (impact on performance, usability, flexibility)
* Give your recommendation regarding the alternatives discribed earlier, with a short reasoning. 
* After I select an approach, draft a detailed plan for implementation.
* Only start working if I agree with the proposed plan.

## TODO priority 2
- [ ] Make sure a classification model is not required for a project. User can also just go with a megadetector version, and then do the identificaiton themselves. 

## TODO priority 3
- [ ] merge all alembic/versions/ into one. We do not have any users yet, so we can make it just the start DB. 
- [ ] make sure on app istall it installs the default models and their environments (MDv5A and DINOv2-B). 

## TODO after beta version
- [ ] add a feature that allows datetime offset. This should happen at the "new deployment" options. Perhaps something that says "your data spans X days/weeks, etc. " Click here to see the burned in pixel dates (show a few images / frames) and show the extracted datetime next to it. Then users can add an offset to all data in the deployment. Add fast options to switch from AM to PM etc. +12:00 and -12:00. 
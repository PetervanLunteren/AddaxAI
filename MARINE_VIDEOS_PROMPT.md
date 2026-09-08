This is a GitHub work tree that is connected to the special branch where we'll work on making AddaxAI marine video friendly. This is a significant task. so, I'll work on this on a different branch and then at the end, we'll merge it to Main.

Please do a full audit of the current project and try to figure out how it works now for camera trap data. 

## audit

Instructions:
* Conduct a complete audit before making any changes. Assess the impact on the entire application.
* Codex will review your output once you are done, so make sure you exceed his expectations
* Do not sugar coat, be honest and clear
* Read all MD file in root to get a understanding of the project. 
* If something is unclear at any point, stop and ask before continuing.
* I'm not in a rush. Please be precise and do the task thoroughly. 
* Follow the KISS principle (Keep It Simple Stupid). Keep things as simple as possible.
* Follow the DRY principle (Don't Repeat Yourself). Avoid duplication and maintain a single source of truth.
* Before changing code, explain the current architecture and the intended change in your own words.
* Make the smallest safe change that solves the problem.
* Prefer simple, boring, maintainable code over clever abstractions.
* Do not introduce new dependencies unless there is a clear benefit.
* Preserve existing behaviour unless explicitly asked to change it.
* Update or add tests for the changed behaviour.
* Run the relevant checks, tests, linters, and type checks before finishing (backend only).
* Document any important trade-offs, risks, or follow-up work.
* If you find unrelated issues, mention them separately, but do not fix them unless asked.
* Follow the YAGNI principle (You Aren't Gonna Need It). Do not build functionality until it is actually needed.
* Use a clear mental model that makes the feature easy for users to understand. Good UX should feel simple.
* Please ask me any question for clarification. I would rather that you ask too many questions than assume certain details. 


A long time ago, I thought about how to add marine video functionality and then I wrote it down in this MD file: future-plans/marine-bruvs-mode.md. Please read this, but also take it with a grain of salt because it was a long time ago and AddaxAI itself has changed, so I'm not sure if this still is the the best solution. 

What I want in the long run is adding fish models, for example, Sharktrack and Community Fish Detector. 
For context:
1: Read the email threat in my Gmail, with these details. The full threat, please. 
        from:	Dan Morris <agentmorris@gmail.com>
        to:	    Peter van Lunteren <peter@addaxdatascience.com>
        cc:	    fppvrn@gmail.com
        date:	Jul 6, 2026, 4:18 PM
        subject:	Re: Fwd: SharkTrack in AddaxAI
2: https://github.com/filippovarini/sharktrack
3: https://www.fvarini.com/sharktrack
4: /Users/peter/Documents/Repos/sharktrack

Please, get yourself into depth about these models, which will be the first two models for the fish, marine functionality in AddaxAI. Do web queries and do a full audit of the repos and the websites and just figure out anything there is to know about these models and how to possibly integrate them into AddaxAI, the pros and cons, etcetera. 

## investigate

Instructions:
* Conduct a complete audit before starting.
* Codex will review your output once you are done, so make sure you exceed his expectations
* Do not sugar coat, be honest and clear
* Read all MD file in root to get a understanding of the project. 
* If something is unclear at any point, stop and ask before continuing.
* I'm not in a rush. Please be precise and do the task thoroughly. 
* Follow the KISS principle (Keep It Simple Stupid). Keep things as simple as possible.
* Follow the DRY principle (Don't Repeat Yourself). Avoid duplication and maintain a single source of truth.
* Follow the YAGNI principle (You Aren't Gonna Need It). Do not build functionality until it is actually needed.
* Document any important trade-offs, risks, or follow-up work.
* Please do webqueries if you wish to clarify things. I'm not in a rush and have enough tokens. 
* If you find unrelated issues, mention them separately, but do not fix them unless asked.
* Please ask me any question for clarification. I would rather that you ask too many questions than assume certain details. 

The first task for this whole marine video's functionality of AddaxAI and Dition is not to actually build it or plan how to build it. It's actually to get a picture of the overarching design. How would AddaxAI benefit most from having this marine videos functionality? What are the main issues and can we build it into the existing pathways like analyze a folder and build a project? Or do we need a separate a third card in the home screen that says marine videos or something like that, which is completely different? Because in fish and marine research, the verification is fundamentally different than with camera traps, right? In camera traps videos, those are typically like ten seconds or maybe twenty seconds, so you have one frame and that's enough and you can draw boxes and stuff like that. YAGNI how that works. You can figure it out from the current repo already. This is all very focused on camera traps, but now we're making something for marine research. And those folks need a tracker and that means frames per second, I don't know, a minimum of frame, five frames per second or something like that. Which is all of course, just machinery, that's fine, but it also means that it's not as simple as just storing one frame, the best frame for the verification. Their whole verification is different. It means, it means a completely new UI. And of course, we will share as much as the code as possible, dry, kiss, dry, YAGNI. But how docs it feel for the user, the user experience completely separate? And we just focus on some kind of analyze a folder, but then for videos, sorry, for marine videos, where the second step is a, a trigger, sorry, a track verifier instead of a detection verifier, something like that. Or do we fold it into the existing camera traps and for marine research, right? I find this very important: "A minimalist set that caters for all is a good design". 

But having enough FPS per video that makes a tracker viable means a lot of frames per second and running mega detector over a lot of frames per second makes it very slow. So I'm not sure if it is viable for camera traps and not a lot of people will do it. But if we can somehow make it that it's just an option and it's not advised for camera traps, but people can do it if they want. It's just if they have a very powerful GPU and they want tracks, then they can do tracks. That makes sense?

Please ask me at least twenty clarifying questions in interactive mode with your recommended answer as number one. Then do all of the stuff above and do a full audit, full investigation, check out how other platforms do it, do web queries, take all the time in the world. I have plenty of tokens. Please check how other platforms do marine video research verifications and how they fold everything in all the statistics, et cetera. This is a very long task. Take your time. I want this to be very thorough. KISS, DRY, YAGNI. Be critical and honest. Prefer simple, robust solutions. Be thorough and don’t cut corners. A minimalist set that caters for all is a good design.. 

Before writing down the final design, please recommend three different options and then I'll choose one which you will focus. 

## Propose options

Workflow:
Instructions:
* Conduct a complete audit before starting.
* Codex will review your output once you are done, so make sure you exceed his expectations
* Do not sugar coat, be honest and clear
* Read all MD file in root to get a understanding of the project. 
* If something is unclear at any point, stop and ask before continuing.
* I'm not in a rush. Please be precise and do the task thoroughly. 
* Follow the KISS principle (Keep It Simple Stupid). Keep things as simple as possible.
* Follow the DRY principle (Don't Repeat Yourself). Avoid duplication and maintain a single source of truth.
* Follow the YAGNI principle (You Aren't Gonna Need It). Do not build functionality until it is actually needed.
* Document any important trade-offs, risks, or follow-up work.
* Please do webqueries if you wish to clarify things. I'm not in a rush and have enough tokens. 
* If you find unrelated issues, mention them separately, but do not fix them unless asked.
* Please ask me any question for clarification. I would rather that you ask too many questions than assume certain details. 
* Ask me clarifying questions before beginning. Based on the conventions set out in CONVENTIONS.md and your knowledge, give your recommended solution to each questions you ask me. The minimum number of questions (in interactive mode with selectable predefined multiple choice answers, the first always being your recommended answer) to ask me is 4
* Based on my answers, suggest a few general approaches. These should range from simple solutions to more sophisticated alternatives, with clear trade-offs for each. For every approach, explain:
   - Complexity (difficulty, dependencies, maintainability)
   - Readability (clarity for collaborators)
   - Effect (impact on performance, usability, flexibility)
* Give your recommendation regarding the alternatives discribed earlier, with a short reasoning. Be short and concise. Key words if possible.

I will also ask this exact thing to CLaude code and to codex, and then we'll compare designs. Please write your design to the root as an MD file. 


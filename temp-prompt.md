
I would like you to focus on the ML part now, and implement a full working example of running two models, namely:

Det: MD1000-REDWOOD-0-0
Cls: NAM-ADS-v1

(both models are prepared already)

On this test directory: /Users/peter/Downloads/test-img, which contains one image giraffe.jpg.

The outcome of this task must be the EXACT same results (that includes the exact same bbox, prediction and confidence) as these: /Users/peter/Downloads/test-img/image_recognition_file.json

I want you to implement the same method as I have already implemented in this repo: /Users/peter/Documents/Repos/streamlit-AddaxAI/, so please first thoroughly investigate how it works there. Each model has its own inference script under the "type" field in the manifest. That is bettle tested, and I want to implement that appraoch into the current repo too. 

Instructions:
* If something is unclear at any point, stop and ask before continuing.
* Prioritize simplicity and clarity over perfection. The code must be clean, easy to read, and understandable for collaborators. Avoid unnecessary complexity.

Workflow:
* Ask at least 3 clarifying questions before beginning.
* Based on my answers, suggest a few general approaches. These should range from simple solutions to more sophisticated alternatives, with clear trade-offs for each. For every approach, explain:
   - Complexity (difficulty, dependencies, maintainability)
   - Readability (clarity for collaborators)
   - Effect (impact on performance, usability, flexibility)
* Give your recommendation regarding the alternatives discribed earlier, with a short reasoning. 
* After I select an approach, draft a detailed plan for implementation.
* Only start working if I agree with the proposed plan.

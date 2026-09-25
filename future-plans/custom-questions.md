# Custom questions about pixels in AddaxAI: investigation and future plan

Status: investigation only, no code written.
Date: 2026-09-25
Branch the investigation ran on: `claude/lay-of-the-land-doulv4`
Repo state at time of audit: `b8b75e2` on `main`, VERSION `0.0.0-dev`

This document is the raw material for a proper implementation plan. It holds the
original brief, the questions to answer, what the audit of the repo found, what the
literature and other platforms do, the three approaches that were compared, why the
simplest one won, how it relates to the planned SpeciesNet fine-tuning feature, what to
deliberately leave out, the risks, and the sources. It is written to be picked up cold
months later without rerunning the investigation.

It deliberately stops short of UI design and field-level schema detail. What matters here
is which approach is sound, which of the example questions it can and cannot answer, and
what the existing code already provides.

---

## 1. The original brief

Verbatim, as given, over three messages:

> I'm investigating the feasibility of adding a feature where poeple can prompt and get
> answers based on their pixel data. Something like: is this animal sleeping ? Yes or no.
> Or what is the antler size if present? None/small/medium/large. Or what coat do the pigs
> have? Black/white/hybrid. Or what is the weather conditions? Sunny/rain/snow.
>
> Can you do webqueries and find out how this would work? No need for details into the
> integration yet, just find out how it would work, what best practise are, how other
> platforms do it, etc, and what the feasibility would be for adding this to the Current
> AddaxAI repo app. Be honest no sugar coating. Be thorough and concise

> Ok, so the well established way is to train a small custom classifier or the existing
> embeddings with a few dozen samples per class? Is that a battle tested method? Is that
> how the well established platforms handle this? Is this best practice?

> Ok. How does this differ from finetuning SpeciesNet with the verified species labels ?
> https://agentmorris.github.io/speciesnet-fine-tuning/ which is also on my todo list to
> build as a feature. It will yield a new species model, where as this will answer custom
> questions. Will it be OK to have both? Or will it be confusing to users? What will be
> the compute necessary for both? Does the embeddings training also work with epochs and
> hyperparametets etc?

## 2. Questions to answer

1. How can a user ask a closed question about their images and get an answer per image
   or per detection?
2. What is best practice, and how do other platforms do it?
3. Which of the four example questions (sleeping, antler size, pig coat, weather) are
   answerable from pixels at all?
4. Is "train a small classifier on stored embeddings from a few dozen examples" a proven
   method, and is it what established platforms do?
5. How does it differ from fine-tuning SpeciesNet on verified labels, and can both live in
   the app without confusing users?
6. What compute does each need?
7. Does the embedding classifier have epochs and hyperparameters to tune?
8. How feasible is it in the current repo?

## 3. Goals

- Let a user define a question with a fixed set of answers and get a suggested answer on
  every detection in a project.
- Stay offline, private and free to run, like the rest of the app.
- Run on any laptop, including CPU-only machines and Apple Silicon.
- Say honestly how well it works on this project's data before a bulk run, not after.
- Feed answers into verification as suggestions a person confirms, never as silent truth.
- Reuse what already exists rather than add a model, an env or a download.

Non-goals for a first version: free-text prompting, scene-level questions (weather),
questions that need several frames, cross-project reuse of a trained question, cloud
models.

---

## 4. Repo audit: the lay of the land

### 4.1 Embeddings already exist for most detections

`Project.embedding_model_id` defaults to `DINOV2-VITS14` (`backend/app/models/project.py:62`),
so a new project embeds by default. Phase 8 of `detection_worker.py` (around line 1029)
computes the vectors, and `backend/app/workers/embedding_worker.py` recomputes them when the
project's embedding model changes. `models.json` offers three: DINOv2 ViT-S/14 (384
dimensions), ViT-B/14 (768) and ViT-L/14 (1024), all in `env-addaxai-base`.

Storage is `detection_embeddings` (`backend/app/models/detection_embedding.py`): one row per
`(detection, embedding model)`, the vector as float16 bytes, its dimension and L2 norm.

Which detections get a vector is decided by `build_embedding_input` in
`backend/app/ml/embedding_utils.py`:

- only detections at or above the project's classification gate, plus every verified one
  (`threshold_or_verified`)
- image detections against the file itself
- video detections only on the best frame (`best_frame_number`), the rest are skipped
- event-level observations with no bbox are skipped

So a custom question can only ever be answered for detections that have a vector, which is
the same population the verify grid already works with.

### 4.2 The crop throws away the edges of wide boxes

This is the one finding that directly limits the feature. `embedding_script.py` crops the
tight bbox (`crop_detection`, line 128, no padding) and then runs the standard DINOv2
transform (`build_transform`, line 115): `Resize(input_size)` on the short side, then
`CenterCrop(input_size)`.

For a non-square box, the centre crop keeps only the middle square of the long side. A deer
walking across the frame has a wide box, so the head and the tail are cut off before the
model ever sees them. Antlers are on the head. A question about antler size would be
learning from vectors that often do not contain the antlers.

This does not matter for similarity search on species, which is what the embeddings were
built for, because the body carries most of the species signal. It matters a lot for
attribute questions that live at one end of the animal (antlers, tusks, a tail posture, a
calf at the flank). Fixing it means letterboxing (pad to square) instead of centre
cropping, which changes every stored vector and therefore needs a re-embed. See section 11.

### 4.3 The app already classifies from embeddings

`backend/app/ml/inference/similarity_script.py` already runs a nearest-neighbour classifier
over the stored vectors. `do_cohorts` (line 1112) and the suggestions sort take each
unverified detection, look at its neighbours in a FAISS index, and when the neighbours
agree on a different label (`_compute_neighbor_signals`, `_is_useful_suggestion`) propose
it as a correction. `frontend/src/components/verify/SuggestionsToolbarPill.tsx` is the
entry point. Suggestions can be dismissed (`suggestion_dismissed`).

That is the same family of method this plan proposes: a classifier on frozen embeddings,
producing suggestions a person reviews. The custom-question feature is a sibling of an
existing, shipped feature, not a new paradigm for the app.

One limit to note: `MAX_EMBEDDINGS = 20_000` (line 48) caps how many vectors the FAISS
passes load. A linear classifier does not need that cap, because prediction is a matrix
product that can stream in chunks.

### 4.4 Attributes on observations are fixed vocabularies

`backend/app/core/observation_attributes.py` holds `SEXES`, `LIFE_STAGES` and `BEHAVIORS`,
mirrored in `frontend/src/lib/observation-attributes.ts`. They live on `event_observations`
(`sex`, `life_stage`, `behavior`), per cohort, set by a person only; the AI never fills
them. NULL means unknown, and the docstring forbids a literal `"unknown"` value because it
is not in the Camtrap DP enums. The rebuild rules for these fields are subtle
(`DEVELOPERS.md`, "Observation cohorts").

Custom questions do not fit here. They are per detection, the answer set is user-defined,
and they would be AI-filled. Forcing them into cohort rows would collide with the rebuild
seed rule. They need their own storage (section 9.2). The one overlap is `behavior`, where
"resting" already exists; section 9.6 covers how not to create two sources of truth.

### 4.5 Validation views exist

`frontend/src/pages/ConfusionMatrixPage.tsx` and `PerClassPerformancePage.tsx` compare the
model's original prediction with the verified label, per detection, with site filters.
That is the shape of the accuracy check this feature needs, and the one SpeciesNet
fine-tuning needs too.

### 4.6 Environments

`env-addaxai-base` carries torch 2.8, torchvision, numpy and faiss-cpu, but not
scikit-learn. scikit-learn exists only in the `tensorflow-v1` and `tensorflow-v2` envs. The
backend's own `requirements.txt` has numpy but neither scipy nor scikit-learn. Section 9.4
covers the choice.

### 4.7 What does not exist

- No full-frame (scene) embedding. Every vector is a detection crop.
- No per-detection storage for arbitrary question answers.
- No training of anything inside the app. Custom classifiers enter through the model zoo
  (`DEVELOPERS.md`, "Creating a custom classification model"), trained elsewhere.
- No reference to fine-tuning anywhere in the backend, frontend or docs.

---

## 5. Literature and prior art

### 5.1 Generative vision-language models (VLMs)

The obvious way to "ask a question about an image" is a VLM: send the image and the
question with the allowed answers, and read the answer back. The evidence on camera trap
data is poor for the model sizes that can run on a laptop.

- Zhou et al. (September 2026) tested Qwen3-VL 2B, 4B and 8B and Gemma 3 4B against BioCLIP
  (300M parameters) on 96 species. BioCLIP beat every VLM by 33 to 59 percentage points,
  and every model lost 10 to 27 points going from iNaturalist photos to camera trap images
  (https://arxiv.org/abs/2609.11916). Species ID, not attributes, but it shows where small
  VLMs stand on this imagery.
- Real-Wild-VLM (CVPR 2026 workshop) found open VLMs biased toward saying something is
  there, failing to reject empty infrared clips, and degrading at night. Only prompts with
  explicit rejection rules helped
  (https://openaccess.thecvf.com/content/CVPR2026W/DataCV/html/Deng_Real-Wild-VLM_Prompting_Large_Vision-Language_Models_for_Wildlife_Recognition_in_Camera-Trap_CVPRW_2026_paper.html).
- The yes-bias on binary questions is a general, well documented VLM failure (POPE,
  https://aclanthology.org/2023.emnlp-main.20.pdf). A question like "is this animal
  sleeping, yes or no" is exactly the format that triggers it.
- A model's stated confidence tracks commitment more than correctness
  (https://arxiv.org/pdf/2606.29490), and counting is poor
  (https://arxiv.org/pdf/2510.04401).
- Constrained output works in principle (JSON schema, temperature 0 in Ollama), but
  enforcement is buggy for some model and endpoint combinations
  (https://github.com/ollama/ollama/issues/15540).

### 5.2 Contrastive zero-shot (CLIP family)

Each answer becomes a text prompt, the image is scored against each, the best match wins.
No training, real scores, cheap.

- Dussert et al. (2025) classified eating, moving and resting in chamois, red deer and roe
  deer with CLIP, SigLIP, WildCLIP and three multimodal LLMs. Best F1 was 86.39%, and the
  resulting activity patterns overlapped citizen-science data by 84 to 90%
  (https://besjournals.onlinelibrary.wiley.com/doi/10.1111/2041-210X.70059). The best
  evidence that behaviour questions of the "is it resting" kind are workable.
- WildCLIP fine-tuned CLIP for natural-language retrieval of scenes and attributes on
  Snapshot Serengeti. GPL-3.0, so not bundleable in an MIT app
  (https://github.com/amathislab/wildclip).
- BioCLIP 2 reports 84.3% zero-shot per-class accuracy on LILA camera trap species and
  trait transfer that improves with scale (https://huggingface.co/imageomics/bioclip-2,
  https://github.com/Imageomics/bioclip-2). A separate paper found zero-shot BioCLIP
  struggles on camera traps (https://link.springer.com/chapter/10.1007/978-3-032-01472-6_11).

### 5.3 A linear classifier on frozen embeddings

Freeze a pretrained model, take its embeddings, fit a small linear classifier on a few
labelled examples. Called linear probing, it is the standard way foundation models such as
DINOv2 are evaluated and reused (https://arxiv.org/html/2304.07193v2).

- Google's Perch team recommends exactly this for bioacoustics, as a workflow they call
  agile modelling: a person labels a few examples, a linear classifier is fitted on the
  Perch embeddings, the most useful next examples are surfaced, repeat. They report a
  working custom classifier within a couple of hours, and it is used on birds, reefs and
  whales (https://arxiv.org/pdf/2505.03071, https://github.com/google-research/perch-hoplite,
  https://research.google/blog/how-ai-trained-on-birds-is-surfacing-underwater-mysteries/).
- Norouzzadeh, Morris, Beery et al. (2021) published the same pipeline for camera trap
  species: detector, crop embeddings, active learning on the embeddings
  (https://besjournals.onlinelibrary.wiley.com/doi/10.1111/2041-210X.13504,
  https://github.com/microsoft/CameraTraps/tree/norouzzadeh-et-al-2020/research/active_learning).
- A 2026 marine study found 10 to 20 labels per species with frozen DINOv2 plus a linear
  classifier reached near-plateau accuracy at a new site (https://pith.science/paper/2607.02559).
- Frozen DINOv3 embeddings cluster camera trap species better than BioCLIP
  (https://arxiv.org/pdf/2510.14596), and logistic regression on frozen foundation
  embeddings is a label-efficient baseline elsewhere in ecology
  (https://arxiv.org/pdf/2604.00313).

### 5.4 What established camera trap platforms do

None of them offer user-defined attribute questions, as far as this investigation could
find.

- Wildlife Insights and SpeciesNet: a fixed species classifier
  (https://www.wildlifeinsights.org/about-wildlife-insights-ai). Research around the
  platform uses VLMs plus retrieval to write reports, not to answer user questions
  (https://pmc.ncbi.nlm.nih.gov/articles/PMC11679253/).
- Zamba Cloud: full custom model training on the user's labels
  (https://www.zambacloud.com/).
- Timelapse imports AddaxAI or MegaDetector results
  (https://saul.cpsc.ucalgary.ca/timelapse/).
- Agouti, Trapper, TrapTagger: no evidence found of prompt or attribute questions.
- Behaviour, sex and age are entered by hand as fixed fields, as in Camtrap DP and in
  AddaxAI's own `behavior` field.

The closest general-purpose analogue is Label Studio Prompts: write the prompt, mark a
subset as ground truth, measure against it, iterate, then run on everything with a person
verifying (https://humansignal.com/platform/prompts/,
https://labelstud.io/blog/how-to-evaluate-and-compare-llms-using-prompts-in-label-studio/).
That loop is the real best practice, whatever model sits underneath.

So: the method is proven in machine learning, in bioacoustics and in camera trap research,
but a camera trap desktop app offering it would be early. There is no shipped template to
copy for the UX or the failure modes.

---

## 6. The three approaches compared

| | Generative VLM | CLIP zero-shot | Linear classifier on embeddings |
|---|---|---|---|
| User input | Question text | Question and answer texts | Question, answers, a few dozen labels per answer |
| Truly free-form | Yes | Mostly | No, closed answers only |
| Accuracy on camera traps | Weak for laptop-sized models | Moderate, good for coarse behaviour | Good where the trait is visible in the crop |
| Night / infrared | Degrades, guesses | Degrades | Learns from the user's own night examples |
| Confidence | Not a probability | A score, roughly calibrated | A probability, reasonably calibrated |
| Can be measured before bulk run | Only with labels | Only with labels | Built in, labels are the input |
| New model / env / download | Yes, multi-GB, new runtime | Yes, one model | None |
| Compute per 100k detections | Hours to a day on GPU, worse on CPU (estimate) | Minutes on GPU | Seconds, any CPU |
| Offline | Local only if bundled | Yes | Yes |

The VLM column is attractive in a demo and weak in practice on this data. CLIP zero-shot is
a reasonable cold start but still needs labels to know if it works. The linear classifier
needs labels up front, but those labels double as the accuracy check, and it costs nothing
to run.

---

## 7. The core decision

Build custom questions as a linear classifier on the embeddings the project already has,
wrapped in a label, measure, correct loop. Do not start with a generative VLM.

The reasoning, in order of weight:

1. **Honesty.** The user sees held-out accuracy on their own data before trusting it. A VLM
   gives a fluent answer with no way to know how often it is wrong.
2. **Zero new infrastructure.** No model, no env, no download, no GPU. The vectors are
   already in the database for most projects.
3. **It matches a shipped pattern.** Neighbour suggestions already classify from these
   vectors and route through verification (section 4.3).
4. **Night works.** The classifier learns from whatever the user labels, infrared included,
   instead of relying on a model that was trained on daylight web photos.
5. **The evidence supports it.** Perch agile modelling and Norouzzadeh et al. are the same
   method in production and in camera trap research.

What the user gives up is free-form typing. They must pick answers and label examples.
That is the price of knowing whether the answers are right.

---

## 8. Answers to the questions

### 8.1 How it works and whether it is best practice (Q1, Q2, Q4)

A user defines a question and its answers, labels a few dozen detections per answer, the
app fits a linear classifier on the stored vectors, reports held-out accuracy, and on
acceptance writes a suggested answer and a score onto every embedded detection. Uncertain
cases are surfaced for labelling next, and the loop repeats.

It is best practice in the sense that it is what the field converges on when labels are
scarce and classes are user-defined (section 5.3). It is not what camera trap platforms
ship today (section 5.4). A few dozen labels is a starting point, not a guarantee; the loop
around the classifier is what makes it work, not the classifier.

### 8.2 The four example questions (Q3)

- **Sleeping, yes or no.** "Resting" is workable per Dussert et al. "Sleeping" versus
  "resting" is often undecidable from one frame, even for a person. Recommend the user
  frames it as posture (lying / standing / moving). Also overlaps the existing `behavior`
  vocabulary, see section 9.6.
- **Antler size, none/small/medium/large.** Hard. No scale reference, depends on angle and
  distance, seasonal, and annotators disagree on "medium". Worse in AddaxAI specifically
  because the centre crop often cuts off the head (section 4.2). Do not promise this one
  until the crop is fixed and a spike shows it works.
- **Pig coat, black/white/hybrid.** Easy in daylight, impossible on infrared, which is
  greyscale. The answer set must allow "cannot tell", and the user must label night
  examples as such so the classifier learns to say it.
- **Weather, sunny/rain/snow.** Out of scope for v1. It is a scene question and there are
  no scene embeddings. Snow detection from cameras is established (92 to 98% with trained
  CNNs, https://pmc.ncbi.nlm.nih.gov/articles/PMC6307743/), rain is rarely visible in a
  still, and historical weather looked up from the deployment's coordinates and timestamp
  would beat pixels for most of it, at the cost of needing a network connection.

### 8.3 Relation to SpeciesNet fine-tuning (Q5)

They reuse pretrained knowledge in two different ways and produce different things.

Fine-tuning SpeciesNet changes part of the network. Dan Morris's tutorial trains the
classification head plus the last 2 of 7 EfficientNetV2-M stages by default, on crops, 20
epochs with early stopping, learning rate 1e-4, batch size 32, dropping classes below 100
crops and calling low thousands per class comfortable
(https://github.com/agentmorris/speciesnet-fine-tuning). The WildObs pipeline trains the
head, the top convolution and block 7 with `ReduceLROnPlateau`
(https://github.com/WildObs/SpeciesNet-FineTuning). A 2026 study found performance
plateaued at 250 to 500 local images per class
(https://www.sciencedirect.com/science/article/pii/S0048969726005905). The output is a
standalone model file that goes into the model list and runs on new folders like any
classifier.

A custom question changes no network. It fits a small linear layer on existing vectors,
needs a few dozen labels, and adds a column to one project's existing detections. It is not
portable between projects unless they share the embedding model, and it is not meant to be.

Could the embedding classifier also do species? Technically yes (Norouzzadeh et al.). But
with thousands of verified labels a fine-tuned specialist will beat a linear layer on
generic DINOv2 features for fine-grained species, and it is portable and shareable. The two
do not compete: many labels and a reusable model on one side, few labels and a local
answer on the other.

Having both is fine if the UI frames them by what the user gets, not how it works:

- "Train your own species model", near the model list, producing a model to run on new
  data.
- "Custom questions", inside a project, adding answers to detections already there.
- No shared vocabulary in the UI. Epochs and backbones never appear in the question flow.
- The question setup refuses or redirects a question whose answers are species names.

What they should share is the validation step: held-out accuracy, confusion matrix, and a
split by deployment or site. Verified images from the same camera are near duplicates, and
a random split leaks them into the test set and inflates accuracy for both features.

### 8.4 Compute (Q6)

SpeciesNet fine-tuning is heavy: EfficientNetV2-M, about 54M parameters, repeated forward
and backward passes over thousands of crops. The tutorial's example trained one unfrozen
block in under an hour and an 80k-crop run in hours, without naming the GPU. By estimate,
not measurement: tens of minutes to a few hours on a decent NVIDIA GPU, several times
slower on Apple Silicon (MPS), and impractical on CPU beyond small datasets. Given macOS
arm64 is the canonical build, that feature needs MPS handling, progress, resume and a
clear "this will take hours" warning.

A custom question is close to free once vectors exist. The only real cost is the DINOv2
pass per crop, which the project already paid for in phase 8. Fitting a logistic
regression on a few hundred 384-dimensional vectors takes well under a second on any CPU.
Predicting 100k detections is one matrix product, seconds at most.

### 8.5 Epochs and hyperparameters (Q7)

Essentially none. Logistic regression is convex; the solver iterates to the single best
answer, so there are no epochs and no learning rate to choose. What remains:

- Regularisation strength C: pick by cross-validation over a handful of values,
  milliseconds, invisible to the user.
- Class weighting: balanced, so a rare answer is not swamped by a common one.
- Which embedding model: already fixed by the project.

A small neural network head would bring epochs back and rarely beats logistic regression
at a few dozen labels. YAGNI. The honest limit is not tuning but whether the trait is in
the crop and in the vector, and the held-out accuracy reports that directly.

### 8.6 Feasibility in the current repo (Q8)

High for detection-level questions. The vectors, the subprocess env, the suggestion
pattern, the verify UI and the confusion matrix view all exist. New work is storage,
one small training script, a job, and the question setup and labelling UI. Two real
obstacles: the centre crop (section 4.2) for end-of-body traits, and the absence of scene
embeddings for image-level questions.

---

## 9. Recommended design, at the level of mechanics

### 9.1 Scope of v1

Detection-level questions only, answered from the existing crop embeddings. Closed answer
sets of two to about six answers. One project at a time. Images and video best frames,
which is what already has vectors.

### 9.2 Storage

Two new tables, sketched rather than specified:

- `custom_questions`: project, question text, ordered answer list, the embedding model the
  classifier was fitted on, the fitted weights (a few kilobytes, a JSON or blob column is
  enough), held-out accuracy, created and fitted timestamps.
- `detection_answers`: detection, question, answer, source (human or AI), score. Human rows
  are the training labels. AI rows are replaced on every refit.

Deleting a project or deployment must remove these rows. They join the leaf-first purge in
`purge_deployment_data()` (`DEVELOPERS.md`, "Deleting analysis data").

### 9.3 The loop

1. Define the question and answers. Always offer a "cannot tell" answer as a trainable
   class, distinct from "not answered" (NULL).
2. Seed labelling: show a diverse sample (the greedy similarity order already exists),
   the user labels until each answer has a minimum count.
3. Fit, with a split by deployment, report held-out accuracy and a confusion matrix.
4. Show the least certain detections next, label, refit. Repeat until the user is happy.
5. Predict all embedded detections, write AI answers with scores.
6. Review in the verify grid: filter by answer and score, confirm or correct. Corrections
   become labels for the next fit.

### 9.4 Where the training runs

Three options, to settle before building:

- Add scikit-learn to `env-addaxai-base` and use `LogisticRegressionCV`. The built-in,
  standard tool (convention 13). Costs an env change for existing installs, which the
  env-drift mechanism already handles (`DEVELOPERS.md`, "Keeping installed models up to
  date").
- Use torch, already in `env-addaxai-base`: one `nn.Linear` trained with LBFGS. No new
  dependency, but it is custom code for something scikit-learn does in one call.
- Run in the backend process with numpy. No subprocess overhead, but no scipy there either,
  so the solver would be hand-written. Not recommended.

The subprocess pattern should follow `similarity_script.py`: stdlib plus numpy, read the
vectors straight from SQLite, NDJSON progress on stdout.

### 9.5 Exports

One column per question in the detection-level tables, and a mention in the run README.
Camtrap DP has no field for arbitrary attributes; `observationComments` or an extension
column is the only honest place, to decide later.

### 9.6 The overlap with `behavior`

"Is it resting" as a custom question and `behavior = resting` on a cohort are two sources
of truth for one fact. Keep them separate in v1 and document it. A later step could offer
"use this question to suggest behaviour", which writes suggestions into the cohort field
through the existing human edit path. Not in v1.

---

## 10. What to deliberately not build

- A generative VLM, local or cloud. Revisit only if users show the labelled approach is not
  enough, and then as an optional power feature.
- Free-text answers. Closed sets only.
- Scene questions and scene embeddings.
- A neural network head, hyperparameter settings, or an epoch count anywhere in the UI.
- Sharing a trained question between projects.
- Multi-frame or event-level questions.
- Writing AI answers into `sex`, `life_stage` or `behavior`.
- Weather from pixels. If weather is wanted, it is a separate feature from coordinates and
  timestamps.

---

## 11. Risks and honest limits

| Risk | Severity | Mitigation |
|---|---|---|
| Centre crop drops the head or tail of wide boxes | High for end-of-body traits | Letterbox instead of centre crop, then re-embed; spike first to measure the gain |
| Trait not captured by a generic DINOv2 vector | Medium, question-specific | Held-out accuracy says so; tell the user plainly when it is low |
| Leakage between near-duplicate frames inflates accuracy | High if ignored | Split by deployment or site, never at random |
| Infrared makes colour questions unanswerable | Medium | "Cannot tell" answer, night examples in the labels |
| Annotator disagreement on ordinal answers (antler size) | Medium | Warn on ordinal scales; fewer, clearer answers |
| Too few labels for a rare answer | Medium | Minimum per answer before fitting; balanced class weights |
| Users expect it to understand any question | Medium, reputational | Name it "custom questions", show accuracy up front, never show a result without it |
| Confusion with SpeciesNet fine-tuning | Low with separate placement | Section 8.3 |
| Projects with embeddings disabled | Low | Offer to run the existing re-embed job |

Changing the crop is not free. It changes every stored vector, so the neighbour suggestions
and similarity sort would shift too, and every existing project needs a re-embed. It may
also improve species similarity for elongated animals. Measure before deciding.

---

## 12. Effort estimate

Backend, roughly a week:

- Migration for the two tables and the purge path: half a day
- Training and prediction subprocess script, plus tests: one day
- Router, job and worker: one day
- Accuracy split by deployment, shared with the confusion matrix endpoint: one day
- Exports: half a day

Frontend, about a week: question setup, the labelling view (reusing the crop grid), the
accuracy panel, and the answer filter in verify.

Docs, half a day: one page under `docs/docs/guides/` with the honest limits, including
which kinds of questions do not work.

Total, about two and a half weeks for a v1. The crop change, if the spike justifies it, is
another day plus a re-embed for every project.

## 13. Step zero, before any of that

Spend a day on a throwaway script against real projects. Pull the stored vectors and the
verified labels for two or three questions (a posture question, a coat colour question with
night images, and antler presence), label a few dozen per answer, fit a logistic
regression, and report accuracy with a split by deployment. Then repeat with letterboxed
crops for the antler question and compare.

If posture and coat colour land well and antlers improve clearly with letterboxing, build
as above. If posture fails on stored vectors, the plan needs rethinking before any UI is
written.

---

## 14. Reproducibility notes

### 14.1 What was audited

Read in full or in part: `CONVENTIONS.md`, `README.md`, `DEVELOPERS.md` (table of contents
plus the observation cohorts, custom classification model and embedding mentions),
`models.json` (the `emb` entries), `backend/app/core/observation_attributes.py`,
`backend/app/models/{project,detection_embedding,event_observation}.py`,
`backend/app/ml/embedding_utils.py`, `backend/app/ml/inference/embedding_model.py`,
`backend/app/ml/inference/embedding_script.py` (transform and crop),
`backend/app/ml/inference/similarity_script.py` (header, `do_cohorts`, `_group_cohorts`,
`_is_useful_suggestion`, `MAX_EMBEDDINGS`), `backend/app/workers/embedding_worker.py`
(header), `backend/app/workers/detection_worker.py` (phase 8),
`backend/app/ml/envs/addaxai-base/linux/environment.yml`, `backend/requirements.txt`,
`frontend/src/components/verify/SuggestionsToolbarPill.tsx`,
`frontend/src/pages/ConfusionMatrixPage.tsx` (header), and the `future-plans/` documents
for structure.

Grepped: `fine-tun|finetun` (only env files), `scikit|sklearn|scipy` across envs and
requirements, `suggest` in routers, `cohort` in the backend.

### 14.2 Egress limits during the investigation

The session's proxy blocked `arxiv.org`, `export.arxiv.org`, `besjournals.onlinelibrary.wiley.com`,
`biorxiv.org`, `link.springer.com`, `openaccess.thecvf.com`, `journals-sol.sbc.org.br` and
`agentmorris.github.io`. `github.com` was reachable.

Therefore: the two SpeciesNet fine-tuning repositories and the WildCLIP repository were
read directly. Everything from arXiv, the journals and the CVF was taken from web-search
summaries, not the primary text. **Before implementation, read Zhou et al. (2609.11916),
Dussert et al. (2025), Real-Wild-VLM and the marine label-count study in full.** The numbers
quoted (33 to 59 points, 86.39% F1, 84 to 90% overlap, 10 to 20 labels, 250 to 500 images
per class, 84.3% zero-shot) are second-hand.

### 14.3 Search queries that produced the useful hits

- `vision language models camera trap images evaluation behavior zero-shot 2025`
- `multimodal large language models camera trap wildlife benchmark GPT-4V Gemini species behaviour accuracy`
- `BioCLIP 2 zero-shot attributes camera trap linear probe few-shot`
- `"Can Edge-Deployable Vision-Language Models Identify Species"`
- `camera trap visual question answering benchmark VLM dataset attributes weather night infrared 2026`
- `camera trap platform "natural language" search images CLIP text query Agouti OR Trapper OR TrapTagger OR "Wildlife Insights" OR EcoSecrets`
- `vision language models yes bias hallucination POPE object existence binary questions calibration`
- `Perch agile modeling bioacoustics linear classifier embeddings few examples Google`
- `Norouzzadeh deep active learning system species identification camera trap embeddings Methods in Ecology and Evolution`
- `linear probe frozen foundation model embeddings few-shot ecology images label-efficient benchmark number of examples per class`
- `Label Studio Prompts LLM auto-labeling image classification ground truth evaluation feature`
- `agentmorris speciesnet fine-tuning tutorial GPU hours images per class`

### 14.4 Open items to settle before building

1. Run the step-zero spike, including letterboxed crops for antlers.
2. Decide letterbox versus centre crop for all embeddings, given it shifts the existing
   suggestions and needs a re-embed.
3. Decide scikit-learn in `env-addaxai-base` versus a torch linear layer.
4. Decide how "cannot tell" is stored: a trainable answer value, with NULL kept for "not
   answered".
5. Decide the export shape, especially for Camtrap DP.
6. Read the second-hand papers in full.
7. Agree the UI placement that keeps this apart from SpeciesNet fine-tuning, and share the
   deployment-split validation between the two.

---

## 15. Plain English summary

Letting people type any question and get an answer is possible with vision-language models,
but on camera trap images the ones that fit on a laptop are weak, biased towards "yes", and
worse at night. The proven route, used by Google for bird and whale sounds and shown for
camera trap species in 2021, is to have the user pick a question with fixed answers, label a
few dozen examples, and fit a tiny classifier on image features the app already stores. It
needs no new model, runs in seconds on any laptop, has no settings to tune, and tells the
user how accurate it is before they trust it. AddaxAI already does something very similar
for species suggestions, so this is a sibling feature, not a new idea for the app. It is
separate from fine-tuning SpeciesNet: that makes a new portable species model from
thousands of labels and needs a GPU for hours, while this adds answers to one project from a
few dozen labels. Both can live in the app if they sit in different places and share one
honest accuracy check split by camera site. Sleeping versus resting, coat colour at night,
antler size and weather are hard or impossible from a single crop, and antler questions are
hurt further because the current crop cuts off the ends of wide boxes. Before building, spend
a day testing a few real questions on stored features to see whether the numbers hold up.

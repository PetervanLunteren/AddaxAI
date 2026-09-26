# Custom fields with optional AI suggestions in AddaxAI: investigation and future plan

Status: investigation only, no code written.
Date: 2026-09-25, revised 2026-09-26
Branch the investigation ran on: `claude/lay-of-the-land-doulv4`
Repo state at time of audit: `b8b75e2` on `main`, VERSION `0.0.0-dev`

This document is the raw material for a proper implementation plan. It holds the
original brief, the questions to answer, what the audit of the repo found, what the
literature and other platforms do, the three approaches that were compared, why the
simplest one won, how it relates to the planned SpeciesNet fine-tuning feature, what to
deliberately leave out, the risks, and the sources. It is written to be picked up cold
months later without rerunning the investigation.

It stops short of detailed UI design and field-level schema detail. What matters here is
which approach is sound, which of the example questions it can and cannot answer, what the
existing code already provides, and how answers are verified without mixing with species
verification (section 9.7), which is the one UI decision it does make.

**Update, 2026-09-26, after user feedback.** The investigation started as "can the AI
answer a user's question about their images". Feedback then asked for something more basic:
fields users fill in themselves, like snow depth in centimetres or antler size as S, M or L,
with no AI involved. Section 15 argues why that should not be a second feature but the
foundation of this one, and turns the plan into **custom fields**, typed values on images
and detections, with AI suggestions as an optional switch on the fields where the method
works. A "question" from the earlier sections is now a choice field with suggestions on.
Sections 1 to 14 are left as written, because their reasoning about the AI part still
holds; where section 15 replaces a part of them (the scope in 9.1, the storage in 9.2, the
effort in 12), they point to it. The file keeps its name so existing links still work.

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

Follow-ups, after the first version of this document was written, also verbatim:

> I see that you do the full verification in the labels page. How to do that when it
> regards a full image question?

> How would one go about verification if there are N questions? Basically the labels are
> for species labels, right? The counts are for counts and sex age behaviour. Wouldn't it
> mix up if we use the same labels page for question answering? When is one verified? What
> if the user had 50 % of the species labels verified and then only 10 % of the question,
> and another question of 40%? Wouldn't it make more sense to have a question page with the
> same dry helpers as in the labels page but not have it mix up??

> Ive also got the feedback that folks want to have the option to add custom values to
> images/events/detections like snow depth (input int cm), antler size (input S/M/L), etc,
> without having to have it pre-filled by AI/embeddings. How would that work? How could we
> combine this? Would it be good UI UX to combine these? And make the AI suggestions
> pre-filled optional?

> Would users need to go through all their data again for every question? Is that OK? How
> does timelapse do it?

The first two are answered in sections 9.7 and 9.8. The third changed the scope of the
plan and is answered in section 15. The fourth is answered in section 15.21, which adds an
entry mode and qualifies the "one question at a time" rule of 9.7.

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
9. How are answers verified when there are several questions, without mixing with species
   verification? (9.7)
10. How would whole-image questions be reviewed? (9.8)
11. How can users record their own values (snow depth, antler size) with no AI at all, on
    images, events and detections? (15)
12. Should manual values and AI answers be one feature, and should AI suggestions be
    optional? (15.3, 15.4)
13. Does every new field or question mean another pass over all the data, and how does
    Timelapse avoid that? (15.21)

## 3. Goals

- Let users record their own structured values on images and detections, such as snow
  depth or antler size, with no AI involved (added after feedback, section 15).
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

### 4.7 Verification is already one flag per job

The app keeps a separate confirmation flag for each kind of review, and never lets one
imply another:

- **Labels** confirms species. `Detection.verified` (`backend/app/models/detection.py:98`),
  rolled up to `File.verified` (`backend/app/models/file.py:103`), which the file's docstring
  calls "the single user-facing verification flag": badges, filters, navigation and the event
  MaxN rollup all read it.
- **Counts** confirms counts and demographics. `Event.confirmed`
  (`backend/app/models/event.py:66`), documented as "distinct from Detection.verified" and
  cleared automatically when the event's species or count set changes.

Custom questions are a third job and need the same separation. Showing question answers on
the Labels page would give `verified` two meanings and corrupt the rollup everything else
reads. Section 9.7 has the design.

The Labels page is built from parts that are already separate components:
`frontend/src/components/verify/{CropGrid,FilesGrid,VerifyFilterBar,FilterChips,ConfidenceRangeFilter,BulkActionBar,SuggestionsToolbarPill}.tsx`
plus `grid-selection.ts`, `shortcuts.ts` and `labels-filters.ts`. `FilesTab` shows one tile per
whole frame, empty frames included, and `CropGrid` one tile per detection.

### 4.8 What does not exist

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

Superseded by section 15, which widens v1 to manual fields on images and detections. What
follows still describes the scope of the AI suggestions within it.

Detection-level questions only, answered from the existing crop embeddings. Closed answer
sets of two to about six answers. One project at a time. Images and video best frames,
which is what already has vectors.

### 9.2 Storage

Superseded by section 15.13, which generalises these two tables to typed fields and values
on both images and detections. Kept for the reasoning.

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
6. Review on the question's own review page (section 9.7): filter by answer and score,
   least sure first, confirm or correct. Corrections become labels for the next fit.

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

### 9.7 Verification with N questions

A question's answers are confirmed on a Questions review page, never on the Labels page.
Labels stays for species, Counts for counts and demographics, Questions for answers.

**What "confirmed" means.** Confirmation belongs to one answer: the pair (detection,
question), stored as the `source = human` row in `detection_answers`. Nothing rolls up across
questions. Question answers never read or write `Detection.verified`, `File.verified` or
`Event.confirmed`, and those never confirm an answer. A project can have species 50%
verified, posture 10% confirmed and coat colour 40% confirmed, and those are three
independent numbers shown in three places. The answers a user gave while teaching count as
confirmed.

**When a question is done.** When the user says so. The question card shows confirmed
count and percentage, how many unconfirmed answers remain, and how many of those the model
was unsure about. No global "done" state is derived.

**How it interacts with species labels.**

- Scope follows the current species label. A question scoped to wild boar covers what is
  labelled wild boar now. A crop relabelled to red deer drops out of scope: its answer is
  hidden, not deleted, and returns if the label is reverted. A crop marked as a false
  detection leaves every question.
- Species confirmation is not a prerequisite. The review page offers a "species verified
  only" filter for users who want answers on top of confirmed species; making it mandatory
  would block the feature on the slowest step.
- Each tile shows the species label and whether it is verified, as context only.

**One question at a time.** The review page has a question selector at the top. Keys 1 to 9
and 0 map to that question's answers and each tile carries one answer badge. Several badges
per tile, or answering several questions per crop in one pass, looks efficient but forces a
change of judgement on every tile; one question per pass is faster on the keyboard and more
accurate.

This rule holds for reviewing AI suggestions and for bulk fills. It is wrong for entering
manual values, where the expensive part is looking at the image and several fields should be
filled in the same look. Section 15.21 adds an entry mode for that.

**Shared parts, not copies.** The page is assembled from the Labels page's components
(section 4.7): `CropGrid` for animal questions, `FilesGrid` for whole-image ones, the filter
bar and chips, the confidence range filter, `BulkActionBar`, the grid selection store, the
shortcut table, and the `SuggestionsToolbarPill` pattern for the "corrections since last
round, retrain" pill. Some of them assume a species label today and need the item type and
the answer set as inputs instead. That generalisation is the main frontend cost and is the
shared-helper work the conventions ask for; the alternative, a second copy of the grid, is
what they forbid.

### 9.8 Whole-image questions, later

Out of scope for v1 (section 10), but the design leaves room. The question setup would ask
"about the animal" or "about the whole image", and the review page would show `FilesGrid`
tiles instead of crops, with the same filters and bulk actions. Before that works:

- It needs a second embedding pass on full frames, one per file, empty frames included,
  which are usually most of a project. Per item it costs about the same as a detection
  embedding; videos get one frame.
- Full frames must be letterboxed. A centre crop of a 16:9 frame drops about 44% of its
  width, which for a scene question is much of the evidence.
- Review needs a time-ordered sort within a deployment, so a run of consecutive frames with
  the same scene can be selected and confirmed in one action.
- Weather specifically is still better served by historical weather data for the
  deployment's coordinates and timestamp. Whole-image questions pay off for things that are
  visible in the scene: snow cover, flooding, vegetation state, a knocked-over camera.

---

## 10. What to deliberately not build

Section 15.17 adds to this list.

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

This covers the AI part only. Section 15.19 adds the manual fields and revises the total to
about four and a half weeks.

Backend, roughly a week:

- Migration for the two tables and the purge path: half a day
- Training and prediction subprocess script, plus tests: one day
- Router, job and worker: one day
- Accuracy split by deployment, shared with the confusion matrix endpoint: one day
- Exports: half a day

Frontend, about a week and a half: question setup, the labelling view, the accuracy panel,
and the Questions review page. The extra half week is generalising the shared Labels
components (section 9.7) to take an item type and an answer set, rather than copying them.

Docs, half a day: one page under `docs/docs/guides/` with the honest limits, including
which kinds of questions do not work.

Total, about three weeks for a v1. The crop change, if the spike justifies it, is
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
`frontend/src/pages/ConfusionMatrixPage.tsx` (header),
`backend/app/models/{detection,file,event}.py` (the verification fields),
`backend/app/api/crud/event.py` (`_RegroupCarry`, `_snapshot_event_carry`),
`frontend/src/components/verify/{FilesTab,FilesGrid,EventCollage}.tsx` (headers) and the
directory listing of `frontend/src/components/verify/`, and the `future-plans/` documents
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
8. List which Labels components assume a species label, and how each takes an item type
   and answer set instead.
9. Decide what a question's scope stores (taxonomy ids, category) so relabelling moves
   detections in and out of scope without deleting answers.

Section 15.20 adds five more.

---

## 15. Custom fields: why the plan widened

Sections 1 to 14 plan a feature where the AI answers a user's question. After they were
written, feedback came in that changes what the feature is for. This section records the
change, argues it in full, and gives the design that replaces parts of section 9. The earlier
sections are left standing because their reasoning about the AI part still holds; where this
section supersedes them, they point here.

### 15.1 The feedback

Quoted in section 1: users want to add their own values to images, events and detections,
such as snow depth as a whole number in centimetres or antler size as S, M or L, **without**
having them filled in by AI. The follow-up question was whether that should be combined with
the AI questions, and whether the AI pre-fill should become optional.

This is a different need from the one the investigation started from. The original brief
was "can the AI answer my question". The feedback is "let me record my own observations in
a structured way". The second is more basic, and the first turns out to be a special case of
it.

### 15.2 What changes, in one sentence

"Custom questions answered by AI" becomes **custom fields**: user-defined, typed values on
images and detections that people fill in, with **AI suggestions as an optional switch** on
the fields where the method works. A question from sections 1 to 14 is now simply a choice
field with AI suggestions switched on.

### 15.3 Why combine instead of building two features

The alternative is two features: a manual "custom attributes" feature for typed values, and
the AI "questions" feature as planned. That was considered and rejected, for five reasons.

1. **One idea for the user.** From the user's side both are "extra things I record about my
   data". Asking them to decide up front whether a field is a question or an attribute, and
   then look in two places, is a distinction the software cares about and the ecologist does
   not. With one concept, whether AI helps is a property of a field, visible on its card and
   switchable later, not a different part of the app.

2. **Manual values are the AI's training data.** This is the strongest argument. Someone who
   has typed antler size by hand for 200 detections has done exactly the labelling that the
   teach round in section 9.3 asks for. In one system, switching AI on for that field starts
   from those 200 labels and can go straight to the check step. In two systems those values
   sit in a table the classifier cannot see, and the user is asked to label the same crops
   again. The reverse also holds: a field that started with AI and whose suggestions the user
   corrected is a manual field with a head start.

3. **One storage model, one export shape, one review page.** Two features would mean two
   tables with near-identical columns, two export paths into the same CSVs, two progress
   indicators, and two review grids assembled from the same components. That is the drift
   the conventions warn about (DRY, shared helpers): the second copy gets the bug fix a month
   late. One values table and one review page carry both.

4. **The manual feature is the foundation anyway.** An AI suggestion is useless until a
   person can confirm or override it, which requires an input for that value, storage for a
   human value, and a review flow. Those are precisely the parts of a manual feature. Built
   the other way round (AI first, manual later) the manual part would be bolted onto a
   design that assumed every value comes from a classifier.

5. **It fixes the scope problem of the original plan honestly.** Sections 8.2 and 9.8
   concluded that weather and other numeric or whole-image values are poorly served by
   pixels. Under the old framing those needs were simply out of scope. Under the new one
   they are served: snow depth becomes a manual number field today, with no AI promised.

What combining costs: the setup dialog and the list page must handle fields with and
without AI without either looking half-finished. Section 15.10 covers how.

### 15.4 Why AI suggestions are optional and off by default

- **Many fields cannot use AI at all.** Numbers, free text and event-level values have no
  honest AI path (section 15.8). A default of "on" would be a default that is invalid for
  half the field types.
- **AI needs setup the user may not want.** Switching it on commits the user to a teach
  round of at least 15 examples per answer and a check step. Someone who wants to type 40
  snow depths should never see that.
- **Trust.** Every AI value is a suggestion that has to be confirmed (section 9.7). A user
  who did not ask for suggestions should not find thousands of unconfirmed values in their
  project.
- **The switch can be flipped later at no loss.** Off to on reuses every manual value as a
  training label. On to off hides AI suggestions and leaves every human value untouched. So
  the cautious default costs nothing.

### 15.5 Precedent

Manual custom fields are an established pattern. Timelapse, widely used for camera trap
review, builds each project on a template of user-defined data fields that people fill in
per image (https://github.com/saulgreenberg/Timelapse). Its website and guides were blocked
from the investigation session, so what follows was read from its source code instead
(commit `4c42a6b`, 2026-09-21):

- Field types are the constants Note, MultiLine, AlphaNumeric, Choice, FixedChoice,
  MultiChoice, Counter, Flag, IntegerAny, IntegerPositive, DecimalAny and DecimalPositive.
- Each field carries a default value and a `Copyable` setting (`DataTables/ControlRow.cs`).
- As far as the source shows, fields belong to the file. There are no per-detection fields;
  recognition boxes are displayed and queried, not annotated with template fields.

How Timelapse keeps data entry to one pass is in 15.21. So the manual half of this design
copies something proven; the optional AI layer on top is the new part, and it is the part
sections 5 to 8 justify. Per-detection fields are also new relative to Timelapse, and they
are what makes AI suggestions possible, since the classifier works on detection crops.

### 15.6 The field model

A field has:

- **Name.** Shown in the UI and used for the export column, for example `snow_depth_cm`,
  `antler_size`, `posture`.
- **Level.** Image (file) or detection. Events are handled as a view, see 15.7.
- **Type.** Choice (a fixed, ordered list), whole number, decimal, yes/no, or short text.
- **Constraints.** Unit and minimum and maximum for numbers; the options for choice fields;
  a maximum length for text. The API validates every write against them and rejects
  anything outside, the same way `observation_attributes.py` guards sex, life stage and
  behaviour. No silent coercion (convention 1).
- **Scope** (optional). Which detections a detection field applies to, by species or
  category, following the current label as in 9.7.
- **AI suggestions.** Off by default. Only offered for choice and yes/no fields at detection
  level in v1.

Two worked examples:

| | Snow depth | Antler size |
|---|---|---|
| Level | Image | Detection |
| Type | Whole number | Choice |
| Constraints | cm, 0 to 300 | S, M, L, cannot tell |
| Scope | All images | Red deer, roe deer |
| AI suggestions | Not available for numbers | Available; off until the user switches it on |

### 15.7 Why there is no stored event level

The feedback asks for values on events too. The honest answer is that events in this app
are not stable enough to hold typed values, and they do not need to.

Events are regenerated. A regroup deletes a deployment's events and builds new ones, and
`_RegroupCarry` in `backend/app/api/crud/event.py` spells out what survives, "with two
different rules". Counts and the confirmed flag are claims about one exact file set, so they
carry only onto a new event with the same files; a merged or split event loses them. Notes
are free text, so they are never lost: a split copies the note to every child and a merge
joins the notes in time order, one per line.

Neither rule works for a typed value. Carrying by exact file set would silently drop a snow
depth whenever a regroup changes an event's boundaries. Carrying by overlap, as notes do,
has no answer for a merge of two events with 20 cm and 35 cm: a number cannot be
concatenated. Every choice there loses a human value or invents one.

So values are **stored only on images and detections**, which are stable. Events get a
**view** instead:

- In `EventDetailModal`, image fields appear as inputs for the whole event. Setting snow
  depth there writes the value to every file in the event.
- Displaying it reads the files: one value if they all agree, "mixed" with the range if they
  do not, empty if none is set.
- A regroup changes nothing, because nothing is stored on the event. A merged event whose
  halves had different depths simply shows "mixed", which is true.

This gives users what they asked for (set snow depth once per visit) without a third level
of storage or a carry rule that cannot be right.

### 15.8 Why fields cannot live on observation cohorts

`event_observations` rows are deleted and recreated on every relabel, threshold change and
regeneration, under the seed rule described in `DEVELOPERS.md`, "Observation cohorts". Sex,
life stage and behaviour survive that only because they are hand-carried field by field in
`PriorObs`, `_snapshot_event_carry` and `deployment_split.py`, and the docs warn that
forgetting one silently loses data. Arbitrary user fields cannot be added to that list.

They stay built-in because they map onto Camtrap DP's own columns. Per-animal attributes
that users invent, like antler size, belong on the detection, which is one animal in one
frame and is stable.

### 15.9 Where AI suggestions can be switched on

| Level | Choice / yes-no | Whole number / decimal | Text |
|---|---|---|---|
| Detection | Yes, v1 (sections 7 to 9) | No | No |
| Image | Later, needs whole-image embeddings (9.8) | No | No |
| Event (view) | No | No | No |

- **Numbers.** A regression on the embeddings could technically output a snow depth, but
  nothing in section 5 supports it, and without a snow stake in view there is nothing in the
  pixels to measure against. It would produce confident, unverifiable numbers. Not offered.
  In the setup dialog the switch is visible but disabled, with a one-line reason, so the user
  learns why instead of wondering where it went.
- **Text.** Nothing to classify.
- **Events.** There is no single thing to embed. A later version could show the most common
  detection answer per event as a summary, but not store it.

### 15.10 Keeping the two kinds of field from cluttering each other

- **The setup dialog** asks name, level, type and constraints first. The AI switch sits
  below them, only enabled when the level and type allow it.
- **A field with AI off** has no rounds, accuracy, retrain pill or teach screens anywhere.
  Its card on the list page shows only filled x of y.
- **A field with AI on** gains the teach, check, rounds and answer-all flow from the canvas
  (screens 3 to 7), and its card also shows confirmed x of y and the held-out accuracy.
- **Switching AI on** for a field that already has manual values counts those values per
  answer. If every answer has the minimum, the flow opens at the check step; if not, the
  teach screen shows which answers still need examples.
- **Switching AI off** hides all AI values of that field from grids, detail views and
  exports. Human values stay. Switching back on restores the last fitted classifier's
  suggestions or refits.

### 15.11 Where values are entered

- **Detail views, the entry mode (15.21).** Every field of the matching level appears in
  one panel: image fields in `FileDetailModal`, detection fields in `DetectionDetailModal`,
  and image fields as event inputs in `EventDetailModal` (15.7), next to the notes, counts
  and demographics that already live there. The panel stays open while stepping to the next
  item, and carries the copy and propagate helpers from 15.21. This is where most manual
  values are entered, in the pass people already make.
- **The field's review page, the review mode, for AI suggestions and bulk work.** The Questions review page from the canvas,
  generalised: pick a field, get `FilesGrid` for image fields or `CropGrid` for detection
  fields, filter to "empty" or "not confirmed", select, set. Choice fields keep the 1 to 9
  and 0 keys. Number fields put an input in the bulk bar, so a run of 30 frames from one
  morning can be set to 35 cm at once; a sort by time within a deployment makes such runs
  easy to select.
- **Progress per field.** Filled x of y for every field, plus confirmed x of y where AI is
  on. Separate from species verification and from Counts, as 9.7 requires.

### 15.12 Filled and confirmed

One flag per value, `source`, human or AI, covers both kinds of field.

- A value a person typed or picked is filled and confirmed at once.
- An AI value is filled but not confirmed until a person accepts or changes it; accepting it
  flips its source to human.
- Filled x of y counts both. Confirmed x of y counts human values only.

So manual fields have no separate confirmation step, and AI fields reuse the model from 9.7
unchanged.

### 15.13 Storage, replacing 9.2

Sketched, not specified:

- `custom_fields`: project, name, export key, level, type, constraints (unit, min, max,
  options, max length) as JSON, scope, `ai_enabled`, and for AI fields the embedding model
  id, fitted weights, held-out accuracy and fitted timestamp. Options keep a stable id per
  option, so renaming "Medium" to "M" does not orphan stored values.
- `custom_field_values`: field, `file_id` or `detection_id` (two nullable foreign keys,
  exactly one set, enforced by a check constraint), `value_num` or `value_text` (the type
  decides which), `source` (human or AI), `score` for AI values, updated timestamp.

Two explicit foreign keys rather than a generic `target_type` and `target_id`: SQLite can
cascade from a real foreign key, and the leaf-first purge in `purge_deployment_data()`
(`DEVELOPERS.md`, "Deleting analysis data") can empty this table first, before detections and
files, so their deletes find nothing to cascade to. A string target id would cascade from
nothing and leave orphans.

One row per (field, target): human and AI never coexist for the same value, because
accepting or overriding a suggestion replaces it.

### 15.14 Exports

- Detection fields: one column per field in the detection-level tables.
- Image fields: one column per field in the file-level tables.
- Event rows in `counts.csv`: the event view from 15.7, a single value or empty when mixed,
  with a documented rule.
- Where AI is on, a companion column says whether the value was confirmed, so an analyst can
  filter to human values.
- Camtrap DP is still open (14.4): the standard has no slot for arbitrary fields, and adding
  columns may break validators.

### 15.15 What stays the same

- The whole AI method: sections 5 to 8, the loop in 9.3, training in 9.4, the deployment
  split, the centre-crop finding and the step-zero spike.
- Verification per value and per field, never mixed with species or counts (9.7).
- Built-in sex, life stage and behaviour on cohorts, and the separation from `behavior`
  (9.6).
- The separation from SpeciesNet fine-tuning (8.3).

### 15.16 Terminology in the UI

The sidebar item, list page and review page become "Fields" (or "Custom fields") instead of
"Questions". "Question" only fit the AI case: nobody asks a question when typing a snow
depth. The AI part is called "suggestions", matching the existing species suggestions
(`SuggestionsToolbarPill`), so users meet one word for one idea across the app.

### 15.17 Additions to what not to build

- A stored event level, for the reasons in 15.7.
- Fields on observation cohorts (15.8).
- AI for numbers, text or events (15.9).
- Formulas or computed fields ("antler size from pixel width").
- Fields shared across projects, or a template library. Useful later, YAGNI now.
- Dates, times, coordinates or file attachments as field types.

### 15.18 Additions to risks

| Risk | Severity | Mitigation |
|---|---|---|
| Users expect event-level storage and are surprised values sit on images | Medium | The event view writes to all files and shows "mixed" honestly; say so in the docs |
| Editing a field's type or options after values exist | High, data loss | Allow adding options and renaming; block type changes and option removal while values exist |
| Setup dialog feels heavy for simple manual fields | Medium | AI switch last, disabled where not possible, nothing about rounds unless on |
| Export tables grow wide with many fields | Low | One column per field is still the most usable shape for analysts in R |
| Generalising the Labels components takes longer than planned | Medium | It was already the main frontend cost in 12; manual fields add number input, not new grids |

### 15.19 Revised effort

On top of section 12's three weeks for the AI part:

- Field model, types, constraints and validation, replacing the question tables: one day
- Inputs per type in the three detail views, including the event view: two days
- Number and text input in the bulk bar, and the time-ordered sort: one day
- Field setup dialog with levels, types and the conditional AI switch: one day
- Exports for image and detection fields, and the event view rule: half a day
- Entry-mode helpers from 15.21 (copy previous values, propagate, copy forward, the
  copyable setting): one and a half days

About a week and a half more, so roughly **four and a half weeks for a v1** that ships
manual fields at both levels and AI suggestions for detection choice fields.

The manual part could ship first and alone, in about two weeks, since it needs no
embeddings, no training and no spike. That is a reasonable order: it delivers what users
asked for now, and every value entered becomes training data for when AI suggestions land.

### 15.20 Additions to open items

1. Decide whether the manual part ships first, as 15.19 suggests.
2. Settle the event view display rule for "mixed" in the UI and in `counts.csv`.
3. Decide which edits to a field are allowed once values exist.
4. Read the Timelapse guides (blocked during this investigation) to check how its helpers
   behave at the edges, for example whether "copy to all" respects the current selection
   exactly as the source suggests, before copying the behaviour.
5. Decide the name: "Fields" or "Custom fields".
6. Decide the boundary for "copy forward to end": the end of the deployment, the end of the
   current filter, or the end of the day. Timelapse uses the end of the current set of
   selected files.
7. Decide which fields appear in the Counts event view by default: all image fields, or
   only those marked for it.

### 15.21 Entry mode: one pass for all fields

**The problem.** The review page in 9.7 shows one field at a time. For AI suggestions that
is right. For manual values it would mean one pass over the data per field: a project with
snow depth, antler size and a collar flag would be walked three times. Opening and looking
at an image is the expensive part of manual annotation; once someone is looking, filling
three fields costs seconds. So N passes for N fields is not acceptable.

**How Timelapse avoids it**, read from its source (15.5):

- All template fields sit in one data entry panel beside the image. The user looks once and
  fills every field before moving on. A new field does not create a new pass; it rides along
  in the pass that already happens.
- **Copy previous values**: a menu item and button, shortcut C, that copies "selected data as
  recorded on the previous file", only for fields marked `Copyable` (`TimelapseWindow.xaml`,
  `MenuItemCopyPreviousValues`).
- Per-field context menu (`ControlsDataEntry/DataEntryHandler.cs`, around line 180):
  "Propagate from the last non-empty value to here" (or last non-zero, for counters), "Copy
  forward to end", described as copying "from this file to the last file in this set", and
  "Copy to all" for the current selection.
- An overview grid of thumbnails where a field is edited for every selected file at once,
  showing an ellipsis when the selected files disagree.
- Fields can be populated from file metadata or episode data in bulk.

**What AddaxAI adopts.** Two modes, each with a clear job.

1. **Entry mode, all fields in one pass.** The detail views show every field of their level
   in one panel (15.11), and the panel stays open while stepping to the next or previous
   item. It gets the Timelapse helpers: copy previous values, propagate from the last
   non-empty value, copy forward to the end of the deployment, and a per-field copyable
   setting. For snow depth this means one entry per change in the snow, propagated, not one
   per image.

2. **Review mode, one field at a time.** The review page from 9.7, for reviewing AI
   suggestions, for filling a field that was added late, and for bulk-setting runs of
   frames.

**The important part: fields ride along with passes users already make.** Users already step
through events on the Counts page to confirm counts, sex, life stage and behaviour. If the
event view also shows the image fields (written to the event's files, 15.7) and the
detection fields of the animals in it, then filling them costs no extra pass. That page is
AddaxAI's closest equivalent of the Timelapse data entry panel, and it is where most manual
values will be entered.

**When a second pass is still needed, and why that is acceptable.** Only when a field is
added after the data was already reviewed. Three things keep it cheap:

- Scope: antler size only appears on deer detections, not on every detection.
- The review grid with propagation: snow depth for a deployment is a handful of entries
  copied forward.
- AI suggestions for choice fields, which turn a full pass into a few dozen examples plus
  review of the unsure ones.

**What this changes elsewhere.** The "one question at a time" rule in 9.7 now applies to the
review mode only (noted there). The effort in 15.19 grows by a day and a half for the
helpers. Nothing changes in storage: entry mode and review mode write the same values table.

---

## 16. Plain English summary

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
honest accuracy check split by camera site. Answers are reviewed on their own Questions
page, one question at a time, built from the same grid, filter and keyboard parts as the
Labels page, and each question keeps its own confirmed percentage that never mixes with
species verification or with other questions. Sleeping versus resting, coat colour at night,
antler size and weather are hard or impossible from a single crop, and antler questions are
hurt further because the current crop cuts off the ends of wide boxes. Before building, spend
a day testing a few real questions on stored features to see whether the numbers hold up.

The plan has since grown on user feedback. People also want to record their own values,
like snow depth in centimetres per image or antler size per animal, with no AI involved.
Rather than build that as a second feature, it becomes the base of this one: custom fields
that users define and fill in on images and detections, with AI suggestions as an optional
switch that is off by default and only offered where it genuinely works, which in v1 means
choice fields on detections. Combining them means one place, one kind of storage and one
export shape, and every value someone types by hand becomes training data if they later
switch suggestions on. Events get no stored values of their own, because the app rebuilds
events and a number cannot survive a merge; setting snow depth on an event writes it to the
event's images instead. The manual part is the cheaper half, about two weeks, and could
ship first; the whole thing is roughly four and a half weeks.

Adding fields must not mean walking through the data once per field. Timelapse, the
established tool for this, avoids that by showing every field beside the image so one look
fills them all, with helpers to copy values from the previous image or carry them forward.
AddaxAI will do the same in its existing detail views, above all the Counts page where people
already step through every event, so new fields ride along in a pass that happens anyway. The
one-field-at-a-time page stays for reviewing AI suggestions and for bulk fills. Only a field
added late needs a second look at old data, and scope, bulk editing and AI suggestions keep
that short.

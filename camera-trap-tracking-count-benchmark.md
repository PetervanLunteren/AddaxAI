# AddaxAI camera-trap tracking count benchmarks

This document defines three related experiments:

1. **Experiment 1: video FPS and track-count benchmark** - determine how detector sampling FPS affects tracking-based animal counts in camera-trap video.
2. **Experiment 2: still-image event tracking benchmark** - determine whether tracking/temporal association across camera-trap photographs improves animal counts, and how performance changes with inter-image interval.
3. **Experiment 3: tracker and association optimisation** - reuse the fixed data and cached detections from Experiments 1 and 2 to compare tracker packages, association mechanisms, real time-delta handling and tracker settings, with the goal of improving predicted animal counts.

---

# Experiment 1: video FPS and track-count benchmark

## Purpose

This document describes a future experiment for evaluating the animal-counting mechanism in AddaxAI camera-trap videos.

The experiment should be performed **after the video tracking implementation is complete and stable**. It is intended as a benchmark for the final counting behaviour, not as an isolated tracker benchmark.

The central question is:

> **How accurately can AddaxAI estimate the number of animals in a camera-trap video by counting tracker-generated tracks, particularly when the detector is run at a low frame rate such as 1 FPS?**

The target benchmark contains approximately **2,000 camera-trap videos or ecological video events** with human-verified information about the animals present.

The proposed benchmark combines three datasets:

| Dataset | Planned sample | Main contribution |
|---|---:|---|
| PanAf500 | 500 videos | Dense frame-level tracks and within-video individual identities |
| MammAlps | 397 ecological events | Wild Alpine mammals, explicit human-annotated group size, dense tracks |
| SA-FARI | 1,103 videos | Large taxonomic, geographic and behavioural diversity |
| **Total** | **2,000** | Diverse camera-trap benchmark |

If PanAf500 video access is unavailable, replace its 500 videos with additional suitable SA-FARI videos. The benchmark should remain close to 2,000 samples.

---

# 1. Background and motivation

## 1.1 The counting problem in camera-trap video

AddaxAI processes camera-trap videos using an object detection model followed by a multi-object tracker.

Conceptually:

```text
video
  ↓
sample frames
  ↓
animal detector
  ↓
bounding-box detections
  ↓
multi-object tracker
  ↓
tracks
  ↓
suggested animal count
```

A track links detections that the tracker believes belong to the same animal across consecutive sampled frames.

The most practical automated count for ordinary terrestrial camera-trap video is therefore:

```text
suggested count = number of animal tracks
```

This is intentionally **not described as a verified number of unique biological individuals**.

True individual recognition is difficult or impossible for many species because individuals do not have reliably distinguishable markings. An automated system usually cannot know with certainty whether an animal that disappears and reappears is the same animal.

The track count is instead used as an **AI-generated suggestion**.

The user can optionally verify or correct that count.

This distinction is important:

```text
AI track count ≠ guaranteed number of unique individuals

AI track count → suggested count
human review → verified count
```

The product does not require the automated estimate to be perfect. It should be sufficiently accurate that it saves researchers time during verification.

---

# 2. Why this benchmark is needed

## 2.1 Camera-trap detection is often performed at low FPS

Wildlife detection models can be computationally expensive.

For large camera-trap datasets it is common and attractive to analyse video at a relatively low sampling rate, for example:

```text
1 FPS
```

instead of processing every frame of a 25 or 30 FPS video.

At 1 FPS, consecutive detections are one second apart.

That is often sufficient for species detection, but it makes tracking more difficult.

For example:

```text
0 s     deer detected at x=150
1 s     deer detected at x=430
2 s     deer detected at x=700
```

The tracker must determine that these detections belong to the same animal despite substantial movement between samples.

---

## 2.2 Main expected failure mode: track fragmentation

Reducing FPS can cause one real animal to be represented by multiple tracks.

Example:

```text
Real animal A

0 s      track 1
1 s      track 1
2 s      tracker loses association
3 s      track 2
4 s      track 2
```

The system could therefore report:

```text
true visible animals = 1
track count = 2
```

This produces an **overcount**.

This is expected to be one of the most important failure modes of the proposed counting mechanism.

---

## 2.3 Other failure modes are possible

Low FPS does not exclusively cause fragmentation.

When several animals are visible, sparse detections can also make data association ambiguous.

Possible outcomes include:

### Track fragmentation

```text
1 real animal
→ 2 or more tracks
→ count too high
```

### Identity switch

```text
animal A → track 1
animal B → track 2

later:

animal A → track 2
animal B → track 1
```

The identities are wrong, although the total number of tracks may remain correct.

### Incorrect association or effective merging

Two different animals may under some circumstances become associated with the same continuing track.

This can result in:

```text
2 real animals
→ 1 effective track
→ count too low
```

For AddaxAI, however, the primary performance measure is **not classical tracking identity accuracy**.

The important product question is:

> Does the final suggested animal count agree with the human-verified count?

---

# 3. Goal of the experiment

The main goal is to determine whether track count is a sufficiently reliable automated suggestion for camera-trap videos.

The benchmark should answer:

1. How often does the number of generated tracks equal the verified animal count?
2. How large is the typical counting error?
3. Does the system systematically overcount because of track fragmentation?
4. How often does it undercount?
5. How strongly does FPS affect counting accuracy?
6. Is **1 FPS** good enough for practical use?
7. Does performance differ strongly between species, body sizes, movement speeds, numbers of animals, day/night footage or environmental conditions?
8. Is there a tracker configuration that works particularly well with sparse camera-trap detections?
9. Is the gain from increasing detector FPS large enough to justify the extra compute cost?

The final decision should be based on the **counting task**, rather than only on standard multi-object tracking metrics.

---

# 4. Primary hypothesis

The working hypothesis is:

> **A detector + tracker running at 1 FPS will produce a useful suggested animal count for most ordinary camera-trap videos, but errors will become increasingly dominated by track fragmentation as movement speed, occlusion and group size increase.**

A secondary hypothesis is:

> Increasing detector FPS will improve exact-count accuracy, but there will be diminishing returns, and the computational cost of high-FPS wildlife detection may not justify the improvement.

The benchmark should test these assumptions rather than taking them for granted.

---

# 5. Experimental unit and ground truth

The preferred experimental unit is one independent camera-trap video.

For datasets organised around ecological events rather than single clips, one event may be used as one experimental unit when that is how the human annotation defines group size.

Each benchmark item should ultimately produce a row similar to:

```csv
dataset,sample_id,species,true_count,predicted_track_count,fps,verified
SA-FARI,sav_000123,red_deer,3,4,1,true
```

Where possible, retain additional metadata:

```text
location
day/night
video duration
original FPS
resolution
species
taxonomic group
ground-truth count
number of annotated tracks
maximum simultaneous animals
movement/behaviour
occlusion information
dataset split
```

---

# 6. Important definition of ground truth

Do not blindly assume that every annotation track corresponds to a unique animal.

The coding agent must inspect the annotation semantics for each dataset.

The desired target is:

> **Number of human-verified animals represented in the video or event.**

Preferred hierarchy:

1. Explicit human-annotated `group_size` or individual count
2. Number of persistent human-assigned individual IDs
3. Number of annotation masklets/tracks only if the dataset documentation confirms these correspond one-to-one with individuals for the relevant sample

This matters because an individual that leaves and re-enters a frame may have:

```text
one biological identity
multiple track segments
```

The benchmark must not accidentally use track fragments as the ground truth for an experiment that is intended to measure track fragmentation.

---

# 7. Proposed 2,000-sample benchmark

The initial target is:

```text
PanAf500       500
MammAlps       397
SA-FARI       1103
-----------------
Total         2000
```

This composition is attractive because the datasets complement each other.

PanAf500 provides extremely detailed ape annotations.

MammAlps provides explicit ecological group-size labels and Alpine wildlife.

SA-FARI provides scale and much broader species, habitat and geographic diversity.

The exact composition can change if access or annotation constraints make a subset unsuitable.

Do **not** force a dataset into the benchmark merely to hit an exact sample count.

Quality of ground truth is more important than the exact value 2,000.

---

# 8. Dataset 1: PanAf500

## Overview

PanAf500 is a camera-trap video dataset of wild great apes.

It is part of the broader PanAf/PanAf20K ecosystem.

PanAf500 contains:

- 500 videos
- wild chimpanzees and gorillas
- approximately 180,000 annotated frames
- approximately 24 FPS footage in the original PanAfrican material
- frame-by-frame bounding boxes
- species labels
- behavioural labels
- **intra-video individual identities**

The individual identities make this dataset particularly useful for the current experiment.

A video can contain multiple animals, and the human annotations identify which bounding boxes belong to the same individual within that video.

This allows a ground-truth count to be calculated as approximately:

```text
true_count =
number of distinct human-assigned individual IDs in that video
```

The annotation semantics must still be verified programmatically before using this rule.

## Why PanAf500 is useful

It is unusually well matched to this experiment because the tracker can be evaluated against human identity assignments.

Example:

```text
Human annotations:
ID1
ID2
ID3

true count = 3

AddaxAI at 1 FPS:
track_1
track_2
track_3
track_4

predicted count = 4

count error = +1
```

This directly measures fragmentation.

## Dataset information

Project:

https://obrookes.github.io/panaf.github.io/

PanAf500 is described as the 500-video fine-grained subset of PanAf.

Associated public metadata and annotations are available through the PanAf project and University of Bristol datasets.

Relevant University of Bristol dataset:

PanAfrican2020

DOI:

```text
10.5523/bris.jh6hrovynjik2ix2h7m6fdea3
```

The public annotation release contains approximately 180,000 XML annotation files covering the 500 videos.

Another relevant annotation resource:

PanAfrican2019 Video Annotations

DOI:

```text
10.5523/bris.1v9op9lc6zi5g25kkwa5smb3vq
```

## Important access caveat

Historically, the University of Bristol annotation datasets did **not necessarily include the original video files themselves**.

The underlying video footage was held by the Pan African Programme.

More recent PanAf20K tooling and research repositories reference workflows for downloading/unpacking PanAf data, so access may have changed.

Therefore, before implementing anything else:

### First action for PanAf500

Verify that the **actual 500 source videos are currently publicly downloadable under terms compatible with this benchmark**.

Do not assume that public annotations imply public video binaries.

Useful starting points:

```text
https://obrookes.github.io/panaf.github.io/
https://github.com/obrookes/panaf.github.io
```

A useful reference implementation is also:

```text
https://github.com/ecker-lab/dap-behavior
```

Its documentation includes a PanAf500 preparation step similar to:

```bash
python dap_behavior/data/eval/panaf500.py data/panaf/panaf500/ data/
```

This can help identify expected directory structure and annotation conversion.

## Expected annotations

Expect frame-level annotation data containing combinations of:

```text
video/frame identifier
bounding box
species
individual identity
behaviour
```

The coding agent should build a parser that:

1. groups annotations by video
2. extracts persistent individual IDs
3. verifies that IDs have the expected semantics
4. calculates one ground-truth count per video
5. records species composition
6. detects potentially ambiguous/crowd annotations

## Planned sample

Use all:

```text
500 videos
```

provided that:

- the video files are accessible
- the individual IDs can be interpreted reliably
- the relevant license/terms permit the benchmark

## Limitation

PanAf500 is taxonomically narrow.

It mostly represents great apes, so it should **not** be used alone to determine the appropriate FPS for general wildlife camera trapping.

That is why MammAlps and SA-FARI are included.

---

# 9. Dataset 2: MammAlps

## Overview

MammAlps is a multi-view wildlife video dataset collected with camera traps in the Swiss National Park.

It contains:

- over 14 hours of wildlife video
- nine camera traps
- three monitoring sites
- audio
- segmentation maps
- approximately 8.5 hours of densely annotated individual tracks
- species labels
- behaviour labels
- ecological events

The dataset was introduced for CVPR 2025.

Project repository:

```text
https://github.com/eceo-epfl/MammAlps
```

Zenodo dataset:

```text
https://zenodo.org/records/15588220
```

The DOI family is associated with:

```text
10.5281/zenodo.15040900
```

At the time this document was written, the Zenodo v3 record included:

```text
mammalps_v1.zip
dense_annotations_v1.zip
README.md
```

## Why MammAlps is especially useful

MammAlps Benchmark II explicitly includes:

> number of individuals / group size

as a human-labelled ecological target.

It contains:

```text
397 multi-view long-term ecological events
```

including false-positive camera triggers.

This provides a direct ecological ground truth and avoids having to infer biological counts entirely from computer-vision track annotations.

## Download

The project repository previously documented commands equivalent to:

```bash
wget https://zenodo.org/records/15040901/files/mammalps_v1.zip
```

For the latest release, prefer using the current Zenodo record rather than hard-coding an old record number.

Current reference at document creation:

```text
https://zenodo.org/records/15588220
```

The main archive is large:

```text
~87.9 GB
```

The dense annotation archive is much smaller:

```text
~13.5 MB
```

Download both if dense tracking analysis is required.

## Expected directory structure

The main dataset is documented roughly as:

```text
MammAlps/
└── mammalps_v1/
    ├── benchmark_1/
    │   ├── audios/
    │   ├── clips/
    │   ├── segmaps/
    │   └── annotations/
    ├── benchmark_2/
    │   ├── events_processed/
    │   └── annotations/
    ├── raw_videos/
    ├── reference_scenes/
    ├── events2files.json
    ├── raw_videos_mammalps_v1.csv
    └── file_list.txt
```

Dense annotations are supplied separately in newer releases.

## Benchmark II annotations

Benchmark II annotations include fields corresponding to:

```text
event
activities
species
meteorological condition
group_size
```

The documentation gives a conceptual structure similar to:

```text
event_path  activities  species  met_condition  group_size
```

The exact parser should follow the current release rather than assuming column positions.

## Event mapping

Use:

```text
events2files.json
```

to map an ecological event to the raw source videos.

Conceptually:

```json
{
  "E123": ["file_a", "file_b", "file_c"]
}
```

This is important because Benchmark II may represent one ecological event using footage from more than one camera.

## Individual track annotations

The dataset naming convention includes:

```text
ID<individual_id>
T<segment_id>
```

This is valuable.

An individual may leave the field of view and return.

That can produce:

```text
ID1_T1
ID1_T2
```

These are multiple track segments belonging to the **same human-labelled individual**.

This is exactly the distinction that matters for AddaxAI's fragmentation experiment.

The coding agent must preserve `ID` separately from `T`.

Do not use:

```text
number of segments
```

as the ground-truth animal count.

Use the explicit group-size labels and/or unique biological IDs as appropriate.

## Planned sample

Use all:

```text
397 Benchmark II ecological events
```

provided that the raw footage corresponding to each event can be reconstructed unambiguously.

### Important implementation decision

Because AddaxAI ordinarily operates on a **single video file**, while MammAlps Benchmark II contains multi-view events, the coding agent should evaluate two possibilities:

**Option A, preferred for ecological ground truth**

Treat each MammAlps event as the unit and reproduce the event-level group-size task.

**Option B, preferred for exact AddaxAI video equivalence**

Use individual raw videos with dense individual annotations and derive unique-individual counts per raw video.

The final choice should be documented.

If using event-level samples, do not pretend these are equivalent to ordinary single-file camera-trap clips. Keep a field such as:

```text
sample_type = mammalps_event
```

so results can be analysed separately.

---

# 10. Dataset 3: SA-FARI

## Overview

SA-FARI stands for:

> Segment Anything in Footage of Animals for Recognition and Identification

It is a large multi-animal tracking dataset built from real camera-trap video.

It contains approximately:

- 11,609 camera-trap videos
- footage collected over roughly 10 years
- 741 locations
- four continents
- 99 species categories
- approximately 46 hours of densely annotated footage
- 16,224 annotated masklet identities
- approximately 942,702 bounding boxes / segmentation masks / species annotations

This is the most important dataset in this benchmark for **taxonomic and environmental diversity**.

Project page:

```text
https://www.conservationxlabs.com/sa-fari
```

Hugging Face annotations:

```text
https://huggingface.co/datasets/facebook/SA-FARI
```

Access is gated on Hugging Face and requires accepting the dataset terms and sharing contact information.

The dataset is described as:

```text
CC BY-NC 4.0
```

Check the current license and terms before use.

## Video storage

The SA-FARI dataset card states that videos and preprocessed frames are stored in the public camera-trap storage location under a structure resembling:

```text
sa_fari/
├── sa_fari_test_tars/
│   ├── JPEGImages_6fps/
│   └── videos/
├── sa_fari_test/
│   ├── JPEGImages_6fps/
│   └── videos/
├── sa_fari_train_tars/
│   ├── JPEGImages_6fps/
│   └── videos/
└── sa_fari_train/
    ├── JPEGImages_6fps/
    └── videos/
```

Important distinction:

```text
videos/
```

contains original full-FPS videos.

```text
JPEGImages_6fps/
```

contains videos sampled into frames at 6 FPS for annotation compatibility.

For the AddaxAI experiment, use the **original videos whenever possible**, because the experiment itself needs to control the detector sampling FPS.

Do not use only the pre-sampled 6 FPS JPEGs if the intention is to compare 0.5, 1, 2, 5 FPS, etc. from a common source.

## Annotation files

The Hugging Face repository contains annotation files similar to:

```text
annotation/
├── sa_fari_test.json
├── sa_fari_test_ext.json
├── sa_fari_train.json
└── sa_fari_train_ext.json
```

The `_ext` versions add metadata such as:

```text
video_num_frames
video_fps
video_creation_datetime
location_id
taxonomy
```

Use the extended files where practical.

## Annotation format

The format is broadly similar to YTVIS.

Top-level sections include:

```text
info
videos
annotations
categories
video_np_pairs
```

Video records include fields such as:

```text
id
video_name
file_names
height
width
length
```

Annotation records include fields such as:

```text
id
segmentations
bboxes
areas
iscrowd
video_id
height
width
category_id
noun_phrase
```

Each positive annotation represents a temporally persistent masklet/track identity.

The coding agent must verify whether one masklet always corresponds to one unique animal for the duration of a video and how disappear/re-entry cases are handled.

Do not simply assume:

```text
number of annotation objects = number of biological individuals
```

until verified against the dataset documentation and examples.

## Planned sample

Use:

```text
1,103 videos
```

for the default 2,000-item benchmark.

If PanAf500 cannot be downloaded, increase SA-FARI to:

```text
1,603 videos
```

to maintain a total near 2,000.

If MammAlps event semantics make direct comparison inappropriate, SA-FARI can also provide additional single-video samples.

---

# 11. SA-FARI sampling strategy

Do not simply sample the first 1,103 files.

The purpose of using SA-FARI is diversity.

Use a reproducible stratified sampling strategy.

Set a fixed random seed, for example:

```python
RANDOM_SEED = 42
```

Try to maximise diversity across:

```text
species
location
camera
continent/domain if available
video duration
number of animals
day/night
movement pattern
body size
group size
```

The exact fields depend on available metadata.

## Avoid excessive dominance by common species

If one species represents a very large proportion of the dataset, cap its contribution.

Possible strategy:

1. determine the number of suitable annotated videos per species
2. assign a reasonable per-species cap
3. prioritise species with multiple examples
4. randomly sample within species
5. fill any remaining slots from the full eligible pool

Keep the selection reproducible and export the selected video IDs.

Example:

```text
benchmark_manifest.csv
```

---

# 12. Inclusion criteria

A sample should normally be included only if:

- the underlying video is available
- annotations can be mapped unambiguously to the video
- there is a defensible human-verified animal count
- the sample contains ordinary camera-trap-style footage
- annotation quality is sufficient for count comparison
- video decoding succeeds
- species information is available or recoverable

Do not require that all videos contain animals.

A small number of verified empty/false-trigger videos can be useful because the desired output is then:

```text
0 tracks
```

However, empties should not dominate the benchmark.

---

# 13. Exclusion or separate-analysis criteria

Flag or exclude samples involving:

```text
crowd annotations without individual identities
severe annotation ambiguity
missing video files
corrupt videos
annotation/video mismatches
humans when the intended detector excludes humans
vehicles unless intentionally included
non-animal triggers
animals visible only in tiny/unusable fragments
```

Some difficult cases are scientifically valuable.

Rather than deleting all difficult footage, use a flag such as:

```text
difficulty = normal
difficulty = difficult
difficulty = ambiguous
```

and report performance separately.

---

# 14. Detector and tracker configuration

The benchmark should use the **actual AddaxAI production pipeline**.

Do not substitute a different detector merely because it makes benchmarking easier.

The objective is to measure the product behaviour users will see.

Record:

```text
detector model name
detector version/hash
detector confidence threshold
class filtering logic
tracker name
tracker version
tracker parameters
frame sampling FPS
video decoding backend
hardware
software commit
date
```

Example:

```yaml
detector:
  name: SpeciesNet
  version: ...
  threshold: 0.20

tracker:
  name: ByteTrack
  parameters:
    ...

sampling:
  fps: 1.0
```

The benchmark must be reproducible.

---

# 15. FPS experiment

## 15.1 Do not rerun detector inference for every FPS

The detector is expected to be the expensive part of the pipeline.

Therefore, the benchmark should **not** process every video independently at 0.5, 1, 2, 5 and 10 FPS.

That would repeat nearly identical detector work many times and make the experiment unnecessarily expensive.

Instead:

> **Run detector inference once at the highest FPS needed for the benchmark, save the raw detector output, and simulate all lower FPS settings by selecting subsets of those detections.**

AddaxAI already writes its raw inference output to:

```text
results.json
```

This is particularly convenient because `results.json` is produced **before tracking**. It contains the raw per-frame inference results rather than tracks.

That means the same detector output can be reused to test many tracker FPS settings without rerunning the neural network.

The preferred architecture is:

```text
                         ┌→ 10 FPS detections → tracker → track count
video → detector @ 10 FPS├→  5 FPS subset     → tracker → track count
       ONCE              ├→  2 FPS subset     → tracker → track count
       ↓                 ├→  1 FPS subset     → tracker → track count
   results.json          └→ 0.5 FPS subset     → tracker → track count
```

The computationally expensive detector is therefore run only once.

The comparatively cheap tracker can be rerun many times.

---

## 15.2 Why subsampling raw detections is valid

Suppose detector inference is run at 10 FPS.

The raw detector output conceptually contains:

```text
0.0 s   detections
0.1 s   detections
0.2 s   detections
0.3 s   detections
...
0.9 s   detections
1.0 s   detections
1.1 s   detections
...
```

A lower-FPS tracking experiment can then select only the relevant timestamps.

For example:

```text
10 FPS:
0.0, 0.1, 0.2, 0.3, ..., 1.0, 1.1, ...

5 FPS:
0.0, 0.2, 0.4, 0.6, 0.8, 1.0, ...

2 FPS:
0.0, 0.5, 1.0, 1.5, 2.0, ...

1 FPS:
0.0, 1.0, 2.0, 3.0, ...

0.5 FPS:
0.0, 2.0, 4.0, 6.0, ...
```

Each subset is then passed through the tracker as if those were the only detector frames available.

This reproduces the important part of the low-FPS tracking problem:

```text
larger time gaps between detections
→ larger animal displacement
→ harder data association
→ potentially more track fragmentation or association errors
```

---

## 15.3 Important: subsample detections, not finished tracks

The high-FPS tracking result must **not** be used to derive the low-FPS result.

For example, this would be wrong:

```text
10 FPS detections
      ↓
10 FPS tracker
      ↓
10 FPS tracks
      ↓
remove intermediate track points
      ↓
pretend this is 1 FPS tracking
```

The 10 FPS tracker has already used the intermediate detections to solve associations.

That gives it information that a genuine 1 FPS tracker would never have had.

Instead, always do:

```text
10 FPS raw detections
      ↓
select 1 FPS detection frames
      ↓
NEW tracker run
      ↓
1 FPS tracks
```

The cached artefact must therefore be the **raw detector output**.

In AddaxAI, that artefact is `results.json`.

---

## 15.4 Use AddaxAI `results.json` as the benchmark cache

The benchmark should use AddaxAI's existing `results.json` format wherever possible rather than inventing a separate detector-cache format.

The future coding agent should first inspect the exact structure of the current AddaxAI `results.json`.

At minimum, the experiment needs to recover for each processed frame:

```text
video identifier
frame identifier and/or timestamp
original frame number if available
bounding boxes
detection confidence
predicted class/species
any other detector fields required by the tracker
```

A conceptual example is:

```json
{
  "video": "camera_001.mp4",
  "detections": [
    {
      "frame": 0,
      "timestamp": 0.0,
      "detections": [
        {
          "bbox": [100, 80, 350, 420],
          "confidence": 0.94,
          "class": "red_deer"
        }
      ]
    }
  ]
}
```

This example is illustrative only.

**Do not change the real AddaxAI schema to match this document.**

Instead, write a small adapter around the actual `results.json` structure.

---

## 15.5 Master inference FPS

The default proposal is:

```text
MASTER_INFERENCE_FPS = 10
```

Then derive:

```text
10 FPS
5 FPS
2 FPS
1 FPS
0.5 FPS
```

from the cached detections.

Ten FPS is useful because it provides a relatively dense common source while still being far cheaper than analysing native 25/30 FPS footage.

However, the master FPS should not be treated as fixed if 10 FPS across 2,000 videos is unnecessarily expensive.

Possible alternatives include:

```text
5 FPS
```

or another rate that covers all relevant production candidates.

The master inference rate only needs to be high enough that every lower FPS condition can be generated from it accurately.

---

## 15.6 Frame-selection rules

FPS subsampling must be deterministic.

Do not simply use:

```python
every_nth_result
```

unless the source FPS and target FPS divide cleanly and the actual timestamps confirm this is correct.

Prefer timestamp-based selection.

Conceptually:

```python
target_times = 0, 1/fps, 2/fps, 3/fps, ...
```

Then choose the cached detection frame corresponding to each target time according to the same rule AddaxAI normally uses when sampling video.

This matters because:

- source videos may have unusual FPS values
- decoded timestamps may not be perfectly regular
- variable-frame-rate video may occur
- 10 FPS to 2 FPS divides cleanly, but arbitrary future test rates may not
- the benchmark should reproduce AddaxAI's real sampling behaviour as closely as possible

Document the exact selection rule.

---

## 15.7 Controlled-comparison advantage

Using one shared inference pass has another advantage beyond compute savings.

All FPS conditions originate from the **same detector run**.

Therefore:

```text
10 FPS experiment
5 FPS experiment
2 FPS experiment
1 FPS experiment
0.5 FPS experiment
```

differ only in which raw detections are shown to the tracker.

This removes detector-run variation as a confounding factor.

Differences in track count can therefore be attributed much more directly to:

```text
temporal sampling
+
tracker behaviour
```

rather than to slightly different decoding or inference executions.

---

## 15.8 Recommended FPS set

At minimum evaluate:

```text
0.5 FPS
1 FPS
2 FPS
5 FPS
```

If the master inference pass is 10 FPS, also include:

```text
10 FPS
```

Suggested default:

```python
FPS_VALUES = [0.5, 1, 2, 5, 10]
MASTER_INFERENCE_FPS = 10
```

The most important production comparison remains around:

```text
1 FPS
```

because 1 FPS is a common practical rate for computationally expensive camera-trap detection models.

---

## 15.9 Detector compute is paid once

For a 10 FPS master run, the expensive part becomes approximately:

```text
2,000 videos × detector @ 10 FPS
```

rather than:

```text
2,000 videos × detector @ 0.5 FPS
+
2,000 videos × detector @ 1 FPS
+
2,000 videos × detector @ 2 FPS
+
2,000 videos × detector @ 5 FPS
+
2,000 videos × detector @ 10 FPS
```

After the master inference pass, new tracker settings or FPS values can be tested repeatedly without rerunning the detector, provided the required frames exist in `results.json`.

This also makes the benchmark reusable for future tracker development.

---

# 16. Running the AddaxAI pipeline

The experiment should be split into two clearly separated phases.

## Phase A: expensive inference pass, performed once

For each benchmark video:

1. Decode the video.
2. Sample frames at the chosen master inference FPS, initially proposed as 10 FPS.
3. Run the actual AddaxAI wildlife detector.
4. Apply the normal AddaxAI detector output logic.
5. Save the raw inference results using AddaxAI's existing `results.json` mechanism.
6. Keep the raw result before any tracking operation.

The output from this phase becomes the persistent benchmark inference cache.

Example:

```text
benchmark_cache/
├── video_000001/
│   └── results.json
├── video_000002/
│   └── results.json
└── ...
```

The exact directory layout can follow existing AddaxAI conventions.

The important point is that detector inference should not need to be repeated for later tracking experiments.

---

## Phase B: cheap repeated tracker experiments

For each cached `results.json`:

1. Load the raw detections.
2. Select only the detections corresponding to the requested target FPS.
3. Feed that subset into a fresh tracker instance.
4. Finalise tracks.
5. Calculate the suggested track count.
6. Calculate the MaxN baseline from the same selected detections.
7. Save the result.
8. Repeat for the next FPS.

Conceptually:

```python
raw_results = load_results_json(video)

for fps in [0.5, 1, 2, 5, 10]:
    sampled_detections = subsample_results(raw_results, fps=fps)

    tracks = run_tracker(sampled_detections)

    track_count = count_tracks(tracks)
    maxn = calculate_maxn(sampled_detections)

    save_result(
        video=video,
        fps=fps,
        track_count=track_count,
        maxn=maxn,
    )
```

The exact code should call existing AddaxAI production functions wherever possible.

Do not create a second independent tracker implementation purely for the benchmark.

---

## 16.1 Cache once, experiment many times

This design should make it cheap to later test:

```text
different tracker parameters
different track timeout settings
different matching thresholds
different FPS values
different track filtering rules
different count derivation rules
```

without rerunning detector inference.

For example:

```text
results.json
      ↓
tracker configuration A @ 1 FPS
tracker configuration B @ 1 FPS
tracker configuration C @ 1 FPS
```

can all use exactly the same detector data.

This is valuable because tracker optimisation may continue after the first benchmark run.

---

## 16.2 Preserve detector confidence rather than filtering too early

Where possible, the cached `results.json` should retain the normal raw confidence values needed to reproduce production filtering.

If threshold changes are likely to be part of future experiments, avoid creating a benchmark cache that has already permanently discarded detections below one experimental threshold.

Ideally:

```text
master inference
→ sufficiently complete raw results.json
→ threshold/filter
→ FPS subset
→ tracker
```

Whether this is possible depends on what AddaxAI currently stores.

Do not change the production format unnecessarily, but document any limitation.

---

## 16.3 Validate the cached approach once

Before running the full benchmark, perform a small validation.

Choose approximately:

```text
10–20 videos
```

and compare:

### Method A

Run AddaxAI directly at:

```text
1 FPS
```

### Method B

Run AddaxAI once at:

```text
10 FPS
```

then reconstruct 1 FPS by selecting the appropriate detections from `results.json`.

The resulting selected raw detections should be identical or functionally equivalent.

If they are not, investigate why.

Possible reasons include:

```text
different frame-selection alignment
timestamp rounding
video decoding differences
stateful detector preprocessing
different confidence/filtering behaviour
```

Only after this validation should the cached-inference approach be used for all 2,000 videos.

This small check protects the experiment from accidentally benchmarking a subsampling method that differs from real AddaxAI 1 FPS processing.

---

## 16.4 Tracker state must be reset for every condition

Every FPS and tracker configuration must use a fresh tracker state.

Do not continue a tracker between conditions.

Correct:

```text
results.json
   ↓
fresh tracker @ 1 FPS
   ↓
result

results.json
   ↓
fresh tracker @ 2 FPS
   ↓
result
```

Incorrect:

```text
tracker @ 10 FPS
   ↓
modify existing tracker state
   ↓
pretend it represents 1 FPS
```

---

## 16.5 Production configuration

Do not manually tune tracker parameters separately for PanAf500, MammAlps and SA-FARI unless the goal is explicitly a research comparison.

A single production configuration should eventually work across datasets.

Dataset-specific results may be analysed separately, but the production algorithm should remain common.

---

# 17. Required output per sample

Create a machine-readable result table.

Suggested schema:

```csv
dataset,
sample_id,
sample_type,
species,
location_id,
duration_seconds,
original_fps,
evaluation_fps,
ground_truth_count,
predicted_track_count,
count_error,
absolute_count_error,
exact_match,
overcount,
undercount,
n_detector_detections,
n_tracks,
day_night,
notes
```

Definitions:

```text
count_error =
predicted_track_count - ground_truth_count
```

```text
absolute_count_error =
abs(predicted_track_count - ground_truth_count)
```

```text
exact_match =
predicted_track_count == ground_truth_count
```

```text
overcount =
predicted_track_count > ground_truth_count
```

```text
undercount =
predicted_track_count < ground_truth_count
```

---

# 18. Primary metrics

## 18.1 Exact count accuracy

The most intuitive product metric:

```text
exact count accuracy =
number of samples with predicted count == true count
/
number of samples
```

Example:

```text
1,720 / 2,000 = 86.0%
```

This directly answers:

> How often would the user receive the correct suggested count without needing to edit it?

---

## 18.2 Mean absolute count error

```text
MAE = mean(abs(predicted_count - true_count))
```

This indicates the typical magnitude of a mistake.

An exact-match rate alone does not distinguish:

```text
true 2 → predicted 3
```

from:

```text
true 2 → predicted 12
```

MAE does.

---

## 18.3 Overcount rate

```text
overcount rate =
samples where predicted_count > true_count
/
all samples
```

This is particularly important because track fragmentation is expected to create an upward bias.

---

## 18.4 Undercount rate

```text
undercount rate =
samples where predicted_count < true_count
/
all samples
```

Compare this against overcounting.

---

## 18.5 Mean signed error

```text
mean(predicted_count - true_count)
```

Interpretation:

```text
> 0  systematic overcount
< 0  systematic undercount
≈ 0  no net bias
```

Be careful: positive and negative errors can cancel, so always report it alongside MAE.

---

# 19. Additional useful metrics

Also consider:

```text
median absolute error
95th percentile absolute error
percentage within ±1 animal
percentage within ±2 animals
correlation between predicted and true counts
```

For higher-count groups:

```text
relative error =
abs(predicted - true) / true
```

Do not use relative error for true count = 0.

---

# 20. Classical tracking metrics are secondary

If the ground-truth annotations allow it, also calculate standard tracking metrics such as:

```text
ID switches
track fragmentation
IDF1
HOTA
MOTA
```

These are useful diagnostics.

However, they are **not the primary AddaxAI success criterion**.

A tracker can have imperfect identity continuity while still returning the correct number of tracks.

Conversely, a tracker may appear reasonable visually while generating too many fragmented tracks and therefore a poor suggested count.

The key product metric remains:

```text
predicted track count versus human count
```

---

# 21. Analyse count accuracy by true group size

This is important.

Break results into:

```text
1 animal
2 animals
3 animals
4 animals
5+ animals
```

Expected pattern:

```text
single animals → easiest
small groups → moderate
large groups → hardest
```

A tracker that achieves 95% overall accuracy may still perform poorly for groups if the benchmark is dominated by single animals.

Always report the group-size distribution.

---

# 22. Analyse by species

Calculate at least:

```text
N
exact accuracy
MAE
overcount rate
undercount rate
```

for species with sufficient sample size.

This can reveal patterns such as:

```text
large slow mammals → easy
small fast mammals → difficult
birds → difficult
arboreal animals → difficult
group-living species → difficult
```

Do not overinterpret species with very low N.

---

# 23. Analyse by movement and behaviour where available

Where annotations allow it, compare:

```text
stationary
walking
running
climbing
feeding
camera interaction
entering/leaving rapidly
```

The expectation is that faster motion creates larger displacement between 1 FPS samples and therefore more track fragmentation.

PanAf500 and MammAlps are especially useful for this analysis because they include behaviour information.

---

# 24. Analyse by day/night

If metadata permits:

```text
day
night / infrared
unknown
```

Night footage may have:

```text
motion blur
IR illumination
lower contrast
eye shine
different detector confidence
```

These effects can affect both detection and tracking.

---

# 25. Separate detector errors from tracker errors

This is very important.

A wrong count can originate from:

```text
detector failure
tracker failure
both
```

Example:

```text
true animals = 3

detector consistently sees only 2
tracker produces 2 perfect tracks

predicted count = 2
```

That is not primarily a tracking failure.

Where ground-truth bounding boxes are available, perform an optional **oracle-detection experiment**.

---

# 26. Oracle-detection experiment

For densely annotated datasets such as PanAf500 and SA-FARI, use human bounding boxes as detector inputs.

Conceptually:

```text
ground-truth bounding boxes
        ↓
      tracker
        ↓
 predicted tracks
        ↓
 ground-truth individual IDs
```

This isolates the tracker.

Run at the same FPS values by sampling the ground-truth annotation frames.

Compare:

```text
A. real detector + tracker
B. human boxes + tracker
```

Interpretation:

If B is excellent but A is poor:

```text
detector is the bottleneck
```

If B is also poor:

```text
tracker / sparse temporal sampling is the bottleneck
```

This is one of the most valuable secondary experiments and should be implemented if annotation formats permit it without excessive work.

---

# 27. Specific 1 FPS fragmentation analysis

Because 1 FPS is the primary production candidate, investigate its errors manually.

Take a stratified sample of incorrect 1 FPS results, for example:

```text
50 overcounts
50 undercounts
```

For every failure, classify the dominant reason:

```text
track fragmentation
missed detection
identity switch
incorrect association
occlusion
fast movement
animal leaves/re-enters
very small animal
multiple overlapping animals
false-positive detection
species/classification interaction
other
```

Produce a simple failure table.

This will be far more actionable than a single accuracy percentage.

---

# 28. Compare FPS against computation cost

The final product decision is a trade-off.

Record approximate:

```text
frames processed
detector runtime
total runtime
GPU time
```

for each FPS.

Example:

| FPS | Exact count accuracy | MAE | Relative detector cost |
|---:|---:|---:|---:|
| 0.5 | 78% | 0.34 | 0.5× |
| 1 | 88% | 0.18 | 1× |
| 2 | 92% | 0.11 | 2× |
| 5 | 94% | 0.08 | 5× |
| 10 | 94.5% | 0.07 | 10× |

These values are hypothetical.

If real results look like this, 1 or 2 FPS may be preferable even though 10 FPS is slightly more accurate.

The question is not:

> Which FPS produces the absolute best tracker?

It is:

> Which FPS produces a sufficiently good suggested count at a reasonable compute cost?

---

# 29. Possible acceptance criteria

Do not hard-code these as requirements before seeing the data, but reasonable product targets might be:

```text
≥ 90% exact count accuracy for single-animal videos
≥ 80–90% overall exact count accuracy
low catastrophic-error rate
MAE comfortably below 0.5 animals/video
clear improvement over simply using MaxN
```

The real acceptance threshold should depend on how much time human verification saves.

A suggested count does not need research-grade autonomous accuracy if:

- verification is optional
- correction is easy
- the estimate is usually correct
- large errors are rare

---

# 30. Compare track count against simple MaxN

Although track count is the proposed camera-trap metric, include a simple baseline:

```text
MaxN =
maximum number of animal detections present in any processed frame
```

Then compare:

```text
track count
versus
MaxN
versus
ground-truth count
```

This empirically tests the design decision.

Expected example:

```text
Two deer cross sequentially and are never visible together.

true count = 2
MaxN = 1
track count = 2
```

Track count should perform better for many terrestrial camera-trap clips where animals pass through sequentially.

However, track fragmentation may make MaxN outperform track count in some conditions.

The benchmark should measure this instead of assuming the answer.

Required baseline fields:

```text
predicted_track_count
predicted_maxn
ground_truth_count
```

---

# 31. Reproducibility

Save a benchmark manifest before running experiments.

Example:

```text
benchmark_manifest_v1.csv
```

Fields should include:

```text
dataset
sample_id
source_video
annotation_source
ground_truth_count
species
location
split
license/reference
checksum where practical
```

Once benchmark v1 is defined, do not silently replace failed videos with new ones.

If a benchmark item is invalid, mark it and create a new manifest version.

Example:

```text
benchmark_manifest_v2.csv
```

Use deterministic sampling.

---

# 32. Avoid evaluation leakage

If any AddaxAI detector or tracker has been trained or tuned on one of these datasets, record this.

Separate results into:

```text
known/training-domain
independent/external
```

A dataset used to tune tracker thresholds should not subsequently be presented as a completely independent final benchmark.

A reasonable workflow is:

```text
development subset
        ↓
parameter tuning
        ↓
frozen configuration
        ↓
held-out benchmark
```

SA-FARI is large enough to support this split if necessary.

---

# 33. Suggested implementation structure

A coding agent could create something like:

```text
tracking_count_benchmark/
├── README.md
├── config.yaml
├── manifests/
│   ├── panaf500.csv
│   ├── mammalps.csv
│   ├── safari.csv
│   └── benchmark_2000.csv
├── adapters/
│   ├── panaf500.py
│   ├── mammalps.py
│   └── safari.py
├── ground_truth/
│   └── build_ground_truth.py
├── run/
│   ├── build_inference_cache.py
│   ├── subsample_results_json.py
│   ├── run_benchmark.py
│   └── run_oracle_tracking.py
├── analysis/
│   ├── metrics.py
│   ├── compare_fps.py
│   ├── failure_analysis.py
│   └── make_report.py
└── results/
```

Dataset-specific parsing should remain separate from AddaxAI's tracking code.

The benchmark should invoke production tracking logic rather than copying/reimplementing it.

---

# 34. Suggested configuration

Example only:

```yaml
benchmark:
  random_seed: 42

datasets:
  panaf500:
    target_samples: 500
  mammalps:
    target_samples: 397
  safari:
    target_samples: 1103

fps:
  master_inference_fps: 10
  tracker_conditions:
    - 0.5
    - 1
    - 2
    - 5
    - 10
  reuse_addaxai_results_json: true
  rerun_detector_per_condition: false

metrics:
  - exact_count_accuracy
  - mae
  - signed_error
  - overcount_rate
  - undercount_rate
  - within_one_accuracy

baselines:
  maxn: true

oracle_tracking:
  enabled: true
```

---

# 35. Expected final report

The coding agent should generate a concise Markdown or HTML report containing at least:

## Dataset composition

```text
number of videos/events
species distribution
group-size distribution
day/night distribution if available
dataset contribution
```

## Main 1 FPS result

```text
exact count accuracy
MAE
signed error
overcount rate
undercount rate
```

## FPS comparison

Table and plot of:

```text
FPS vs exact-count accuracy
FPS vs MAE
FPS vs runtime
```

## Track-count vs MaxN comparison

This is required.

## Group-size performance

At least:

```text
1
2
3
4
5+
```

## Dataset-specific performance

```text
PanAf500
MammAlps
SA-FARI
```

## Species-level results

For sufficiently represented species.

## Failure analysis

Especially for 1 FPS.

## Recommendation

The report should finish with a concrete recommendation such as:

```text
Use 1 FPS in production.
```

or:

```text
Increase camera-trap video analysis to 2 FPS because exact-count
accuracy improves substantially for only a 2× detection cost.
```

or:

```text
Keep 1 FPS detection but change tracker configuration because
oracle-box tracking shows that detection FPS itself is not the
primary bottleneck.
```

The recommendation must follow from the measured data.

---

# 36. Questions the experiment should ultimately answer

At completion, we should be able to answer all of these:

1. Is number of tracks a useful AI suggestion for camera-trap animal count?
2. Is 1 FPS sufficient?
3. How frequently does 1 FPS split one animal into multiple tracks?
4. Does low FPS create a meaningful undercount problem as well?
5. How does error change with animal group size?
6. Which species or animal types are hardest?
7. Which behaviours are hardest?
8. Does nighttime footage significantly reduce performance?
9. Is the detector or tracker the larger source of counting error?
10. Does track count outperform MaxN for terrestrial camera-trap video?
11. Would increasing FPS meaningfully improve user experience?
12. What production default gives the best accuracy/compute trade-off?

---

# 37. Why this matters for AddaxAI

The purpose of the feature is not to claim fully automated wildlife abundance estimation.

The intended workflow is:

```text
AI analyses video
        ↓
AI suggests species + count
        ↓
researcher optionally verifies
        ↓
verified observation
```

The benchmark therefore evaluates the system in the context in which it will actually be used.

If the AI suggests the right number for most videos, it can remove a substantial amount of manual work.

If the count is occasionally wrong but easy to correct, the feature can still be valuable.

The benchmark should therefore optimise for:

```text
useful suggestions
predictable failure behaviour
low manual correction burden
reasonable computational cost
```

rather than perfect individual re-identification.

---

# 38. Dataset references

## PanAf / PanAf500

PanAf project:

https://obrookes.github.io/panaf.github.io/

PanAf project GitHub:

https://github.com/obrookes/panaf.github.io

PanAfrican2020, University of Bristol:

```text
DOI: 10.5523/bris.jh6hrovynjik2ix2h7m6fdea3
```

PanAfrican2019 annotations:

```text
DOI: 10.5523/bris.1v9op9lc6zi5g25kkwa5smb3vq
```

Useful PanAf500 processing reference:

https://github.com/ecker-lab/dap-behavior

Important note:

**Verify current source-video availability before relying on PanAf500.**

---

## MammAlps

GitHub:

https://github.com/eceo-epfl/MammAlps

Zenodo current release checked when this document was created:

https://zenodo.org/records/15588220

Dataset DOI family:

```text
10.5281/zenodo.15040900
```

Associated paper:

```text
MammAlps: A multi-view video behavior monitoring dataset of wild
mammals in the Swiss Alps
Gabeff et al., CVPR 2025
```

License documented as:

```text
CC BY-NC 4.0
```

---

## SA-FARI

Project:

https://www.conservationxlabs.com/sa-fari

Hugging Face:

https://huggingface.co/datasets/facebook/SA-FARI

Associated paper:

```text
The SA-FARI Dataset:
Segment Anything in Footage of Animals for Recognition and Identification
```

Dataset scale reported by the authors:

```text
11,609 camera-trap videos
741 locations
4 continents
99 species categories
16,224 masklet identities
~942,702 frame-level annotations
```

Hugging Face access is gated and requires accepting the dataset terms.

License is documented by the dataset as:

```text
CC BY-NC 4.0
```

---

# 39. Final instruction to the future coding agent

Do not start by writing the benchmark runner.

Start by validating the datasets.

For each dataset:

1. download or obtain a small example
2. inspect the annotation structure
3. determine exactly how biological individual count can be recovered
4. confirm video-to-annotation mapping
5. document any ambiguity
6. only then build the unified benchmark manifest

The most dangerous mistake in this experiment would be treating annotation **track segments** as ground-truth **individual animals**.

The entire purpose of this benchmark is partly to measure whether AddaxAI fragments one animal into multiple tracks.

Ground truth must therefore remain biologically independent from the track fragmentation being evaluated.

Once the ground-truth adapters have been validated, freeze the 2,000-sample manifest and run the production AddaxAI detector + tracker against it.

The main deliverable is not a tracker score.

The main deliverable is a defensible answer to:

> **At the computationally practical frame rates used by AddaxAI, how good is track count as an automatic suggested animal count for real camera-trap videos?**


---

# Experiment 2: tracking across still-image camera-trap events

# 40. Purpose

This second experiment evaluates whether the same tracking concept used for video can also improve animal-count suggestions for **still-image camera-trap events**.

Camera traps often capture multiple photographs of the same trigger or animal encounter. Depending on camera configuration, these may be rapid bursts separated by fractions of a second, slower sequences separated by several seconds, or more irregular series of photographs.

AddaxAI already groups still images into ecological events using a temporal independence interval.

The default AddaxAI independence interval is:

```text
30 minutes
```

The event rule is based on the interval between **consecutive images**, not simply the total duration of the event.

Conceptually:

```text
image A → 10 min → image B → 20 min → image C → 25 min → image D
```

All four images belong to the same event because every consecutive gap is <= 30 minutes, even though the event as a whole lasts 55 minutes.

Formally:

```text
start new event if:

timestamp(current_image) - timestamp(previous_image) > 30 minutes
```

Therefore an AddaxAI event can theoretically last much longer than 30 minutes if detections/images continue to occur with gaps of 30 minutes or less.

The central question of Experiment 2 is:

> **Can tracking detections across the still images within an AddaxAI event produce a better suggested animal count than simply using the maximum number of animals visible in one photograph, and how does that benefit change as the time between photographs increases?**

---

# 41. Why this experiment is needed

For still images, a simple counting method is MaxN:

```text
MaxN =
maximum number of animals detected simultaneously
in any one image in the event
```

This works well when all animals are visible together.

However, it can undercount when animals pass through the camera sequentially.

Example:

```text
image 1 → deer A
image 2 → deer A
image 3 → deer B
image 4 → deer B
```

If deer A and deer B are never visible simultaneously:

```text
MaxN = 1
true event count = 2
```

A tracker or temporal association algorithm could potentially produce:

```text
track 1 = deer A
track 2 = deer B

TrackCount = 2
```

This would provide a better suggested count.

At the same time, still-image sequences are much sparser than ordinary video.

A camera might capture images at:

```text
0.5 seconds
1 second
2 seconds
5 seconds
30 seconds
2 minutes
10 minutes
```

or at irregular intervals.

As the interval grows, position and motion information become less informative.

A deer photographed five minutes later may be the same deer or a completely different deer. Without individually identifying markings, the tracker may have little defensible evidence either way.

Therefore tracking should not automatically be assumed to work across the entire 30-minute AddaxAI event definition.

The experiment should determine **empirically where temporal association remains useful**.

---

# 42. Product distinction: ecological event versus tracking window

This experiment must keep two concepts separate.

## Ecological event

The AddaxAI event is defined using the normal independence interval:

```text
30 minutes by default
```

This remains the ecological grouping mechanism.

## Tracking/association window

The maximum temporal gap across which AddaxAI should attempt to continue an animal track may be much shorter.

For example, the experiment could eventually support a rule such as:

```text
Event threshold:
30 minutes

Tracking threshold:
2 seconds
```

An event could then contain several independent tracking segments:

```text
AddaxAI event
│
├── tracking segment A
│   ├── image 1
│   ├── image 2
│   └── image 3
│
├── larger temporal gap
│
├── tracking segment B
│   ├── image 4
│   └── image 5
│
└── ...
```

The benchmark should determine whether such a distinction is useful and what the tracking threshold should be.

Do **not** change the 30-minute ecological independence interval merely because tracking becomes unreliable at shorter intervals.

---

# 43. Main goal

The experiment should determine:

1. Whether tracking detections across still images improves animal-count suggestions.
2. Whether TrackCount outperforms MaxN.
3. How tracking accuracy changes with inter-image interval.
4. At what temporal gap tracking stops adding useful information.
5. Whether the useful interval differs by species, movement speed, body size or habitat.
6. Whether rapid camera bursts behave essentially like very-low-FPS video.
7. Whether tracking remains useful for cameras that do **not** use rapid bursts.
8. Whether irregular image sequences can still benefit from temporal association.
9. Whether the same production tracker used for video can be reused for still images.
10. Whether AddaxAI should use a separate tracking-gap threshold inside its normal 30-minute events.

---

# 44. Primary hypotheses

## Hypothesis 1

> **Tracking across rapid still-image bursts will improve animal-count accuracy compared with MaxN, especially when different animals pass through the camera sequentially.**

## Hypothesis 2

> **Tracking performance will decline as the interval between photographs increases because position and motion become less predictive of animal identity.**

## Hypothesis 3

> **At sufficiently large temporal gaps, continuing tracks will no longer be defensible and may produce incorrect merging of different animals.**

## Hypothesis 4

> **There will be a practical temporal region in which tracking is useful enough to justify an AddaxAI tracking window that is substantially shorter than the 30-minute ecological event threshold.**

## Hypothesis 5

> **The useful temporal threshold will not be identical for all species and camera projects. Slow or stationary large mammals may remain associable across longer gaps than fast, small or highly mobile animals.**

---

# 45. What counts as success

The experiment is not intended to prove that AddaxAI can identify biological individuals with certainty.

The intended product workflow remains:

```text
images
  ↓
detections
  ↓
temporal association/tracking
  ↓
AI suggested count
  ↓
optional human verification
```

The automated count only needs to be sufficiently useful that it reduces manual work.

The main comparison is:

```text
human verified count
        ↑
        │
 ┌──────┴──────┐
 │             │
MaxN       TrackCount
```

If TrackCount is more accurate than MaxN for useful portions of the data, tracking has product value.

---

# 46. Event reconstruction

For datasets that already provide a burst or sequence identifier, preserve the dataset's original sequence definition for the primary benchmark.

Examples:

```text
sequence_id
capture_event_id
burst_id
```

For datasets that provide individual timestamped photographs but no suitable sequence grouping, reconstruct **AddaxAI-style events**.

Sort photographs by:

```text
camera/location
timestamp
```

Then create a new event whenever:

```text
gap between consecutive images > 30 minutes
```

Equivalent pseudocode:

```python
events = []

for camera in cameras:
    images = sort_by_timestamp(camera.images)

    current_event = []

    for image in images:
        if not current_event:
            current_event = [image]
            continue

        gap = image.timestamp - current_event[-1].timestamp

        if gap <= 30 * 60:
            current_event.append(image)
        else:
            events.append(current_event)
            current_event = [image]

    if current_event:
        events.append(current_event)
```

This reproduces the AddaxAI default event logic.

Keep both:

```text
dataset_native_sequence_id
addaxai_30min_event_id
```

where possible.

This allows the benchmark to compare native camera bursts with broader AddaxAI event grouping.

---

# 47. Important ground-truth problem for long events

For rapid bursts, datasets often provide a human count such as:

```text
group_size = 3
```

That is excellent ground truth.

For reconstructed 30-minute AddaxAI events, however, care is needed.

Suppose:

```text
12:00 deer
12:10 deer
12:20 deer
```

and each image has a count of 1.

It is **not valid** to conclude:

```text
true event count = 3
```

because this could be the same deer three times.

Similarly, it is not valid to conclude:

```text
true event count = 1
```

because three different deer could have passed sequentially.

Therefore, long-event count ground truth should only be used when the dataset provides one of:

1. explicit sequence/event-level human individual count
2. persistent individual identities
3. sufficiently detailed human annotation to establish unique individuals
4. another defensible event-level count definition

Datasets with only per-image counts can still be used for:

```text
MaxN
burst analysis
detection analysis
```

but must not be treated as verified unique-individual ground truth for long 30-minute events.

This distinction is critical.

---

# 48. Recommended benchmark size

Use approximately:

```text
5,000 still-image sequences/events
```

The goal is diversity rather than simply maximising N.

Recommended starting composition:

| Dataset | Target | Primary contribution |
|---|---:|---|
| iWildCam 2022 | 1,780 | Explicit sequence counts, variable burst structure, global cameras |
| Snapshot Serengeti Gold Standard | 1,200 | Expert-verified African burst events |
| Snapshot Safari 2024 Expansion | 1,200 | Many independent projects, habitats and camera configurations |
| SNAPSHOT USA 2024 | 820 | North American projects, broad camera-array diversity, longer/non-burst sequences |
| **Total** | **5,000** | |

The exact numbers may change after validating annotation semantics.

Do not force the benchmark to exactly 5,000 if doing so requires weaker ground truth.

A final benchmark between approximately:

```text
4,500–5,500
```

high-quality events is preferable to exactly 5,000 poorly defined ones.

---

# 49. Dataset 1: iWildCam 2022

## Why it is the core dataset

iWildCam 2022 is unusually well matched to this experiment because its competition task explicitly involved **counting unique animals across camera-trap image sequences**.

Sequences contain approximately:

```text
1–10 images
```

and the metadata includes sequence structure.

Relevant fields include:

```text
seq_id
seq_num_frames
frame_num
datetime
location
sub_location
```

This allows reconstruction of:

```text
image order
sequence membership
actual inter-image interval
```

The training data includes approximately:

```text
1,780 sequences
```

with explicit sequence-count annotations.

This should be the core Experiment 2 benchmark.

## Ground truth

The relevant count metadata is provided in a file such as:

```text
metadata/train_sequence_counts.csv
```

The future coding agent must inspect the current dataset release and confirm:

```text
count semantics
species relationship
whether counts represent unique animals across the sequence
handling of multiple species
handling of empty sequences
```

before building the benchmark.

## Download/reference locations

LILA BC dataset page:

```text
https://lila.science/datasets/iwildcam-2022/
```

Competition data:

```text
https://www.kaggle.com/competitions/iwildcam2022-fgvc9/data
```

Competition/project code:

```text
https://github.com/visipedia/iwildcam_comp
```

The full dataset is large.

Prefer downloading metadata first, identifying the 1,780 count-labelled sequences, and then downloading only the required images if the storage mechanism permits selective retrieval.

## Planned sample

Use all approximately:

```text
1,780 count-labelled sequences
```

assuming annotation validation succeeds.

## Why this dataset matters

It should provide the clearest direct curve of:

```text
tracking accuracy
        versus
inter-image interval
```

because image timestamps and explicit sequence counts are both available.

---

# 50. Dataset 2: Snapshot Serengeti Gold Standard

## Overview

Snapshot Serengeti is one of the largest and best-known camera-trap datasets.

The original project contains approximately:

```text
3.2 million photographs
1.2 million capture events
225 camera locations
~48 species/species groups
```

Camera triggers commonly produced:

```text
1–3 photographs
```

in a rapid burst of approximately one second.

Human observers classified the **capture event/image set**, including:

```text
species
number of individuals
behaviour
presence of young
```

This is very useful because the count is naturally associated with the image sequence rather than being inferred independently from each photograph.

## Gold-standard subset

A particularly valuable subset contains approximately:

```text
4,149 expert-classified capture events
```

These can provide higher-confidence validation than relying only on crowd consensus.

Relevant reference:

```text
https://zenodo.org/records/5026386
```

General dataset page:

```text
https://lila.science/datasets/snapshot-serengeti
```

## Planned sample

Randomly/stratifiably select approximately:

```text
1,200 events
```

from the expert gold-standard events.

Stratify by:

```text
species
count
location
day/night where possible
```

Do not simply take the first 1,200.

## Main role

Snapshot Serengeti should primarily answer:

> Does tracking improve count estimates for ordinary rapid ~1-second camera bursts?

Its temporal interval variation is less useful than iWildCam, but its independent ecological project and expert labels make it a strong external validation source.

---

# 51. Dataset 3: Snapshot Safari 2024 Expansion

## Overview

Snapshot Safari 2024 Expansion is particularly useful for **project diversity**.

It combines many independent camera-trap projects from southern and eastern Africa.

The release contains approximately:

```text
4.0 million images
1,824 camera locations
151 categories
>1.3 million animal-count annotations
```

and represents roughly:

```text
15 camera-trapping projects
```

Reference:

```text
https://lila.science/datasets/snapshot-safari-2024-expansion/
```

## Why it matters

The experiment should not derive a tracking threshold from one camera model or one field protocol.

Snapshot Safari provides variation in:

```text
project
habitat
camera placement
species community
camera configuration
image sequence behaviour
environment
```

This is exactly the type of variation required to test whether an AddaxAI default generalises.

## Planned sample

Use approximately:

```text
1,200 sequences/events
```

but distribute them across the constituent projects.

Do not let one project dominate.

A possible strategy is:

```text
approximately equal allocation per eligible project
```

followed by stratification within projects.

## Validation required

Before inclusion, inspect:

```text
sequence IDs
timestamps
count semantics
camera/project metadata
```

Confirm whether count labels represent the full sequence/event or individual images.

Only use samples with defensible sequence-level count ground truth for the primary TrackCount-versus-human comparison.

Other samples may still be useful for secondary analyses.

---

# 52. Dataset 4: SNAPSHOT USA 2024

## Overview

SNAPSHOT USA provides a very different ecological domain from the African datasets.

The 2024 release includes approximately:

```text
377,000 sequence records
3,127 deployments
49 US states
12 ecoregions
184 camera-trap arrays/subprojects
```

and a broad range of North American mammals and birds.

The metadata includes fields such as:

```text
Sequence_ID
Group_Size
```

where Group_Size represents the number of individuals observed in the sequence.

This provides direct count information.

## Why it is particularly interesting

Wildlife Insights can group photographs into sequences over a substantially broader interval than classic rapid bursts.

This makes SNAPSHOT USA useful for testing:

```text
rapid bursts
versus
less burst-like image sequences
```

and for exploring whether temporal association remains useful outside classic burst photography.

It also introduces:

```text
different continents
different species
different research teams
different camera deployments
different ecoregions
```

## Planned sample

Use approximately:

```text
820 sequences
```

with deliberate stratification across:

```text
subprojects/arrays
species
group size
ecoregion
inter-image interval
```

Where image-level timestamps are available, calculate the actual gaps rather than relying only on the sequence definition.

## Wildlife Insights references

Public data information:

```text
https://www.wildlifeinsights.org/get-started/data-download/public
```

Sequence documentation:

```text
https://www.wildlifeinsights.org/get-started/identify-images/sequence-projects
```

## Role in the benchmark

This dataset should deliberately provide more **non-burst and irregular-sequence cases**.

It is therefore important not to filter everything down to only 1-second bursts.

---

# 53. Additional non-burst data

The four core datasets above should be sufficient for the first implementation.

However, the future coding agent should actively inspect whether the selected data adequately cover:

```text
<1 second
1–2 seconds
2–5 seconds
5–10 seconds
10–30 seconds
30–60 seconds
1–5 minutes
5–30 minutes
```

If the longer-gap bins contain too few samples with valid event-level count ground truth, add another public timestamped camera-trap dataset.

Good candidates should have:

```text
image timestamps
camera/location identifiers
species labels
count/group-size labels
preferably sequence/event identifiers
```

The key requirement is not that the source project calls the photographs a "burst".

The key requirement is that AddaxAI can reconstruct events and that there is defensible count ground truth.

Do not add a dataset solely because it is large.

---

# 54. Sampling strategy

The 5,000-event benchmark should intentionally maximise variability.

Stratify where possible across:

```text
dataset
project
camera/location
species
taxonomic group
habitat/ecoregion
group size
day/night
sequence length
event duration
median inter-image gap
maximum inter-image gap
```

Use a fixed random seed:

```python
RANDOM_SEED = 42
```

Export the final selection as:

```text
still_image_tracking_manifest_v1.csv
```

Suggested fields:

```text
dataset
project
event_id
native_sequence_id
camera_id
species
verified_count
n_images
event_duration_seconds
min_gap_seconds
median_gap_seconds
mean_gap_seconds
max_gap_seconds
day_night
ground_truth_type
source
```

---

# 55. Temporal-gap analysis

This is the central analysis of Experiment 2.

Do not reduce each event to only one label such as "burst".

Calculate actual timing statistics.

For every event:

```text
minimum gap
median gap
mean gap
maximum gap
event duration
number of images
```

Start with descriptive bins such as:

```text
≤0.5 s
>0.5–1 s
>1–2 s
>2–5 s
>5–10 s
>10–30 s
>30–60 s
>1–5 min
>5–10 min
>10–30 min
```

These bins can be merged if sample sizes are small.

The final analysis should preferably treat time as a continuous variable as well.

The most important figure is conceptually:

```text
benefit of TrackCount over MaxN
               ↑
               │\
               │ \
               │  \
               │   \
               │    \____
               └──────────────→ temporal gap
```

The goal is to find where the tracking benefit approaches zero or becomes negative.

---

# 56. Two related experiments within Experiment 2

Experiment 2 should actually contain two complementary analyses.

## 56.1 Native-sequence analysis

Use the sequence/event definitions supplied by the source datasets.

This provides the cleanest ground truth.

Question:

> How well does tracking work for the image sequences that real camera-trap projects actually produce?

## 56.2 AddaxAI 30-minute event analysis

Where timestamps and suitable ground truth permit, regroup photographs according to AddaxAI's default event rule:

```text
consecutive gap <= 30 minutes
```

Question:

> Within AddaxAI's broader ecological events, over what portions of the event is tracking still useful?

Do not manufacture a unique-animal ground truth when one is unavailable.

If a reconstructed event cannot be assigned a defensible verified count, it should not enter the primary accuracy metric.

It can still contribute to descriptive timing analysis.

---

# 57. Tracker input

Each still image is independently analysed by the normal AddaxAI detector.

Conceptually:

```text
image 1 → detections
image 2 → detections
image 3 → detections
...
```

Then:

```text
timestamped detections
        ↓
tracker / association
        ↓
tracks
        ↓
TrackCount
```

Use the **actual image timestamps**.

Do not pretend the images are consecutive video frames with a fixed frame interval.

The tracker should receive or otherwise account for:

```text
Δt between image observations
```

if its implementation supports this.

If the production tracker cannot handle variable time steps, explicitly test how this limitation affects the experiment.

---

# 58. Conservative association

For sparse still-image data, false merging can be particularly problematic.

Example:

```text
12:00:00 → deer A
12:00:05 → deer B
```

If there is insufficient evidence that these are the same animal, the algorithm should not aggressively merge them merely to reduce track fragmentation.

For AddaxAI's suggested-count use case, a conservative association policy is preferable.

The experiment should separately report:

```text
overcount
undercount
```

because tracker behaviour may shift from fragmentation-driven overcounting at short intervals to incorrect-association undercounting at longer intervals.

---

# 59. Baselines

At least three methods should be compared.

## Baseline A: MaxN

```text
maximum number of animals detected in any single image
```

## Method B: TrackCount

```text
number of tracks produced across the event
```

## Baseline C: per-image sum

For diagnostic purposes only:

```text
sum of animal detections across all images
```

This is expected to overcount repeated appearances badly, but it provides a useful naive baseline.

Where species are mixed, calculate counts per species.

---

# 60. Optional tracking-window experiment

Rather than always tracking across the entire event, explicitly test maximum allowed continuation gaps.

For example:

```text
0.5 s
1 s
2 s
5 s
10 s
30 s
60 s
5 min
30 min
```

For a selected threshold:

```text
if gap <= tracking_window:
    allow track continuation
else:
    terminate tracks / begin new tracking segment
```

This may be more informative than simply asking whether the tracker can handle an arbitrary interval.

The resulting analysis can answer:

> What maximum temporal gap should AddaxAI allow when associating still-image detections?

The expected product architecture may become:

```text
30-minute ecological event
        ↓
split internally according to tracking gap
        ↓
track within short temporal segments
        ↓
derive suggested event count
```

However, the correct method for combining counts across multiple tracking segments must be validated against human ground truth.

Do not automatically sum segment counts, because the same biological individual could appear in more than one segment.

---

# 61. Primary metrics

Use the same core metrics as Experiment 1:

```text
exact count accuracy
mean absolute error
median absolute error
mean signed error
overcount rate
undercount rate
percentage within ±1 animal
```

Calculate these separately for:

```text
MaxN
TrackCount
```

Then calculate:

```text
TrackCount exact accuracy - MaxN exact accuracy
```

and:

```text
MaxN MAE - TrackCount MAE
```

Positive values should indicate improvement from tracking when defining the report.

---

# 62. Core result: tracking improvement curve

The most important result should show:

```text
temporal interval
versus
TrackCount improvement over MaxN
```

Produce this for:

```text
all datasets combined
each dataset
major species groups
group-size classes
```

If possible, use both:

```text
binned summaries
continuous fitted trend
```

Do not choose a product threshold solely from one arbitrary bin boundary.

---

# 63. Group-size analysis

Break performance into:

```text
1 animal
2 animals
3 animals
4 animals
5+ animals
```

Tracking may provide little advantage when only one animal is present.

Its main value may appear when:

```text
multiple animals move sequentially
```

Therefore overall accuracy alone could hide the most useful product effect.

---

# 64. Sequence-length analysis

Analyse:

```text
2 images
3 images
4–5 images
6–10 images
>10 images
```

Longer sequences provide more tracking information but also more opportunities for:

```text
fragmentation
re-entry
false association
```

---

# 65. Species and movement analysis

Where possible, compare:

```text
large mammals
small mammals
birds
primates
ungulates
carnivores
```

and behaviour/movement categories such as:

```text
stationary
feeding
walking
running
camera interaction
resting/sleeping
```

Expected pattern:

```text
stationary/slow animals
→ longer useful association interval

fast animals
→ shorter useful association interval
```

This should be measured rather than assumed.

---

# 66. Project/camera-setting analysis

Because the benchmark deliberately combines multiple projects, report performance per project.

Useful variables include:

```text
camera model if available
burst size
configured trigger interval
deployment
habitat
project
```

This can reveal whether one global tracking threshold is reasonable.

For example:

```text
Project A: 0.5-second bursts → strong benefit
Project B: 5-second sequences → modest benefit
Project C: irregular 30-second captures → no benefit
```

This is one of the reasons for using several datasets rather than only iWildCam.

---

# 67. Detector errors versus association errors

As in Experiment 1, distinguish:

```text
detection error
association/tracking error
```

If human bounding boxes or individual annotations are available, an oracle-detection analysis is useful.

Compare:

```text
real AddaxAI detections → tracker
```

with:

```text
human detections → tracker
```

This answers whether a poor count is caused by:

```text
missed animals
false detections
or
temporal association
```

---

# 68. Failure analysis

Manually inspect a stratified sample of incorrect results.

Suggested categories:

```text
track fragmentation
incorrect merge
animal left and returned
different animal entered
missed detection
false-positive detection
occlusion
large displacement
small animal
multiple overlapping animals
camera trigger gap too long
species mismatch
timestamp problem
annotation ambiguity
other
```

Specifically inspect failures near the candidate tracking threshold.

If accuracy falls sharply between 2 and 5 seconds, manually inspect cases in that range.

---

# 69. Expected outcomes

The expected result is **not known** and should not be forced.

A plausible pattern is:

```text
very short gaps:
tracking substantially better than MaxN

short gaps:
tracking still useful

medium gaps:
benefit declines

long gaps:
little reliable identity information

very long gaps:
attempting to maintain tracks may become harmful
```

For illustration only, the final curve might resemble:

```text
≤0.5 s      strong benefit
0.5–1 s     strong benefit
1–2 s       useful
2–5 s       mixed
5–10 s      weak
>10 s       little/no benefit
```

These are **not expected thresholds to encode in AddaxAI**.

The purpose of the experiment is to discover the actual curve.

---

# 70. Possible product outcomes

Several conclusions are possible.

## Outcome A: tracking works well up to a clear threshold

Example:

```text
Track detections across still images when gap <= 3 seconds.
```

This would be straightforward to implement.

## Outcome B: only rapid bursts benefit

Example:

```text
Track only when gap <= 1 second.
Otherwise use MaxN.
```

## Outcome C: species-dependent behaviour is substantial

The product could potentially remain conservative and use one short global threshold rather than adding complicated species-specific logic.

## Outcome D: tracking provides little benefit

If MaxN performs equally well, do not add unnecessary tracking complexity to still-image events.

## Outcome E: tracking helps, but only with appearance/ReID

This could motivate future work, but should not automatically expand the initial implementation.

Keep KISS/DRY/YAGNI in mind.

---

# 71. Recommended implementation structure

Add Experiment 2 alongside Experiment 1:

```text
tracking_count_benchmark/
├── README.md
├── experiment_1_video/
│   ├── manifests/
│   ├── run/
│   └── analysis/
├── experiment_2_still_events/
│   ├── manifests/
│   │   ├── iwildcam2022.csv
│   │   ├── snapshot_serengeti.csv
│   │   ├── snapshot_safari.csv
│   │   ├── snapshot_usa.csv
│   │   └── benchmark_5000.csv
│   ├── adapters/
│   │   ├── iwildcam2022.py
│   │   ├── snapshot_serengeti.py
│   │   ├── snapshot_safari.py
│   │   └── snapshot_usa.py
│   ├── events/
│   │   └── build_addaxai_events.py
│   ├── run/
│   │   ├── run_detector.py
│   │   ├── run_tracking.py
│   │   └── run_tracking_windows.py
│   └── analysis/
│       ├── metrics.py
│       ├── interval_curve.py
│       ├── compare_maxn.py
│       ├── project_analysis.py
│       └── failure_analysis.py
└── shared/
```

Use production AddaxAI detector/tracker code wherever possible.

---

# 72. Suggested Experiment 2 configuration

Example only:

```yaml
experiment_2:
  random_seed: 42

  addaxai_event_definition:
    independence_interval_minutes: 30
    rule: consecutive_gap

  datasets:
    iwildcam2022:
      target_samples: 1780
    snapshot_serengeti_gold:
      target_samples: 1200
    snapshot_safari_2024:
      target_samples: 1200
    snapshot_usa_2024:
      target_samples: 820

  target_total_samples: 5000

  tracking_windows_seconds:
    - 0.5
    - 1
    - 2
    - 5
    - 10
    - 30
    - 60
    - 300
    - 1800

  baselines:
    maxn: true
    per_image_sum: true

  metrics:
    - exact_count_accuracy
    - mae
    - signed_error
    - overcount_rate
    - undercount_rate
    - within_one_accuracy
```

---

# 73. Expected final report for Experiment 2

The report should include:

## Dataset composition

```text
number of events
number of images
species
projects
locations
group sizes
inter-image interval distribution
```

## MaxN baseline

Report overall and per interval.

## TrackCount

Report overall and per interval.

## TrackCount versus MaxN

This is the main product comparison.

## Temporal curve

Plot:

```text
tracking benefit
versus
inter-image interval
```

## Tracking-window comparison

Compare candidate maximum association gaps.

## Project-level performance

Determine whether results generalise across field projects.

## Species/group-size analysis

Determine where tracking helps most.

## Failure analysis

Explain why it fails when temporal gaps become larger.

## Recommendation

Finish with a concrete product recommendation.

Examples:

```text
Use still-image tracking only for consecutive images <= 2 seconds apart.
```

or:

```text
Tracking does not improve enough over MaxN to justify adding it.
```

or:

```text
Use tracking for native rapid bursts, but do not continue tracks across
the full 30-minute AddaxAI ecological event.
```

---

# 74. Questions Experiment 2 must answer

At completion, we should be able to answer:

1. Does tracking improve animal counts for still-image camera traps?
2. Does it outperform MaxN?
3. At what inter-image interval does the advantage disappear?
4. At what interval does tracking become actively harmful?
5. Does the threshold differ substantially between datasets/projects?
6. Does it differ between species?
7. Does it differ with group size?
8. Does it differ between day and night?
9. Does it work only for explicit camera bursts?
10. Does it provide any benefit for irregular/non-burst sequences?
11. Can the existing video tracker be reused?
12. Should AddaxAI introduce a still-image tracking-gap parameter?
13. Should that tracking gap be separate from the normal 30-minute ecological event threshold?
14. Is one global default sufficiently robust for AddaxAI users?

---

# 75. Relationship between the two experiments

The two experiments answer related but different questions.

## Experiment 1: video

```text
continuous video
↓
detector sampling FPS
↓
tracker
↓
TrackCount
```

Main question:

> How sparse can video detections become before track-count accuracy becomes unacceptable?

## Experiment 2: still images

```text
timestamped photographs
↓
detections
↓
tracker / temporal association
↓
TrackCount
```

Main question:

> How far apart can still photographs be before temporal tracking stops improving animal-count estimates?

Together, they should provide an empirical basis for one coherent AddaxAI counting architecture:

```text
VIDEO
frames → detections → tracks → suggested count

STILL IMAGES
event → timestamped detections → temporal association → suggested count
```

while preserving the ecological distinction:

```text
AddaxAI event definition
≠
maximum defensible tracking interval
```

The 30-minute event threshold remains an ecological grouping parameter.

The tracking interval, if Experiment 2 supports one, is a computer-vision association parameter.

---

# 76. Final instruction to the future coding agent for Experiment 2

As with Experiment 1, **validate the datasets before building the benchmark runner**.

For each dataset:

1. download metadata first
2. inspect sequence/event structure
3. verify timestamps
4. verify count semantics
5. determine whether count is per image or per sequence/event
6. verify species/count handling for multi-species events
7. inspect a small number manually
8. document uncertainties
9. only then download/build the full image sample
10. freeze the benchmark manifest before evaluating production tracking

Do not infer a unique biological event count by summing per-image counts.

Do not assume a dataset's "sequence" definition is identical to AddaxAI's 30-minute event definition.

Do not assume tracking should span the whole AddaxAI event.

The key deliverable is a defensible answer to:

> **For real camera-trap still-image data from diverse projects and camera configurations, does tracking across photographs improve the suggested animal count, and how does that improvement change as the time between photographs increases?**



---

# Experiment 3: tracker and association optimisation

# 77. Purpose

Experiments 1 and 2 establish how well the **current AddaxAI tracking/counting setup** performs on:

```text
Experiment 1:
camera-trap video sampled at different FPS values

Experiment 2:
still-image camera-trap events with different temporal gaps
```

Once those two baseline experiments are complete, a third experiment should use the **same frozen benchmark datasets** to determine whether the counting performance can be improved by changing:

```text
tracker implementation
tracking package
association mechanism
time-delta handling
motion model
appearance/ReID support
track lifecycle settings
matching thresholds
other tracker parameters
```

The central question becomes:

> **Given the same detections and the same ground truth, which tracking approach gives AddaxAI the most accurate suggested animal counts across both video and still-image camera-trap data?**

The objective is not to find the tracker with the best generic multi-object-tracking leaderboard score.

The objective is:

> **maximise animal-count accuracy for AddaxAI's actual use case.**

---

# 78. Why Experiment 3 comes after Experiments 1 and 2

Experiments 1 and 2 should be completed first.

They provide:

```text
fixed datasets
fixed ground truth
cached detector outputs
baseline tracker results
known failure cases
known difficult temporal intervals
```

This makes Experiment 3 much cheaper and scientifically cleaner.

The detector should generally **not need to be rerun**.

For Experiment 1:

```text
video
↓
cached high-FPS results.json
↓
subsample detections
↓
different trackers
```

For Experiment 2:

```text
still images
↓
cached image detections
↓
timestamped event detections
↓
different trackers / association methods
```

Therefore most of Experiment 3 should operate only on existing detector outputs.

This allows many tracker configurations to be tested for relatively little compute compared with rerunning the neural network.

---

# 79. Main goals

Experiment 3 should determine:

1. Whether another tracker improves count accuracy over the current AddaxAI tracker.
2. Whether the best tracker differs between video and still-image events.
3. Whether explicitly using real elapsed time (`Δt`) improves association.
4. Whether trackers that assume consecutive equally spaced frames perform poorly on sparse or irregular camera-trap data.
5. Whether appearance/ReID features materially improve animal counting.
6. Whether simple motion-only tracking is sufficient.
7. Which parameters most strongly affect overcounting and undercounting.
8. Whether one configuration can work well across both experiments.
9. Whether separate video and still-image tracker configurations are justified.
10. Whether improvements are large enough to justify additional implementation complexity or dependencies.

---

# 80. Main hypotheses

## Hypothesis 1

> **Different trackers will produce meaningfully different animal-count accuracy even when given identical detections.**

The largest differences are expected in difficult cases involving:

```text
sparse observations
fast movement
multiple animals
occlusion
entry/exit
irregular timing
```

## Hypothesis 2

> **Trackers that can correctly account for real time delta will perform better when observation intervals are irregular or large.**

This is especially relevant to Experiment 2, where still-image gaps may vary from fractions of a second to minutes.

## Hypothesis 3

> **For dense video observations, simple motion-based trackers may perform nearly as well as more complex approaches.**

If this is true, AddaxAI should prefer the simpler method.

## Hypothesis 4

> **Appearance/ReID information may reduce fragmentation and identity confusion in sparse observations, but may not justify its compute and implementation cost unless the improvement in count accuracy is substantial.**

## Hypothesis 5

> **Tracker parameters that are optimal for generic MOT benchmarks may not be optimal for counting camera-trap animals.**

The optimisation target should therefore be AddaxAI count accuracy, not only IDF1/HOTA/MOTA.

---

# 81. Freeze the benchmark before tracker optimisation

Before Experiment 3 begins:

```text
Experiment 1 benchmark manifest
Experiment 2 benchmark manifest
```

must be frozen.

Do not change the video/event sample in response to tracker results.

Likewise, freeze:

```text
ground-truth counts
detector outputs
confidence thresholds
species filtering rules
evaluation metrics
```

unless the purpose of a specific subexperiment is explicitly to test one of those components.

This prevents tracker optimisation from becoming mixed with dataset or detector changes.

---

# 82. Reuse exactly the same detections

Every tracker should receive the same detector observations.

For Experiment 1:

```text
same results.json
same selected timestamps
same bounding boxes
same confidence values
same classes
```

For Experiment 2:

```text
same images
same detections
same timestamps
same event definitions
```

The only variable should initially be:

```text
tracking / association
```

This gives a fair comparison.

---

# 83. Tracker families to compare

Do not commit in advance to a specific package list.

At implementation time, inspect maintained Python tracking libraries and select several representative tracker families.

At minimum, compare approaches from the following categories.

## 83.1 SORT-style tracking

Characteristics:

```text
Kalman motion prediction
IoU-based association
Hungarian assignment
little or no appearance information
```

Advantages:

```text
simple
fast
low dependency burden
easy to understand
```

Potential weakness:

```text
sparse observations and large motion between frames
```

SORT-style methods provide a useful simple baseline.

---

## 83.2 ByteTrack-style tracking

Characteristics:

```text
motion-based association
uses high- and lower-confidence detections
strong practical MOT baseline
```

Potential advantages:

```text
robust handling of detector confidence
fast
widely used
```

Potential questions:

```text
How does it behave at 1 FPS?
How does it behave with 5-second image gaps?
Can its motion model correctly account for variable Δt?
```

---

## 83.3 BoT-SORT-style tracking

Characteristics may include:

```text
motion model
camera-motion compensation
appearance/ReID
more sophisticated association
```

Potential advantage:

```text
better identity continuity
```

Potential disadvantage:

```text
greater complexity
additional dependencies
higher compute
ReID models may not transfer well to wildlife
```

---

## 83.4 DeepSORT-style tracking

Characteristics:

```text
motion prediction
appearance embeddings / ReID
association based on motion + visual similarity
```

This is useful for determining whether appearance information helps with sparse camera-trap observations.

Important caveat:

Most generic ReID models are trained primarily on people or conventional MOT data.

Do not assume that a generic ReID embedding is useful for:

```text
deer
foxes
birds
chimpanzees
small mammals
etc.
```

Measure it.

---

## 83.5 OC-SORT / observation-centric methods

Observation-centric tracking approaches should be considered because they can behave differently from conventional Kalman-filter trackers when motion is nonlinear or detections are intermittent.

These may be useful for:

```text
camera-trap animals entering suddenly
large displacement between detections
nonlinear movement
short tracklets
```

---

## 83.6 Other maintained trackers

At the time Experiment 3 is implemented, inspect the current tracking ecosystem for other maintained and relevant packages.

Do not restrict the comparison to libraries named in this document if better approaches are available later.

However, avoid adding trackers merely to create a large leaderboard.

A focused set of representative algorithms is preferable.

---

# 84. Tracking mechanisms beyond standard MOT packages

Experiment 3 may also compare mechanisms that are not conventional frame-by-frame MOT trackers.

These could include:

```text
IoU-only association
centroid-distance association
motion-model association
appearance-embedding association
hybrid geometry + appearance association
optical-flow-assisted tracking
single-object visual tracking
segmentation propagation
species-aware association
```

Not all of these should automatically enter production.

Their purpose is to establish whether a fundamentally different association mechanism provides a meaningful count benefit.

The default principle remains:

```text
prefer simple, robust methods
unless extra complexity produces a meaningful measured improvement
```

---

# 85. Time delta is a primary variable

A major goal of Experiment 3 is to understand whether candidate trackers can use **real elapsed time**.

This is critical because camera-trap observations are not always evenly spaced.

Examples:

```text
video at 1 FPS:
Δt = 1 second

video at 2 FPS:
Δt = 0.5 seconds

still burst:
Δt = 0.7 seconds

still sequence:
Δt = 4.3 seconds

irregular event:
0.8 s → 1.1 s → 5.4 s → 23 s
```

A tracker that always assumes:

```text
Δt = 1 frame
```

may make poor motion predictions when actual elapsed time changes substantially.

Experiment 3 should explicitly determine whether each tracker:

```text
supports variable Δt directly
can be modified to support Δt
can approximate Δt through frame indices
assumes constant frame intervals
cannot meaningfully support irregular timing
```

Document this per tracker.

---

# 86. Tracker time-delta capability table

Create a table similar to:

| Tracker | Native variable Δt | Can modify | Appearance/ReID | Motion model | Notes |
|---|---|---|---|---|---|
| Current AddaxAI tracker | ? | ? | ? | ? | baseline |
| SORT | investigate | likely | no | Kalman | simple |
| ByteTrack | investigate | likely | usually no | Kalman | strong baseline |
| BoT-SORT | investigate | likely | optional/yes | Kalman | more complex |
| DeepSORT | investigate | likely | yes | Kalman | appearance-based |
| OC-SORT | investigate | investigate | usually no | observation-centric | sparse motion candidate |

Do not fill this table from assumptions.

Inspect the actual version/package source code used in the experiment.

Tracker APIs and implementations differ between packages even when they use the same algorithm name.

---

# 87. How to test time-delta support

For trackers with explicit variable-time support:

```python
tracker.update(detections, dt=actual_seconds)
```

or equivalent functionality should be used.

For trackers without explicit support but with accessible motion models, investigate whether the state-transition model can be updated according to real elapsed time.

For example, conceptually:

```text
position prediction:
x(t + Δt)
```

rather than assuming:

```text
x(t + 1 frame)
```

If a tracker cannot support variable Δt cleanly, compare:

```text
standard tracker
versus
modified Δt-aware tracker
```

where practical.

This should be treated as an experimental branch, not an automatic production rewrite.

---

# 88. Controlled Δt experiment

For each tracker capable of both modes, compare:

```text
A. constant-time assumption
B. actual timestamp Δt
```

Use exactly the same detections.

This provides a direct answer to:

> Does using real elapsed time materially improve animal-count accuracy?

Run this separately on:

```text
Experiment 1 video benchmark
Experiment 2 still-image benchmark
```

The expected benefit may be small for regular video and much larger for irregular still images.

---

# 89. Parameter optimisation

Each tracker has parameters that can strongly influence track count.

Examples include:

```text
association threshold
IoU threshold
maximum distance
matching score threshold
track buffer
max_age
minimum hits
track confirmation threshold
lost-track timeout
confidence thresholds
ReID similarity threshold
motion gating threshold
```

These settings can directly change count bias.

For example:

```text
short max_age
→ more fragmented tracks
→ likely overcount

long max_age
→ more opportunities to reconnect
→ potentially fewer fragments
→ but more risk of merging different animals
```

Therefore parameters should be optimised against:

```text
count accuracy
```

rather than only generic tracking metrics.

---

# 90. Express temporal parameters in seconds where possible

A parameter such as:

```text
max_age = 30 frames
```

has different meaning at:

```text
30 FPS
10 FPS
1 FPS
0.5 FPS
```

For camera-trap applications, temporal parameters should preferably have a real-time interpretation.

For example:

```text
max_age_seconds = 2.0
```

which can then be converted internally according to timestamps/FPS.

Experiment 3 should investigate whether converting frame-based tracker settings into **time-based settings** improves consistency across:

```text
different video FPS values
different camera burst intervals
different projects
```

This may be one of the most practically important outcomes of the experiment.

---

# 91. Development set versus final evaluation set

Do not tune tracker parameters on the full benchmark and then report performance on the same data as if it were independent.

Split the fixed benchmarks into:

```text
development/tuning subset
held-out evaluation subset
```

Prefer splitting by:

```text
camera location
project
```

rather than randomly by individual event where possible.

This reduces leakage from nearly identical sequences.

A possible split:

```text
70% development
30% final evaluation
```

or:

```text
80% development
20% final evaluation
```

The exact split can depend on dataset size.

Once parameters are selected, freeze them and evaluate once on the held-out set.

---

# 92. Optimisation objective

The primary optimisation objective should be:

```text
exact count accuracy
```

with secondary consideration of:

```text
mean absolute count error
overcount rate
undercount rate
catastrophic error rate
runtime
memory usage
implementation complexity
```

Do not select a tracker solely because it achieves slightly higher exact accuracy if it introduces major costs.

Example:

```text
Tracker A:
91.2% accuracy
very simple
negligible compute

Tracker B:
91.7% accuracy
large ReID model
GPU dependency
much higher maintenance
```

Tracker A may be the better AddaxAI choice.

---

# 93. Catastrophic errors

Experiment 3 should explicitly measure large mistakes.

For example:

```text
abs(predicted_count - true_count) >= 3
```

or another suitable threshold.

A tracker with slightly better average accuracy but occasional severe merges may be less desirable than a conservative tracker with small overcounts.

Report:

```text
catastrophic overcount rate
catastrophic undercount rate
```

---

# 94. Video-specific tracker evaluation

Use Experiment 1 to compare trackers under:

```text
0.5 FPS
1 FPS
2 FPS
5 FPS
10 FPS
```

Important questions:

1. Which tracker degrades least as FPS decreases?
2. Does appearance information become useful below 2 FPS?
3. Does Δt-aware motion prediction help?
4. Which tracker produces the least fragmentation at 1 FPS?
5. Does improving fragmentation introduce more incorrect merges?
6. Which tracker gives the best count accuracy at the intended production FPS?

The main production target remains likely around:

```text
1 FPS
```

unless Experiment 1 establishes another preferred operating point.

---

# 95. Still-image-specific tracker evaluation

Use Experiment 2 to compare trackers across real inter-image gaps.

Important questions:

1. Which tracker works best for rapid bursts?
2. Which remains useful at 2–5 second gaps?
3. Does any tracker retain useful performance at longer intervals?
4. Does actual `Δt` substantially improve motion prediction?
5. Does appearance/ReID extend the useful tracking interval?
6. At what point should all trackers stop attempting association?
7. Is a simple event-association algorithm better suited to still images than a standard video MOT tracker?

This experiment may show that the optimal still-image method is **not** the same tracker used for video.

That is acceptable if the improvement is substantial and the additional complexity remains manageable.

---

# 96. Test tracking threshold jointly with tracker choice

Experiment 2 introduces the concept of a maximum still-image tracking gap.

Experiment 3 should optimise this jointly with tracker choice.

For example:

```text
Tracker A + 1-second window
Tracker A + 2-second window
Tracker A + 5-second window

Tracker B + 1-second window
Tracker B + 2-second window
Tracker B + 5-second window
```

This avoids incorrectly concluding that one tracker is poor simply because it was evaluated with an unsuitable time window.

---

# 97. Appearance/ReID subexperiment

For trackers supporting visual appearance embeddings, explicitly test:

```text
motion only
versus
motion + appearance
```

This matters because wildlife is different from pedestrian tracking.

Potential outcomes:

### Appearance helps substantially

This could reduce:

```text
track fragmentation
incorrect association
```

especially with sparse observations.

### Appearance provides little benefit

Do not retain the additional model/dependency.

### Appearance harms performance

Generic ReID embeddings may confuse visually similar animals or behave unpredictably across infrared/night imagery.

Again, measure rather than assume.

---

# 98. Species-aware association

An obvious rule is:

```text
detections of different species should not normally belong to the same track
```

Where the detector provides species labels with sufficient reliability, compare:

```text
class-agnostic tracking
versus
species-aware tracking
```

However, be cautious.

Species classification itself can fluctuate between frames.

Example:

```text
frame 1 → red deer
frame 2 → roe deer
frame 3 → red deer
```

A strict species rule could fragment one real animal.

Potential strategies include:

```text
track at animal level, classify track afterward
soft species compatibility
majority species per track
strict species gating
```

Experiment 3 can test these approaches if classification instability is a real source of errors.

---

# 99. Tracker parameter search strategy

Do not perform an enormous blind hyperparameter search.

Start with:

```text
package defaults
current AddaxAI settings
small sensible parameter grid
```

Identify which parameters materially affect count errors.

Then narrow the search.

A staged process:

```text
Stage 1:
compare tracker families using sensible/default parameters

Stage 2:
select top 2–3 trackers

Stage 3:
tune important parameters on development data

Stage 4:
compare Δt-aware variants

Stage 5:
evaluate frozen configurations on held-out data
```

This prevents compute and engineering effort from exploding unnecessarily.

---

# 100. Required tracker result schema

Extend the benchmark result table with:

```text
tracker_name
tracker_package
tracker_version
tracker_config_id
uses_actual_dt
uses_reid
tracking_window_seconds
runtime_seconds
```

Example:

```csv
dataset,sample_id,evaluation_fps,tracker_name,tracker_config_id,uses_actual_dt,ground_truth_count,predicted_track_count,exact_match,absolute_count_error
SA-FARI,video_123,1.0,ByteTrack,bt_dt_v3,true,3,3,true,0
```

Keep full tracker configuration files under version control.

---

# 101. Runtime and dependency cost

For every candidate tracker record:

```text
CPU/GPU runtime
memory requirements
additional model weights
package size
installation complexity
platform compatibility
license
maintenance status
```

AddaxAI is a desktop application.

A tracker that depends on a difficult GPU stack or large extra model may be unattractive even if its benchmark accuracy is slightly higher.

The final recommendation must consider deployment practicality.

---

# 102. Cross-platform constraints

AddaxAI should remain practical on its supported desktop environments.

Before recommending a tracker, verify:

```text
Windows compatibility
macOS compatibility
CPU support where required
GPU requirements
Python/conda compatibility
licensing
packaging implications
```

A tracker that performs well only in a research environment but is difficult to bundle reliably may not be appropriate.

---

# 103. Tracker licence review

Before integrating any external tracking package into AddaxAI, check:

```text
software licence
model-weight licence
commercial/non-commercial restrictions
redistribution rights
attribution requirements
dependency licences
```

This must be done for the exact package/version/model used.

Do not assume the algorithm's academic paper determines the code licence.

---

# 104. Expected comparison table

The final Experiment 3 report should include a table similar to:

| Tracker/config | Video 1 FPS exact | Still burst exact | Still 2–5 s exact | Overall MAE | Uses Δt | ReID | Relative runtime |
|---|---:|---:|---:|---:|---|---|---:|
| Current AddaxAI | ... | ... | ... | ... | ... | ... | 1.0× |
| SORT variant | ... | ... | ... | ... | ... | ... | ... |
| ByteTrack variant | ... | ... | ... | ... | ... | ... | ... |
| OC-SORT variant | ... | ... | ... | ... | ... | ... | ... |
| BoT-SORT variant | ... | ... | ... | ... | ... | ... | ... |
| DeepSORT variant | ... | ... | ... | ... | ... | ... | ... |

The exact algorithms may change when the experiment is actually implemented.

---

# 105. Improvement should be measured against baseline

For every candidate configuration calculate:

```text
Δ exact count accuracy
Δ MAE
Δ overcount rate
Δ undercount rate
Δ catastrophic error rate
Δ runtime
```

relative to the current AddaxAI tracker.

Example:

```text
Current tracker:
86.2% exact

Candidate:
90.4% exact

Improvement:
+4.2 percentage points
```

This makes the engineering trade-off explicit.

---

# 106. Avoid optimising for one dataset only

A tracker may perform extremely well on:

```text
PanAf500
```

but poorly on:

```text
SA-FARI
```

or vice versa.

Likewise, a tracker tuned for:

```text
rapid iWildCam bursts
```

may fail on longer SNAPSHOT USA intervals.

Report both:

```text
overall result
per-dataset result
```

A robust moderate improvement across all datasets is preferable to a large improvement on one dataset and regression elsewhere.

---

# 107. Generalisation test

After selecting the best candidate tracker/configuration, perform a final generalisation check.

Ideally use:

```text
a project/location not used during tuning
```

or hold out one complete constituent project where dataset size allows.

This answers:

> Does the improvement transfer to an unseen camera-trap project?

This is particularly important because AddaxAI is used across many different wildlife projects.

---

# 108. Possible final product architectures

Experiment 3 may support several outcomes.

## Outcome A: one tracker for everything

```text
video
still bursts
irregular image events
```

all use the same tracker and time-aware configuration.

This is the simplest preferred outcome.

## Outcome B: same tracker, different settings

Example:

```text
video:
max_age_seconds = 2

still images:
max_age_seconds = 5
```

This retains one implementation while allowing domain-specific settings.

## Outcome C: different association mechanisms

Example:

```text
video → ByteTrack-style MOT
still images → time-aware geometry/appearance association
```

Only choose this if the accuracy improvement clearly justifies the extra complexity.

## Outcome D: current tracker is already good enough

If alternatives produce negligible improvement, keep the current implementation.

This is a successful experimental outcome.

---

# 109. Recommended decision rule

Do not replace the current tracker because another tracker is academically more sophisticated.

Replace or modify it only if there is a meaningful improvement in the actual product metric:

```text
verified animal-count accuracy
```

while maintaining acceptable:

```text
runtime
robustness
licensing
packaging
maintainability
cross-platform support
```

A small accuracy gain may not justify substantial complexity.

---

# 110. Relationship between all three experiments

The complete benchmark programme becomes:

## Experiment 1

> **How does video sampling FPS affect tracking-based animal counts?**

```text
video
→ cached detections
→ different FPS subsets
→ current tracker
→ count accuracy
```

## Experiment 2

> **Does tracking across still photographs improve animal counts, and how does performance change with temporal gap?**

```text
image event
→ timestamped detections
→ current tracker
→ count accuracy versus interval
```

## Experiment 3

> **Given the findings and fixed data from Experiments 1 and 2, can a better tracker, association method, time-delta implementation or configuration improve the counts?**

```text
same cached detections
             ↓
 ┌───────────┼───────────┐
 │           │           │
tracker A  tracker B   tracker C
 │           │           │
count       count       count
 └───────────┼───────────┘
             ↓
     compare with ground truth
```

This ordering is important.

First understand the current system.

Then improve it.

---

# 111. Final instruction to the future coding agent for Experiment 3

Do not start Experiment 3 until the baseline results from Experiments 1 and 2 are reproducible.

Then:

1. freeze both benchmark manifests
2. freeze cached detector outputs
3. record the current AddaxAI tracker as the baseline
4. inspect the current tracker implementation and whether it already supports real `Δt`
5. select a small representative set of maintained tracker families
6. inspect the exact source/API of each for variable-time support
7. benchmark defaults first
8. select the strongest candidates
9. tune only on a development subset
10. explicitly compare constant-time versus real-`Δt` operation where possible
11. freeze final configurations
12. evaluate on held-out data
13. compare count accuracy, errors, runtime and implementation cost
14. recommend the simplest approach that produces a meaningful improvement

The final question is:

> **What is the simplest tracking or temporal-association approach that gives AddaxAI the most accurate and robust animal-count suggestions across both camera-trap video and still-image events?**

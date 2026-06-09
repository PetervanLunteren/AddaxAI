# Observation count audit: box-less observations vs an explicit count field

Status: investigation complete, no code written. This is decision support for
how AddaxAI represents "number of individuals", especially the hidden-individual
case (3 deer walk through a video but only 2 are visible in any single frame, so
the 3rd has no box; or 5 elephants pass an event while no single image shows 5).

Addresses TODO line: "add an observation without a box ... now it writes fake
detections to a frame number which is also not true ... how would that work in
the CSV output? Should we add that to the events view too? Do a full audit."

---

## TL;DR

- Today an "extra individual" is stored as a real `Detection` row with all four
  bbox columns NULL, `verified=True`, and `frame_number` set to the video's
  `best_frame_number`. That last part is the "fake frame number": the animal is
  not actually in the best frame, we just anchor it there so the count math works.
- The count of individuals is never stored as a number. It is derived by counting
  detection rows per frame (MaxN). A box-less observation is "+1 row".
- This leaks into outputs three ways: recognition.json silently drops box-less
  observations (data loss for Timelapse users), the Camtrap-DP export is
  non-conformant (it writes N event-level rows of `count=1` instead of one row of
  `count=N`), and the flat CSV has no count column at all.
- Every other camera-trap platform represents abundance as an explicit count
  number, almost always at the sequence/event level, decoupled from boxes. No one
  else makes you click once per hidden individual.
- The data standards (Camtrap-DP, Darwin Core, GBIF) all have a first-class
  `count` / `individualCount` field, and Camtrap-DP separates media-level boxes
  from event-level counts via `observationLevel`.
- Recommendation: option A, an explicit human-authoritative count per species per
  event, seeded from the detector's MaxN and editable upward (the Wildlife
  Insights pattern), with boxes kept only for localization. Retire the fake-frame
  box-less rows. This is also the cheapest path to a correct Camtrap-DP export.

---

## 1. How it works today

### 1.1 Adding an observation without a box (UI)

There is a binoculars button (keyboard shortcut `N`, tooltip "Add observation
without a bounding box") in two places:

- `frontend/src/components/verify/EventDetailModal.tsx:918` (button), `:421`
  (`observeMutation`), `:799` (the `N` key).
- `frontend/src/components/verify/FileDetailModal.tsx:589` (button), `:289`
  (`observeMutation`), `:482` (the `N` key).

Both call `detectionsApi.createObservation({ file_id, category, label })`. No box
is drawn, no frame is chosen. The active label (the species the user is tagging)
is auto-defaulted to the most common label on the file and can be overridden with
the tag picker. To record three hidden deer the user presses `N` three times.

### 1.2 What gets stored (the fake frame number)

Endpoint: `backend/app/api/routers/detections.py:62` (`POST /api/detections/observation`).
CRUD: `backend/app/api/crud/detection.py:246` (`create_observation`).

```python
file = db.query(File).filter(File.id == data.file_id).first()
frame_number = file.best_frame_number if file else None   # the "fake" frame
db_detection = Detection(
    category=data.category,
    confidence=1.0,
    bbox_x=None, bbox_y=None, bbox_width=None, bbox_height=None,
    label=data.label,
    classification_method="human",
    frame_number=frame_number,
    verified=True,
    verified_at_utc=datetime.now(UTC),
)
```

So a box-less observation is a normal detection row with:

| column | value |
|---|---|
| `bbox_x/y/width/height` | NULL (set-or-null together) |
| `confidence` | 1.0 |
| `classification_method` | "human" |
| `verified` | True |
| `frame_number` | the video's `best_frame_number`, or NULL for images |

The code comment (`detection.py:257`) is honest about the trade: anchoring to the
best frame puts the observation in the same MaxN group as the AI's best-frame
boxes so the group counts correctly, at the cost of claiming the animal is in a
frame where it is not. For an image, `best_frame_number` is NULL, so an image
observation has a NULL frame and no lie. The lie only exists for videos.

There is no count column anywhere. `Detection` has no count
(`backend/app/models/detection.py`). Each individual is one row.

### 1.3 How it becomes a count (MaxN)

`backend/app/api/crud/event_observation.py:30` (`calculate_max_n_for_event`)
groups detections by `(file_id, frame_number, label_taxonomy_id)`, counts the rows
in each group, and takes the maximum across frames per species. The result is
stored in `EventObservation.max_n` (`backend/app/models/event_observation.py:29`),
one row per species per event, with `max_n_file_id` pointing at the peak frame.

Because box-less observations are anchored to the best frame, they add to that
frame's group. Example: AI finds 2 deer in the best frame, the user presses `N`
three times, the best-frame group now has 5 rows, MaxN for the event is 5. Without
the anchoring they would land in a NULL-frame group and MaxN would be max(2,3)=3.
So the fake frame number exists purely to make the additive count work.

This means the current model already has an event-level count (`max_n`), it is
just derived by counting rows rather than stored as a number, and the only way a
human can push it above the per-frame box count is to insert fake rows.

### 1.4 How it flows into outputs

CSV / XLSX (`backend/app/api/crud/export.py:106` headers, `:504` cells): one row
per detection. A box-less observation is a row with empty strings for the four
bbox columns and the `frame_number` of the best frame. There is no count column;
abundance is implicit in the number of rows. A consumer cannot tell "5 elephants
in this event" without grouping and replicating the MaxN logic themselves.

recognition.json (`backend/app/ml/postprocessing_outputs/recognition_json.py:101`,
`:235`): box-less observations are skipped entirely.

```python
bbox = _bbox_for_detection(det)
if bbox is None:
    # Event-level observation, no spatial annotation to write. The
    # canonical Timelapse / AddaxAI JSON has no representation for this.
    continue
```

This is silent data loss: a verifier adds the hidden deer, exports recognition.json
for Timelapse, and the deer is gone. Defensible (the MegaDetector format has no
count field and no event level), but worth stating as an accepted limitation
rather than a surprise.

Camtrap-DP (`backend/app/api/crud/export.py:184` headers, `:837` rows): each
detection becomes one observation row. Box-less rows get `observationLevel="event"`
with empty bbox columns, boxed rows get `observationLevel="media"`. The `count`
column is hardcoded to `1` on every row. This is non-conformant: the spec says
event-level rows "should be mutually exclusive so that their count can be summed".
Today an event with 5 deer emits 5 event-level rows of `count=1` (some boxed, some
not), so a downstream tool that sums event-level counts will either double count
(event rows plus media rows) or, summing only event rows, get the right number by
accident only because we emit one row per individual. Either way it is not the
intended one-row-per-species-per-event shape.

Dashboard stats (`backend/app/api/crud/statistics.py:264`): total observations is
`sum(EventObservation.max_n)`, so box-less observations contribute correctly to the
headline number. Stats are the one place the current model behaves well.

### 1.5 Events view

The event detail (`EventDetailModal.tsx`) shows MaxN per species (`MaxN: {label}
×{max_n}`) and a thumbnail of the peak frame. The user can add a box-less
observation here too (same `N` key). There is no number the user can edit; to
change the count they add or delete rows. So the event view already thinks in
events and counts, it just expresses the count as a pile of rows.

---

## 2. Is the current approach confusing?

Short answer: yes, in three ways.

1. The mental model is unusual. Every other tool asks "how many?" and you type a
   number. AddaxAI asks you to press `N` once per invisible animal. A user who has
   used Agouti, Wildlife Insights, eMammal, or Zooniverse will look for a count
   field and not find one. This is a "match between system and the real world"
   and "consistency with the outside world" gap (Nielsen Norman heuristics 2 and 4).

2. The stored data lies. `frame_number = best_frame_number` says the animal is in a
   frame it is not in. Anything that later trusts `frame_number` (per-frame review,
   time-of-frame, future per-frame export) inherits a wrong value. It also makes
   "observation" and "detection" the same table row with different meanings, which
   is hard to reason about.

3. The outputs disagree with each other. The dashboard counts the hidden deer, the
   CSV shows it as a bbox-less row, recognition.json drops it, and Camtrap-DP
   emits it as a `count=1` event row. Same animal, four different representations,
   one of them missing. That is the opposite of a single source of truth.

The "binoculars = I saw it but did not box it" icon is a nice touch and the
intent is right. The problem is the representation, not the feature.

---

## 3. How other platforms do it

Every system below treats the count of individuals as an explicit value, not as a
function of boxes. The split is whether that value lives per image or per sequence,
and whether boxes exist at all.

| Platform | Unit | Count field | Boxes? | Hidden-individual rule |
|---|---|---|---|---|
| Agouti | sequence (event) | "amount" | no | single per-sequence number; model allows it, no explicit doc rule |
| Camelot | per photo, merged into sightings | "Sighting Quantity" | no | independence window merges frames; max-vs-sum undocumented |
| TRAPPER | sequence (event) | "Count" (default 1) | optional | explicit: classify whole group in one sequence, take the max per sequence |
| Wildlife Insights | sequence (60s) | "Group size" | only from MegaDetector | auto-seed = max boxes across frames, human edits upward |
| WildTrax | per image | "Count (camera)" | display only | per-image count; "TMTT" (too many to tag) escape hatch; merged downstream |
| eMammal | sequence | "Group_Size" (1-50) | no (uses foreground detection) | reviewer scrolls the sequence and types one number |
| Zooniverse / Snapshot Serengeti | capture event (3-image burst) | "How many?" buckets 1-10, 11-50, 51+ | no | explicit: "if a 5th zebra walks into frame 2, record five"; count is the median across volunteers |
| TrapTagger | cluster of frames | count derived from boxes | yes (core) | no native concept; relies on the animal appearing in some frame |
| MegaDetector | per image | none (boxes only) | yes (core) | cannot represent it at all |

Key observations for our decision:

- The two purpose-built box tools (MegaDetector, TrapTagger) are exactly the two
  that cannot express a hidden individual. Box-derived counts cannot exceed what is
  visible. This is the structural reason AddaxAI invented box-less rows in the
  first place.
- The sequence-native tools (Agouti, TRAPPER, eMammal, Wildlife Insights sequence
  mode, Zooniverse) all put one count number on the event. That is the natural home
  for "5 elephants across the event".
- Wildlife Insights is the closest precedent to where AddaxAI already is: it seeds
  group size from the maximum bounding-box count across frames, then lets the human
  override it. AddaxAI already computes that maximum (MaxN). The missing piece is
  the editable number.
- TRAPPER states the rule we want almost verbatim: "with big groups, or groups
  where animals are never all in the picture, classify the whole group in one
  sequence ... by just selecting the maximum per sequence."
- Useful escape hatches exist for large herds: WildTrax "TMTT" and Zooniverse's
  11-50 / 51+ buckets. Worth keeping in mind, not core.

---

## 4. What the standards say

Camtrap-DP (https://camtrap-dp.tdwg.org/data/) is the de facto standard and the
one our export already targets. Two fields matter:

- `observationLevel` (required): `media` for a classification tied to one media
  file (carries `mediaID` and optional `bboxX/Y/Width/Height`, "useful for machine
  learning", rows need not be mutually exclusive) or `event` for a classification
  of a whole sequence (linked by `eventID`, "useful for ecological research",
  rows "should be mutually exclusive").
- `count` (optional, integer, minimum 1): "Number of observed individuals
  (optionally of given life stage, sex and behavior)." On an event-level row this
  is the authoritative "how many in this sequence" number.

So the standard explicitly separates per-image boxes (media level) from the
event-level count, and bridges them with `eventID`. The intended shape for our
case is one event-level row per species (`observationLevel=event`,
`scientificName`, `count=5`) plus, optionally, media-level rows for the boxes of
the individuals that were visible. The count is not constrained to the max boxes
in any frame.

Darwin Core (https://dwc.tdwg.org/terms/), the broader standard Camtrap-DP maps to
for GBIF, uses `dwc:individualCount` ("The number of individuals present at the
time of the dwc:Occurrence"), or the pair `dwc:organismQuantity` +
`dwc:organismQuantityType` (for example 5 + "individuals"). There is no bbox term
in Darwin Core at all; counts live on the occurrence, fully decoupled from image
geometry. An event-level Camtrap-DP `count` maps straight to `dwc:individualCount`.

GBIF camera-trap guide (https://docs.gbif.org/camera-trap-guide/en/) is the most
explicit on our scenario. It recommends event-based observations as the default
for ecological use: "All media files that belong to the event are assessed as a
whole to determine the species and their number of individuals." It treats the
count as a property of the assessed-as-a-whole sequence, not a function of any
single frame, and notes that media-based (per image) classification "can
substantially increase the number of occurrences, with little added benefit for
ecological research."

FAIR (https://www.go-fair.org/fair-principles/): principles I1, I2 and R1.3 push
toward storing the count in the standard, named field (Camtrap-DP `count`, Darwin
Core `individualCount`) rather than an app-specific representation, so the data is
interoperable with GBIF and reusable by other tools. A pile of NULL-bbox rows with
a fake frame number is the opposite of that.

Net: the standards, the GBIF guidance, and the FAIR principles all point to the
same thing, an explicit event-level count field.

---

## 5. Recommendation

The three options below match the framing from the kickoff question.

### Option A: explicit count field (recommended)

Store the count of individuals as a number the human controls, at the event +
species level, seeded from the detector and editable upward. Retire the box-less
detection rows and the fake frame number.

Data model:

- Keep `EventObservation.max_n` as the detector/box-derived count (what the AI and
  the human boxes say is visible).
- Add `EventObservation.count_override` (nullable int). The effective count is
  `count_override ?? max_n`. This is exactly the Wildlife Insights pattern: MaxN is
  the auto seed, the override is the human authority.
- Stop inserting box-less `Detection` rows. The hidden individual is expressed by
  bumping the count, not by creating a fake row. The fake frame number disappears
  because the thing that needed it is gone.
- Optionally add `lifeStage` / `sex` later (Camtrap-DP fields), out of scope here.

UI:

- Replace the binoculars "add observation" action with a small count stepper per
  species in the event detail and the file panel. Something like:

  ```
  Observations
    Red deer    [ - ] 5 [ + ]     (3 detected, +2 added by you)
    Wild boar   [ - ] 1 [ + ]
  ```

  The number starts at MaxN, the human can raise it for the animals they saw but
  the boxes missed, or lower it if the detector double-counted. A reset link
  returns it to the detected value. The binoculars icon can stay as the visual cue
  for "count above the detected number".
- Boxes are untouched. Visible individuals keep their `Detection` rows and boxes,
  so crops, blurring, visualizations, and per-frame review all keep working. The
  count is an additional, additive signal, not a replacement for boxes.

Outputs:

- Camtrap-DP: emit one event-level row per species per event with
  `observationLevel=event`, `count = effective count`, no bbox (conformant and
  summable), plus the existing media-level rows for the boxes. This fixes the
  current `count=1`-per-row bug.
- CSV: keep the per-detection flat grain for boxes, and surface the effective
  count so abundance is not implicit. Either add an `event_count` column on each
  row (denormalized, easy to consume) or add a second event-level summary sheet /
  file. Recommend the denormalized column for the flat CSV plus an explicit
  count on the Camtrap export.
- recognition.json: leave as is and document the limitation. The MegaDetector
  format has no count and no event level; the count lives in CSV and Camtrap-DP.
  Box geometry still exports fully.
- Dashboard: switch `sum(max_n)` to `sum(count_override ?? max_n)`. One line.

Migration sketch:

1. Alembic migration adds `count_override` (nullable int) to `event_observations`.
2. Backfill in the same migration: for each event + species that currently has
   verified box-less detection rows, set `count_override` to the existing `max_n`
   (which already includes those rows via the anchoring), then delete the box-less
   rows. This preserves every count the user already entered and removes the fake
   frames. This is a proper migration, not an ad-hoc script, so it respects the
   "no ad-hoc DB fixes" rule.
3. Frontend: swap the `N`-key add-observation flow for the count stepper; update
   the file panel grouped list to show "detected vs total".
4. Exports updated as above.

Pros: standards-aligned (Camtrap-DP `count`, DwC `individualCount`), matches the
mental model of every other tool, kills the fake frame number and the lying data,
fixes the Camtrap-DP conformance bug, single source of truth for abundance. Cons:
it is the biggest change of the three (schema, UI, three exports, a backfill).

### Option B: keep box-less detections, fix the lies

Keep the row-per-individual model but stop lying and stop losing data.

- Set `frame_number = NULL` on box-less observations instead of `best_frame_number`,
  and change MaxN so NULL-frame box-less rows are added to the per-frame max for
  their species rather than forming their own group. This removes the fake frame
  while keeping the additive count.
- Add a count column to the CSV (count of rows per event + species, denormalized).
- Fix Camtrap-DP to collapse event-level rows into one per species with the summed
  count.
- recognition.json stays as is (still drops them).

Pros: smallest change, no new UI, no schema column. Cons: still no number the user
can type (you still click `N` per individual, the part users find odd), still two
meanings in one table, still standards-awkward (abundance is implicit), and the
MaxN grouping change is fiddly to get right. It fixes correctness but not the UX.

### Option C: hybrid

Boxes stay for localization and seed the count; a single editable count field per
event + species is the human authority for abundance. Box-less detection rows are
retired.

This is option A. The "hybrid" label just emphasizes that boxes and count coexist
(boxes for where, count for how many) rather than one replacing the other. Listed
separately because the kickoff question named it, but the concrete design is option
A above. There is no meaningful third design between "explicit count" and "keep
the rows".

### Recommended path

Option A. It is the only one that gives users the count field they expect, removes
the fake frame number, makes the data honest, and produces a conformant Camtrap-DP
export, all from one concept (an editable count seeded by MaxN). Option B is the
fallback if the appetite for a schema + UI change is low right now; it fixes the
worst correctness bugs (the lie and the silent CSV gap) without the UX win, and it
does not block moving to A later.

Sequence the work as: (1) the Camtrap-DP count fix and the CSV count column can
ship independently and immediately, they are pure export bugs; (2) the schema +
stepper + backfill is the larger follow-up that delivers the UX.

---

## 6. Open decisions to confirm before building

1. Count granularity: per event + species (recommended, matches GBIF and every
   sequence tool) is the authoritative ecological count. Confirm we are happy that
   a single image is just an event of one file, so the same field covers images.
2. Large herds: do we want an "uncountable / too many to tag" state (WildTrax TMTT)
   or high-end buckets (Zooniverse 11-50, 51+), or just a plain integer for now?
   Recommend plain integer first, TMTT later if asked.
3. recognition.json: confirm it is acceptable that box-less / count-only individuals
   never appear there (it is a detection-geometry format for Timelapse). Recommend
   yes, with a one-line note in the docs.
4. Do exports follow the count everywhere, or only Camtrap-DP and CSV? Recommend
   Camtrap-DP and CSV carry it; recognition.json cannot.
5. Backfill: confirm deleting the existing box-less detection rows after copying
   their count into `count_override` is acceptable (it is reversible only via a DB
   backup, but it is the clean end state).

## Key files (quick index)

- `frontend/src/components/verify/EventDetailModal.tsx:421,799,918` and
  `FileDetailModal.tsx:289,482,589` (the `N`-key add-observation UI).
- `backend/app/api/routers/detections.py:62` and
  `backend/app/api/crud/detection.py:246` (box-less creation, the fake frame).
- `backend/app/models/detection.py`, `backend/app/models/event_observation.py`
  (no count column; `max_n` is the only count).
- `backend/app/api/crud/event_observation.py:30` (MaxN, the row-counting).
- `backend/app/api/crud/export.py:106,184,504,837` (CSV and Camtrap-DP).
- `backend/app/ml/postprocessing_outputs/recognition_json.py:101,235` (drops box-less).
- `backend/app/api/crud/statistics.py:264` (dashboard `sum(max_n)`).

## Sources

- Camtrap-DP observations table: https://camtrap-dp.tdwg.org/data/
- Darwin Core terms: https://dwc.tdwg.org/terms/
- GBIF camera-trap guide: https://docs.gbif.org/camera-trap-guide/en/
- FAIR principles: https://www.go-fair.org/fair-principles/
- Agouti annotate docs: https://docs.agouti.eu/using/annotate.html
- Camelot sighting fields: https://camelot-project.readthedocs.io/en/latest/sightingfields.html
- TRAPPER tutorial: https://trapper-project.readthedocs.io/en/latest/tutorial.html
- Wildlife Insights sequence projects: https://www.wildlifeinsights.org/get-started/identify-images/sequence-projects
- WildTrax guide: https://abbiodiversity.github.io/wildtrax-guide/
- eMammal data management: https://emammal.si.edu/node/119881
- Snapshot Serengeti FAQ + data paper: https://www.zooniverse.org/projects/zooniverse/snapshot-serengeti/about/faq and https://www.nature.com/articles/sdata201526
- TrapTagger: https://wildeyeconservation.org/traptagger/
- MegaDetector output format: https://github.com/agentmorris/MegaDetector/blob/main/megadetector-output-format.md

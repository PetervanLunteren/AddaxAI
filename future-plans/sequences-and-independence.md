# Sequences and independence as two units

Plan for a future change. Not scheduled. Captured here so we can pick it up without re-deriving the design. Written 2026-10-12 after a beta report (Simon Kravis, 2026-10-04) and a survey of how other platforms and the standards define events. None of this is built.

The short version: AddaxAI's "event" does two jobs with one unit. It is the stored thing people confirm counts on, and it is the 30-minute independence rule behind every rate. Every other platform splits those into a short, stored, timestamp-only **sequence** and a computed, species-aware **independence interval**. The split is what removes the root cause of the time-lapse problem and of the "events follow the threshold" wish, and it is the shape Camtrap DP exports expect. It costs a redesign of Counts, so it belongs to a release of its own.

## Why

A tester ran Reconyx cameras in time-lapse mode (three photos every five minutes, plus motion triggers). Events are built from all files, blanks included, split at a gap longer than the independence interval (30 min by default). A photo every five minutes never leaves a 30-minute gap, so each `100RECNX` folder (about 10,000 photos) became one event lasting days: 24,000 photos, 4 events, a lace monitor seen on two days reported as two events of eight days each. The same giant events feed the MegaDetector sequence smoother as one sequence, which can outvote a rare species.

The workaround given (2026-10-08): put time-lapse cameras in their own project with an interval below the time-lapse period. It works, at the cost of a much stricter independence than the usual 30 to 60 minutes.

On 2026-10-12 we reopened it with a sharper idea: let only "occupied" files (a detector box at a fixed floor) start and end events, with blanks attached by time span. We designed it in full, then stopped. See "The middle road we rejected" below for what it needed and why we did not build it.

## What we discussed (the questions that shaped the decision)

- **Should events follow the detection threshold?** It is the simplest mental model ("an event is what I see at my threshold"), but it means a regroup on every slider move, and a regroup can split or shrink events people have confirmed. Our events carry human work (confirmed counts, demographics, notes), so they must be stable. That rules it out for the stored unit.
- **Should a fixed, detector-agnostic floor decide occupancy?** A box of any category at 0.2 (`threshold_or_verified(0.2)`, so a verified weak box counts and a rejected one never does). Fixed, so sliders never regroup. Workable, but it needed a new carry rule (confirmations keyed on the occupied file set, or nearly every confirmed event loses its confirmation when it drops a blank edge frame), a documented hole (a file with only unverified weak boxes has no event until a box is verified), and a rollout where two definitions coexist unseen until the next reprocess. Too much machinery for one report, and still a patch on the one-unit design.
- **Can a regroup under a new rule merge events?** No, only split or shrink. Two old events are separated by a gap with no files at all. That matters for any migration: the carry can be reasoned about.
- **What happens on update?** Nothing until a regroup: end of an analysis, any reprocess, a paired-cameras or offset edit, the manual generate endpoint. And `generate_events_for_project` rebuilds every deployment of the project, so adding one folder regroups all of it. Already true today.

## Lay of the land (what the code does today)

- **One definition.** `cluster_files_into_events` in `backend/app/services/event_clustering.py` is the only code that decides which files form an event: bucket by parent folder (or the whole deployment when `paired_cameras` is on), sort by `captured_at_local`, split at a gap longer than `Project.independence_interval`. Dateless files each become a single-file event.
- **Three callers, same input.** Writing Event rows (`crud/event.py`, `_regenerate_deployment_events`), the REGROUP preview (`count_regroup_impact`) and the smoother's input (`ml/postprocessing.py`, `build_smoother_input`). Each loads all image and video files of the deployment itself; the rule is DRY, the input is not.
- **Every file is in an event.** `event_files` links each file to one event; an event with only blank files is "empty" and the Counts page hides it by default (`empty` filter: hide, show only, all). A file without an event is allowed by the schema and tolerated by every reader (`routers/detections.py`, `crud/export.py` writes blank event cells, the Labels sorts put them last).
- **What depends on events.** Counts page and MaxN (`calculate_max_n_for_event`), `Event.confirmed`, human counts and demographics (`event_observations`), notes, the dashboard Events tile and frequency ("independent events per 100 trap nights"), the Map and insights rates, `counts.csv`, Camtrap DP event-level observations, the event viewer and its filmstrip, the "by event" sorts on Labels. Trap nights do not use events.
- **What survives a regroup.** `_RegroupCarry` (`crud/event.py`): confirmed flag, human counts and demographics carry only onto a new event with the identical file set; notes carry by overlap. Labels-page verifications live on the boxes and are untouched.
- **Fixed constants** live in `backend/app/core/confidence.py`: `MD_OUTPUT_CONFIDENCE_THRESHOLD = 0.01` (what is stored), `DEFAULT_CLASSIFICATION_GATE = 0.1`, `DEFAULT_COUNTING_THRESHOLD = 0.2`, `ROLLUP_THRESHOLD` (policy, never user-facing).
- **The smoother** (`ml/smoothing_script.py`, MegaDetector's sequence-level smoothing) takes our events as its sequences. MegaDetector's own tooling builds sequences from a 60 s timestamp gap and skips images without detections.
- **The embedding model has nothing to do with events.** DINOv2 only orders events on the Labels page and feeds suggestions; the tester assumed otherwise because "Sort by event" places look-alike events next to each other.

## How other platforms do it (survey 2026-10-12, quotes and URLs verified)

| Platform | Stored short unit | Rule | Blanks in it | Independence (long) | Time-lapse | AI in grouping |
|---|---|---|---|---|---|---|
| Wildlife Insights | sequence, 60 s, fixed, "cannot be changed" | gap between consecutive images, per deployment | yes; blanks inside a mixed sequence are not labelled | analytics: 1, 2, 30 or 60 min, per species per deployment | not documented | no (65% species / 95% blank only for display) |
| eMammal / Smithsonian standard | sequence, 60 s | gap | yes | study-specific (10 min example) | not documented | no |
| Agouti | sequence, 120 s default, staff-changed, not retroactive | gap | yes, "Blank" sequences | none in the app; camtraptor `minDeltaTime` on the export | project switch: include or exclude time-lapse images | no; AI aggregates image results to the sequence |
| TRAPPER | sequence, 5 min default ("regular use is 15") | gap | yes, "classify empty pictures that are part of the sequence as empty" | none | `captureMethod` metadata only | no |
| WildTrax | series, user-set gap | gap, motion images only | yes (tagged NONE) | `wildrtrax::wt_ind_detect`, 30 min in examples, per species per location, NONE/STAFF removed | hidden by default, toggle | no (MD threshold is display only) |
| Timelapse | episode, 2 min default (1 s to 30 min) | gap, over the files in view | yes | none | none | no, but recomputed over the filtered view |
| TrapTagger | cluster, 60 s, max 15 min | gap for motion surveys; for time-lapse surveys and clusters over 15 min: starts a cluster when the site flips empty/occupied | yes, runs of empties are their own clusters | none found | survey setting "Motion / Time" | **yes** for time-lapse: `DETECTOR_THRESHOLDS = {'MDv4': 0.8, 'MDv5a': 0.2, 'MDv5b': 0.1}` in `config.py`, 7.5 min empty reset |
| Camelot | none (per photo) | n/a | n/a | report-time, 20 min default, fixed window from the first sighting, per species per session | not documented | no |
| camtrapR / camtraptor | none (records) | n/a | blanks never become records | `minDeltaTime`, per species per station, "lastRecord" (rolling) or "lastIndependentRecord" (fixed window), default 0 | not documented | no |
| Snapshot Safari / Zooniverse | capture event = one trigger (camera-made) | 1 min camera delay | yes, "nothing here" | 30 min filter in analysis datasets | not documented | pre-sort only |
| Zamba | none (per file / per box) | n/a | n/a | none | not documented | n/a |
| MegaDetector tooling | `seq_id` via `create_sequences`, 60 s | gap per location | yes; smoothing skips them | n/a | n/a | no (smoothing uses a fixed 0.15 box floor, but does not group) |

Sources: Wildlife Insights glossary and help (wildlifeinsights.org/node/283, /node/24563, /get-started/analyze/generate); Forrester et al. 2016 "An Open Standard for Camera Trap Data"; docs.agouti.eu/using/project_settings.html; trapper-project.readthedocs.io tutorial and ai-pipeline; cms.wildtrax.ca guide and abbiodiversity.github.io/wildrtrax; timelapse.ucalgary.ca reference guide part 11 and `Constant.cs`; TrapTagger `app/functions/globals.py` and `config.py` (github.com/WildEyeConservation/TrapTagger); camelot-project.readthedocs.io reports and `sighting_independence.clj`; rdrr.io/cran/camtrapR/man/recordTable.html; inbo.github.io/camtraptor; Swanson et al. 2015 and Pardo et al. 2021 for Snapshot; github.com/agentmorris/MegaDetector `cct_json_utils.py` and `classification_postprocessing.py`.

The pattern, in one sentence: the stored unit is a short, timestamp-only burst over all photos, cut before any model or person has looked at them, so a threshold can never move it; the 30-minute visit is a species-aware calculation done afterwards over species records only, where blanks cannot enter and nothing is stored. TrapTagger is the one exception that lets detections decide a boundary, and only for time-lapse surveys, with a hard-coded floor. Nobody re-cuts stored bursts after the fact; Agouti refuses to, Wildlife Insights cannot.

## What the standards say

- **Camtrap DP** has no events table and no definition of an event. An event exists only as `eventID`, `eventStart`, `eventEnd` on observation rows. `observationLevel` is `media` (per file, "especially useful for machine learning", need not be mutually exclusive) or `event` ("consider an event (comprising a collection of media files)", "should be mutually exclusive, so that their `count` can be summed"). `media.captureMethod` is `activityDetection` or `timeLapse`, with no semantics. In the official example (an Agouti export) the three time-lapse frames sit outside every event and have only a media-level blank observation. Blank event-level observations exist.
- **GBIF camera trap guide** glossary: *sequence* "Series of media files taken in rapid succession but separated by a time interval less than the set independence interval"; *independence interval* "Minimum duration between consecutive triggers to be considered belonging to separate sequences. This duration (e.g. 120 seconds) can be defined in a data management system to automatically group media into sequences"; *event* "animal activity recorded through one or more triggers and forming a sequence, but other definitions might be used when analysing data". The guide recommends publishing event-level animal observations and filtering blanks, humans and vehicles out of the Occurrence core.
- **Darwin Core** maps the sequence to `eventID`, the deployment to `parentEventID`, `eventStart/eventEnd` to `eventDate` as an interval; `camtrapdp::events()` derives events from `observationLevel == "event"` rows.

Note the vocabulary: in the standards, "independence interval" is the *sequence* gap (120 s), and the 30-minute species-level rule is an analysis convention from camtrapR. AddaxAI uses "independence interval" for the 30-minute rule. The plan below keeps our user-facing word for the long rule and introduces "sequence" for the short one; the docs will need one sentence on the difference.

## Target model: two units

1. **Sequence** (stored). A burst of files from one camera folder (or one paired deployment) cut purely by timestamp gap, default 120 s (Agouti, GBIF example), never user-visible as a setting in the first version. Built once at ingest, blanks included, never dependent on any model, threshold or label, and therefore never rebuilt except after a capture-time correction (Adjust dates, camera offsets, relink does not change times). This is what the Counts page shows, what MaxN is computed on, what `confirmed`, human counts, demographics and notes attach to, what the smoother gets as a sequence, and what Camtrap DP exports as an event. A 5-minute time-lapse cadence can never bridge a 120 s gap.

2. **Visit** (computed, the user-facing name for the 30-minute rule; keep the setting name `independence_interval`). "Sequences of the same species at the same camera less than N minutes apart are one visit." Computed on demand over species records that pass the current counting threshold, per species, with the rolling rule (`deltaTimeComparedTo = "lastRecord"` in camtrapR terms, the same gap rule we use today). Never stored. Changing the threshold, verifying a box, relabelling, or changing the interval changes the next chart and nothing else. Blanks cannot enter it by construction. This is what "independent events per 100 trap nights", the trend, the map rates and the activity insights read.

Why this removes the root cause: the stored unit is dumb, so it is stable, and the smart unit is cheap to recompute, so it may follow every filter. The "events follow the threshold" wish comes true on the computed layer, where it costs nothing.

## Design sketch

**Data model.** Rename nothing in a hurry; the `events` table becomes the sequence table in meaning. Add `Project.sequence_gap_seconds` (default 120, not in the UI at first; a constant until someone asks). Keep `event_files`, `event_observations`, `confirmed`, `notes`. Add `File.capture_method` (`activityDetection` / `timeLapse` / null) filled from EXIF where a camera writes it (Reconyx `TriggerMode`; needs an exiftool pass on images, today only videos get one) so Camtrap DP can carry `captureMethod` and the Counts page can show the time-lapse badge; not used for grouping.

**Grouping.** `cluster_files_into_events` keeps its shape with the gap parameter now the sequence gap. The paired-cameras bucket rule stays. Dateless files stay single-file sequences. The smoother keeps getting these clusters (MegaDetector's smoothing was designed for bursts, not for 30-minute visits).

**Visits.** One function, `visits(db, project, *, independence_interval, threshold, site_ids, date_from, date_to) -> rows (species, deployment, visit_start, visit_end, sequence_ids, count)`, in `crud/statistics.py` or a new `crud/visits.py`. Count per visit = max over its sequences of the sequence's effective count (confirmed count or AI MaxN), which keeps the "do not add up two views of one animal" rule. Every rate that says "independent events" reads this and nothing else: `get_species_distribution` (frequency mode), the trend, the map, the activity analyses, `get_filter_options`, the dashboard tile (relabel it "Visits"), and `counts.csv` gains `visit_id` so a person can regroup in Excel. The Camtrap DP export is unchanged in shape and becomes more honest: event = sequence.

**Counts page.** The unit of confirmation becomes the sequence. To keep the work bearable: group the cards by visit (a header per visit, sequences inside, like the event dividers on Labels today), "Same as last" and keyboard confirm already exist, and a "confirm visit" action that applies one count to every sequence of a visit is the one new control. An animal that lingers for twenty minutes is then one header with a few cards, not one card as today.

**Event viewer.** Opens a sequence; the filmstrip shows the sequence. A "previous / next sequence in this visit" pair replaces nothing and adds continuity.

**Trap nights.** Untouched; they never used events.

**Migration.** Existing events are visits by today's rule, so they split into sequences by the 120 s gap. Carry: for every old event, each new sequence inside it inherits `notes`; `confirmed` and human counts carry onto a new sequence only when the old event contained exactly one sequence (then they describe the same files); otherwise the visit-level count becomes a `visit` computation and the sequences get the AI MaxN back, unconfirmed. Report the counts of "carried" and "reset" in the migration log and in a one-time notice. Test on the real backups under `~/AddaxAI/backups/` with `backend/scripts/check_migration_on_backups.py` before release.

**Docs.** `docs/docs/understanding/detections-events-observations.mdx` is rewritten around three words: detection, sequence, visit (observation stays). One paragraph on time-lapse cameras. `trap-nights.md` unchanged. DEVELOPERS.md "Observation cohorts" and "Paired cameras" updated; the "Datetime conventions" rule about `async def` applies to the visits endpoint.

## Effort and when to do it

A release of its own, not a feedback-round fix. Rough shape: backend model and grouping 2 days, visits computation and the seven consumers 3 days, Counts page regrouping and confirm-visit 3 days, migration with backup tests 2 days, docs 1 day, bug hunt 1 day. About two and a half weeks of focused work, plus the beta cycle. Do it when the Counts page is next redesigned, or when a second user hits the time-lapse problem or asks for Camtrap DP exports that match Agouti's. Until then the workaround holds.

## The middle road we rejected (so nobody rebuilds it)

Keep one unit, but let only occupied files start and end events: occupancy = `threshold_or_verified(EVENT_OCCUPANCY_FLOOR)` with a fixed floor of 0.2 (TrapTagger's MDv5a value, our own "mostly false positives below" line), any detector category, any video frame; blanks attached by time span; files outside every span eventless. It fixes time-lapse without touching the slider, and it is the right answer if the two-unit model is never built. We stopped because, to be safe, it needs: a changed carry rule (key confirmations on the occupied file set, or almost every confirmed event loses its confirmation when it drops a blank edge frame); a documented hole (a file with only unverified weak boxes has no event until one is verified, which hurts the user who sets the threshold to 0.01); and a rollout where old and new definitions coexist unseen until each project's next reprocess. It is a patch on the one-unit design, and the one-unit design is the problem.

## Open questions

- Sequence gap: 120 s (Agouti, GBIF example) or 60 s (Wildlife Insights, eMammal, MegaDetector)? Either stops time-lapse chaining; 120 s keeps longer bursts together. Decide with Simon's data set and one or two others.
- Should a person or vehicle sequence count toward a visit of its own kind in the rates? Camtrap DP treats them as observation types; the dashboard shows wildlife only.
- Does the Counts page need a visit-level count at all (confirm once per visit), or is per-sequence confirmation with "confirm visit" enough? Wildlife Insights and Agouti confirm per sequence and never show a visit.
- `independence_interval` default: keep 30 min; camtrapR users often use 60. Leave as is.
- Name: the user-facing word for the long unit. "Visit" is proposed because "event" is taken by the standards for the sequence and "independent event" reads as a statistic.

## Key files

| File | Role today |
|---|---|
| `backend/app/services/event_clustering.py` | The one grouping rule |
| `backend/app/api/crud/event.py` | Event rows, regroup carry (`_RegroupCarry`), preview, events sort and filters |
| `backend/app/api/crud/event_observation.py` | MaxN per event, cohorts, human counts |
| `backend/app/ml/postprocessing.py` | Smoother input from the same clusters |
| `backend/app/api/crud/statistics.py` | Frequency and abundance, trend, map rates ("independent events") |
| `backend/app/api/crud/export.py` | `counts.csv`, Camtrap DP event-level observations |
| `backend/app/core/confidence.py` | The fixed policy constants |
| `frontend/src/components/verify/VerifyView.tsx` | The Counts page |
| `docs/docs/understanding/detections-events-observations.mdx` | The user-facing definitions |

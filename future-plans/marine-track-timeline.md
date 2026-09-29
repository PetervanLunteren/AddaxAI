# Track timeline and track entity (option B of the marine design)

Plan for a future change. Not scheduled. This is option B from `MARINE_VIDEOS_DESIGN.md` (2026-09-05), written down so it can be built on top of option A without re-deriving it. Everything here is additive to A: no table, column or rule of A is undone.

## Why it may be wanted

Option A reviews tracks as cards in the crop grid. That is SharkTrack's proven loop and it is right for shark BRUVs, where a drop holds tens of tracks. Two situations push past it:

1. **Dense scenes.** A reef video with 300 tracks is 300 cards. The person cannot see when each animal was in the clip, which tracks overlap in time, or where in the video the peaks are. A timeline shows that at a glance and lets them review by scrubbing rather than by paging.
2. **Per individual metrics.** Time on screen per track, first arrival per track, behaviour per track, and later lengths from stereo pairs all hang off the track as a thing with its own attributes. In A the track has no attributes of its own except its span and its frame.

## What A already leaves room for

- `tracks` table with `id, file_id, track_key, start_frame, end_frame, frame_count, max_confidence, representative_frame_number, frame_path`.
- `Detection.track_id`.
- `FileWithDetections.tracks` on the wire.
- `VideoPlayer` colours boxes by species and outlines the opened track.
- The visibility rule admits a track's representative box.
- `frame_time(file, frame_number)` derives any frame's wall clock time.

Nothing below needs a change to those; it adds beside them.

## The change

### 1. The track carries its own label and verdict

Add to `tracks`: `label`, `label_taxonomy_id`, `verified`, `verified_at_utc`, `classification_method`, `rejected` (or reuse the non label label as detections do, decided at build time to match `is_a_real_detection`).

The rule: **the track is the label's home for tracked boxes; its detections mirror it.** Every write to a track (`relabel_track`, `reject_track`, `verify_track`) writes the track row and then the same fields onto every detection with that `track_id`, in one transaction, through one helper in `crud/track.py`. The reverse write (a detection edited directly) is refused for tracked boxes on the API, except drawn boxes, which have no track.

Why two homes and not one: the per frame boxes must keep the label for every existing query (MaxN groups by `Detection.label`, exports read detections, the player draws per frame). Moving the label off detections would touch every consumer. Mirroring keeps them all and adds a parity test: `tests/api/test_track_label_parity.py` asserts, after every write path, that a track's label equals the label on each of its boxes. That test is the guard that makes two homes safe.

The reprocess path skips verified detections already; it must also skip boxes whose track is verified, which follows from the mirror.

### 2. A tracks API

| Endpoint | Purpose |
|---|---|
| `GET /api/files/{file_id}/tracks` | tracks of a video with label, span, frame count, representative frame, first and last time (derived) |
| `GET /api/projects/{project_id}/tracks` | paged list with the Labels page filters (species, verified, confidence, deployment, date), sorted by file then start frame; the timeline's data source and a "track list" view if ever wanted |
| `PATCH /api/tracks/{id}` | label, verified, rejected; cascades to boxes through the helper |
| `POST /api/tracks/bulk-relabel`, `bulk-reject`, `bulk-verify` | the grid's bulk actions, by track id |

The grid keeps working through detections (option A). These endpoints are for the timeline and for anything that wants to think in tracks.

### 3. The track card

`CropCard` gains a track variant: the same crop, plus a thin strip under it showing the track's span as a bar over the clip's length, the duration in seconds, and the frame count. The strip is the same component the timeline uses, at card size. A card of a video without tracks looks as today.

### 4. The timeline under the player

In `FileDetailModal` and `EventDetailModal`, when the file has tracks, a timeline sits under `VideoPlayer`:

- One horizontal bar per track, in the species colour, from `start_frame` to `end_frame` over the clip's duration. Rejected tracks are drawn hollow, verified ones solid, unverified ones hatched.
- A playhead synced with the player. Click a bar to seek to the track's representative frame and select the track (its boxes get the thick outline the player already draws).
- Per species MaxN markers: a small triangle at the MaxN frame of each species present in the event, read from `EventObservation.max_n_frame_number` (increment 2 of A). Click to seek.
- Rows are grouped by species and sorted by start frame. A dense video collapses rows beyond a height cap into a scrollable list.
- Keyboard: `,` and `.` step frames (the player has these), `[` and `]` go to the previous and next track start, `Home` and `End` to the selected track's start and end (the VIAME keys), and the existing label keys act on the selected track.

The timeline is one component, `TrackTimeline.tsx`, taking `tracks`, `frameRate`, `durationSeconds`, `currentFrame`, `selectedTrackId`, `maxNMarkers` and callbacks. No new state store: the modal already holds the current frame and the player.

### 5. Times on screen

With `frame_time` from A: each track shows its first and last time in camera local time, the event panel shows first arrival per species (increment 2 of A) and time on screen per species (sum of track spans, overlapping spans counted once), and the export gains `time_on_screen_seconds` per observation row if anyone asks.

### 6. Split and merge (later, and only if asked)

Merge: select two tracks in the timeline, press `M`, the helper rewrites `track_id` on the second's boxes to the first, recomputes span and count, deletes the empty track row, and the label of the first wins. Refused when the two tracks share a frame (that is two animals). Split: at the playhead, `S` moves every box after the current frame to a new track with the same label. Both are single transactions through `crud/track.py` and both invalidate the event's observations so MaxN is rebuilt. This is the VIAME toolset; it matters for per individual metrics, not for MaxN, which is why it is last.

## Order of work

1. Track label and verdict columns, the write helper, the mirror, the parity test.
2. Tracks API.
3. `TrackTimeline.tsx` in the two modals, with seek and select.
4. Track card strip.
5. Times on screen and first arrival display.
6. Split and merge, if and when asked.

Each step ships on its own. Steps 1 and 2 are backend only and change nothing visible.

## Costs to state

- Two homes for one label, guarded by a test. Any new write path to detections must go through the helper or the test catches it.
- A timeline is a real UI component with keyboard handling and a collapsed mode; budget it like the event filmstrip, not like a caption.
- The tracks list endpoint over a project is a new paged query; index `tracks(file_id, start_frame)` covers it.

## What stays out even here

Stereo lengths and calibration, MeanCount, EventMeasure and GlobalArchive formats, a species classifier for fish, and a separate review page. Those are their own designs.

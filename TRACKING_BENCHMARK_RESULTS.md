# Tracking count benchmark results

Run overnight on 2026-09-14. Plan and audit: `camera-trap-tracking-count-benchmark.md` and the session notes. Data, caches, code and every CSV are on the external drive under `addaxai-benchmarks/tracking-count/`. Read the numbers before the prose: the prose is templated from the tables.

## What ran

| stage | status |
|---|---|
| setup | done 2026-09-13T22:39 |
| download | done 2026-09-13T22:54 |
| manifests | done 2026-09-13T22:54 |
| detect_panaf | done 2026-09-14T04:14 |
| detect_validate | done 2026-09-14T04:16 |
| detect_iwildcam | done 2026-09-14T05:51 |
| replay_panaf | done 2026-09-14T05:55 |
| replay_stills | done 2026-09-14T05:56 |
| analyse | done 2026-09-14T05:56 |
| report | done 2026-09-14T05:56 |
| detect_safari | done 2026-09-14T07:15 |
| replay_safari | done 2026-09-14T07:16 |
| tune_threshold | done 2026-09-14T09:58 |
| tune_video | done 2026-09-14T10:14 |
| tune_stills | done 2026-09-14T10:16 |
| detect_spruce | done 2026-09-14T11:48 |
| replay_spruce | done 2026-09-14T11:57 |
| report_3 | done 2026-09-14T11:57 |

## Setup

- Repo commit: `c6f6b2eb` on branch `marine-videos`, plus the uncommitted `--raw_detections_json` flag on `tracking_script.py`.
- Detector: MD5A (`md_v5a.0.0.pt`), the app's default, through the megadetector package in `env-addaxai-base`, on Apple MPS (M1 Pro, 16 GB).
- Tracker under test: ultralytics 8.3.177 BoT-SORT exactly as `tracking_script.py` builds it (match 0.97, two-second buffer, track start 0.2, boxes down to 0.01, no ReID). Offline replays skip its camera-motion compensation; the validation section measures that.
- Other trackers: Roboflow `trackers` 2.6.0 ByteTrack and OC-SORT, given the same floors and the same two-second budget, run once counting calls and once with real timestamps; the `1 frame` variants set `minimum_consecutive_frames=1`.
- Sampling: `round(native_fps / fps)`-th frame, the app's rule. PanAf500 master at 12 fps (stride 2 on 24 fps clips); replays at 0.5, 1, 2, 3, 4, 6 and 12 fps use exactly the frames the app would sample. SA-FARI master at 6 fps.
- Counts: track count = tracks whose best box is an animal; MaxN = most animal boxes at or above 0.2 in one sampled frame, the app's own count.
- Detection time, panaf: 343 min for 500 videos at 12.0 fps.
- Detection time, iwildcam: 95 min for 13190 images.
- Detection time, safari: 79 min for 200 videos at 6.0 fps.

## Datasets

**PanAf500**: 500 clips, 15 s, 24 fps, 720 px wide, chimpanzees and gorillas, human `ape_id` per individual. Apes per clip: 1: 291, 2: 116, 3: 53, 4: 23, 5: 10, 6: 3, 7: 1, 8: 2, 9: 1.

**iWildCam 2022**: 1780 counted sequences, 1530 kept (dropped: count_9_cap 182, empty 68). Human count of distinct individuals across the sequence. Counts: 1: 340, 2: 307, 3: 351, 4: 259, 5: 156, 6: 71, 7: 31, 8: 15. Photos per sequence 3 to 10; median gap between photos: median 1 s, max gap up to 60 s.

**SA-FARI test**: 833 clips available, 200 run tonight (83 species, every clip with three or more animals, then round-robin by species). Count = masklets per clip (one per individual, exits and re-entries kept). Animals per clip: 1: 127, 2: 19, 3: 25, 4: 16, 5: 6, 6: 3, 8: 4.

## Findings in one place

- PanAf500, track count: exact accuracy is highest at 1.0 fps (61.2%). 0.5 fps 58.0%, 1.0 fps 61.2%, 2.0 fps 59.8%, 3.0 fps 57.6%, 4.0 fps 56.4%, 6.0 fps 50.6%, 12.0 fps 43.0%.
- At 1 fps: track count 61.2% exact, MAE 0.486, signed +0.02 (over 21%, under 18%); MaxN 63.0% exact, signed +0.09.
- At 2 fps: track count 59.8% exact, MAE 0.53, signed +0.28 (over 30%, under 10%); MaxN 59.0% exact, signed +0.18.
- At 3 fps: track count 57.6% exact, MAE 0.56, signed +0.40 (over 35%, under 7%); MaxN 57.2% exact, signed +0.24.
- Human boxes through the same tracker at 2 fps: 86.6% exact against 59.8% with MD5A boxes.
- Best tracker condition at 2 fps: OC-SORT at 60.2% exact (MAE 0.542).
- iWildCam sequences: MaxN 64.3% exact (signed +0.16), BoT-SORT track count 59.7% (signed +0.20), sum over photos 0.7%.
- Best still-image condition: BoT-SORT, window 1s, max of segments at 64.8% exact (MAE 0.488).
- Replay fidelity: 19/20 validation clips give the same track count offline as the production run at 2 fps.
- MD5A versus MD1000-SPRUCE, track count exact: 1 fps 61.2% vs 50.6%, 2 fps 59.8% vs 49.8%, 3 fps 57.6% vs 47.4%.
- MD5A versus MD1000-SPRUCE, MaxN exact: 1 fps 63.0% vs 51.8%, 2 fps 59.0% vs 46.8%, 3 fps 57.2% vs 45.4%.
- panaf_heldout at 1 fps: app 55.0% exact, best frozen config `h0.2_l0.01_m0.97_b2_sharktrack` (app+sharktrack) 64.0%, MAE 0.55 to 0.48.
- panaf_heldout at 2 fps: app 53.0% exact, best frozen config `h0.2_l0.01_m0.97_b2_sharktrack` (app+sharktrack) 68.0%, MAE 0.64 to 0.43.
- panaf_heldout at 3 fps: app 49.0% exact, best frozen config `h0.2_l0.1_m0.97_b2.0_sharktrack` (best_at_2fps) 71.0%, MAE 0.67 to 0.4.
- safari at 1 fps: app 60.9% exact, best frozen config `h0.2_l0.01_m0.97_b2_sharktrack` (app+sharktrack) 62.0%, MAE 0.739 to 0.685.
- safari at 2 fps: app 60.3% exact, best frozen config `h0.3_l0.1_m0.97_b4.0_sharktrack` (best_at_3fps) 65.8%, MAE 0.707 to 0.582.
- safari at 3 fps: app 63.0% exact, best frozen config `h0.2_l0.01_m0.97_b2_sharktrack` (app+sharktrack) 69.6%, MAE 0.658 to 0.538.
- MaxN threshold, panaf at 2 fps: 0.2 gives 59.0%, best is 0.4 at 69.8% (signed -0.12).
- MaxN threshold, safari at 2 fps: 0.2 gives 65.2%, best is 0.6 at 71.2% (signed -0.34).
- MaxN threshold, iwildcam: 0.2 gives 64.3%, best is 0.6 at 71.9% (signed -0.23).
- Photo sequences, held-out locations: app 57.8%, tuned 59.9%, MaxN 63.9%.

## Does the offline replay match a real run

Production run at 2 fps (camera-motion compensation on) against the offline replay of the 12 fps master subsampled to the same frames (compensation off), 20 clips:

- same track count: 19/20 clips, mean absolute difference 0.10
- same MaxN: 17/20 clips
- production tracked boxes with a raw box at the same place (IoU 0.9): 393/602 over 600 frames (the detector is deterministic on the same frame; a miss is a Kalman offset, not a different detection)

## Experiment 1: video sampling rate

### PanAf500: the app's tracker and MaxN per sampling rate

| fps | condition | n | exact | mae | signed | over | under | within_1 | catastrophic |
|---|---|---|---|---|---|---|---|---|---|
| 0.5 | BoT-SORT (the app) | 500 | 58.0% | 0.534 | -0.186 | 14.8% | 27.2% | 91.6% | 1.8% |
| 0.5 | MaxN (what the app shows) | 500 | 66.2% | 0.424 | -0.016 | 17.6% | 16.2% | 93.2% | 1.4% |
| 1.0 | BoT-SORT (the app) | 500 | 61.2% | 0.486 | 0.022 | 20.8% | 18.0% | 92.0% | 1.4% |
| 1.0 | MaxN (what the app shows) | 500 | 63.0% | 0.466 | 0.086 | 22.8% | 14.2% | 92.4% | 1.4% |
| 2.0 | BoT-SORT (the app) | 500 | 59.8% | 0.53 | 0.278 | 30.2% | 10.0% | 90.4% | 2.8% |
| 2.0 | MaxN (what the app shows) | 500 | 59.0% | 0.5 | 0.176 | 28.2% | 12.8% | 92.8% | 1.4% |
| 3.0 | BoT-SORT (the app) | 500 | 57.6% | 0.56 | 0.396 | 35.4% | 7.0% | 89.0% | 2.6% |
| 3.0 | MaxN (what the app shows) | 500 | 57.2% | 0.528 | 0.236 | 31.4% | 11.4% | 92.2% | 1.8% |
| 4.0 | BoT-SORT (the app) | 500 | 56.4% | 0.59 | 0.49 | 39.0% | 4.6% | 88.2% | 2.6% |
| 4.0 | MaxN (what the app shows) | 500 | 55.2% | 0.554 | 0.298 | 34.6% | 10.2% | 91.8% | 2.0% |
| 6.0 | BoT-SORT (the app) | 500 | 50.6% | 0.708 | 0.624 | 45.2% | 4.2% | 86.0% | 5.0% |
| 6.0 | MaxN (what the app shows) | 500 | 53.8% | 0.584 | 0.344 | 37.0% | 9.2% | 90.6% | 2.4% |
| 12.0 | BoT-SORT (the app) | 500 | 43.0% | 0.992 | 0.952 | 55.0% | 2.0% | 75.6% | 10.6% |
| 12.0 | MaxN (what the app shows) | 500 | 48.0% | 0.666 | 0.458 | 43.8% | 8.2% | 88.8% | 3.0% |
| 12.0 | BoT-SORT production run (with camera motion) | 500 | 42.8% | 0.996 | 0.956 | 55.2% | 2.0% | 75.6% | 10.8% |

Detector cost per rate (frames run over the whole set, minutes at the master run's pace, decode included):

| fps | frames | detector_minutes |
|---|---|---|
| 0.5 | 4000 | 15.2 |
| 1.0 | 7500 | 28.6 |
| 2.0 | 15000 | 57.2 |
| 3.0 | 22497 | 85.8 |
| 4.0 | 29997 | 114.5 |
| 6.0 | 44994 | 171.6 |
| 12.0 | 89984 | 343.4 |

### Detector or tracker: the same tracker on human boxes

| fps | condition | n | exact | mae | signed | over | under | within_1 | catastrophic |
|---|---|---|---|---|---|---|---|---|---|
| 0.5 | BoT-SORT (the app) | 500 | 58.0% | 0.534 | -0.186 | 14.8% | 27.2% | 91.6% | 1.8% |
| 0.5 | MaxN on human boxes | 500 | 80.4% | 0.244 | -0.244 | 0.0% | 19.6% | 96.6% | 1.0% |
| 0.5 | BoT-SORT on human boxes | 500 | 69.0% | 0.402 | -0.306 | 4.4% | 26.6% | 93.4% | 1.8% |
| 1.0 | BoT-SORT (the app) | 500 | 61.2% | 0.486 | 0.022 | 20.8% | 18.0% | 92.0% | 1.4% |
| 1.0 | MaxN on human boxes | 500 | 82.6% | 0.214 | -0.214 | 0.0% | 17.4% | 97.0% | 0.6% |
| 1.0 | BoT-SORT on human boxes | 500 | 78.6% | 0.256 | -0.136 | 5.8% | 15.6% | 96.2% | 0.4% |
| 2.0 | BoT-SORT (the app) | 500 | 59.8% | 0.53 | 0.278 | 30.2% | 10.0% | 90.4% | 2.8% |
| 2.0 | MaxN on human boxes | 500 | 83.6% | 0.196 | -0.196 | 0.0% | 16.4% | 97.6% | 0.6% |
| 2.0 | BoT-SORT on human boxes | 500 | 86.6% | 0.154 | 0.034 | 7.8% | 5.6% | 98.6% | 0.6% |
| 3.0 | BoT-SORT (the app) | 500 | 57.6% | 0.56 | 0.396 | 35.4% | 7.0% | 89.0% | 2.6% |
| 3.0 | MaxN on human boxes | 500 | 84.2% | 0.192 | -0.192 | 0.0% | 15.8% | 97.4% | 0.6% |
| 3.0 | BoT-SORT on human boxes | 500 | 89.6% | 0.124 | 0.064 | 7.4% | 3.0% | 98.6% | 0.4% |
| 4.0 | BoT-SORT (the app) | 500 | 56.4% | 0.59 | 0.49 | 39.0% | 4.6% | 88.2% | 2.6% |
| 4.0 | MaxN on human boxes | 500 | 83.6% | 0.196 | -0.196 | 0.0% | 16.4% | 97.6% | 0.6% |
| 4.0 | BoT-SORT on human boxes | 500 | 90.4% | 0.118 | 0.07 | 7.2% | 2.4% | 98.4% | 0.4% |
| 6.0 | BoT-SORT (the app) | 500 | 50.6% | 0.708 | 0.624 | 45.2% | 4.2% | 86.0% | 5.0% |
| 6.0 | MaxN on human boxes | 500 | 84.2% | 0.19 | -0.19 | 0.0% | 15.8% | 97.6% | 0.6% |
| 6.0 | BoT-SORT on human boxes | 500 | 90.2% | 0.126 | 0.086 | 7.8% | 2.0% | 98.0% | 0.6% |
| 12.0 | BoT-SORT (the app) | 500 | 43.0% | 0.992 | 0.952 | 55.0% | 2.0% | 75.6% | 10.6% |
| 12.0 | MaxN on human boxes | 500 | 84.2% | 0.19 | -0.19 | 0.0% | 15.8% | 97.6% | 0.6% |
| 12.0 | BoT-SORT on human boxes | 500 | 90.4% | 0.132 | 0.096 | 7.8% | 1.8% | 97.4% | 0.6% |

### Other trackers on the same boxes

| fps | condition | n | exact | mae | signed | over | under | within_1 | catastrophic |
|---|---|---|---|---|---|---|---|---|---|
| 1.0 | BoT-SORT (the app) | 500 | 61.2% | 0.486 | 0.022 | 20.8% | 18.0% | 92.0% | 1.4% |
| 1.0 | ByteTrack | 500 | 51.6% | 0.652 | 0.392 | 36.8% | 11.6% | 88.4% | 3.6% |
| 1.0 | ByteTrack, 1 frame | 500 | 50.0% | 0.696 | 0.464 | 39.6% | 10.4% | 86.8% | 4.4% |
| 1.0 | ByteTrack, real time | 500 | 51.6% | 0.656 | 0.4 | 37.0% | 11.4% | 88.2% | 3.8% |
| 1.0 | ByteTrack, real time, 1 frame | 500 | 50.2% | 0.698 | 0.47 | 39.6% | 10.2% | 86.6% | 4.6% |
| 1.0 | OC-SORT | 500 | 53.6% | 0.622 | -0.186 | 17.0% | 29.4% | 87.2% | 2.8% |
| 1.0 | OC-SORT, 1 frame | 500 | 52.4% | 0.66 | 0.216 | 29.0% | 18.6% | 87.2% | 3.8% |
| 1.0 | OC-SORT, real time | 500 | 53.8% | 0.616 | -0.188 | 16.6% | 29.6% | 87.6% | 2.6% |
| 1.0 | OC-SORT, real time, 1 frame | 500 | 51.6% | 0.682 | 0.226 | 29.0% | 19.4% | 87.0% | 5.0% |
| 2.0 | BoT-SORT (the app) | 500 | 59.8% | 0.53 | 0.278 | 30.2% | 10.0% | 90.4% | 2.8% |
| 2.0 | ByteTrack | 500 | 44.8% | 0.828 | 0.716 | 50.2% | 5.0% | 80.4% | 5.4% |
| 2.0 | ByteTrack, 1 frame | 500 | 44.2% | 0.856 | 0.752 | 51.0% | 4.8% | 79.4% | 6.2% |
| 2.0 | ByteTrack, real time | 500 | 44.6% | 0.834 | 0.722 | 50.4% | 5.0% | 80.4% | 5.8% |
| 2.0 | ByteTrack, real time, 1 frame | 500 | 44.0% | 0.862 | 0.758 | 51.2% | 4.8% | 79.4% | 6.6% |
| 2.0 | OC-SORT | 500 | 60.2% | 0.542 | 0.202 | 26.2% | 13.6% | 88.0% | 2.2% |
| 2.0 | OC-SORT, 1 frame | 500 | 50.2% | 0.842 | 0.662 | 43.0% | 6.8% | 77.8% | 8.6% |
| 2.0 | OC-SORT, real time | 500 | 59.8% | 0.562 | 0.226 | 26.4% | 13.8% | 87.4% | 3.2% |
| 2.0 | OC-SORT, real time, 1 frame | 500 | 49.2% | 0.876 | 0.696 | 43.8% | 7.0% | 77.2% | 9.4% |
| 3.0 | BoT-SORT (the app) | 500 | 57.6% | 0.56 | 0.396 | 35.4% | 7.0% | 89.0% | 2.6% |
| 3.0 | ByteTrack | 500 | 42.6% | 0.976 | 0.9 | 53.6% | 3.8% | 75.6% | 10.4% |
| 3.0 | ByteTrack, 1 frame | 500 | 42.0% | 1.01 | 0.942 | 54.6% | 3.4% | 74.4% | 11.6% |
| 3.0 | ByteTrack, real time | 500 | 42.6% | 0.98 | 0.908 | 53.8% | 3.6% | 75.6% | 10.4% |
| 3.0 | ByteTrack, real time, 1 frame | 500 | 42.0% | 1.014 | 0.95 | 54.8% | 3.2% | 74.4% | 11.6% |
| 3.0 | OC-SORT | 500 | 59.0% | 0.588 | 0.384 | 32.0% | 9.0% | 87.2% | 4.0% |
| 3.0 | OC-SORT, 1 frame | 500 | 44.6% | 0.978 | 0.882 | 51.0% | 4.4% | 74.4% | 9.2% |
| 3.0 | OC-SORT, real time | 500 | 58.6% | 0.6 | 0.396 | 32.4% | 9.0% | 86.6% | 4.0% |
| 3.0 | OC-SORT, real time, 1 frame | 500 | 44.0% | 1.004 | 0.904 | 51.4% | 4.6% | 73.4% | 10.2% |

### By true group size, at 2 fps

| size_bin | condition | n | exact | mae | signed | over | under | within_1 | catastrophic |
|---|---|---|---|---|---|---|---|---|---|
| 1 | BoT-SORT (the app) | 291 | 69.8% | 0.378 | 0.316 | 27.2% | 3.1% | 94.2% | 1.4% |
| 1 | MaxN (what the app shows) | 291 | 66.3% | 0.392 | 0.357 | 32.0% | 1.7% | 94.8% | 0.3% |
| 1 | BoT-SORT on human boxes | 291 | 94.5% | 0.058 | 0.052 | 5.1% | 0.3% | 99.7% | 0.0% |
| 2 | BoT-SORT (the app) | 116 | 51.7% | 0.629 | 0.319 | 34.5% | 13.8% | 90.5% | 4.3% |
| 2 | MaxN (what the app shows) | 116 | 60.3% | 0.448 | 0.155 | 25.0% | 14.7% | 97.4% | 1.7% |
| 2 | BoT-SORT on human boxes | 116 | 85.3% | 0.155 | 0.017 | 7.8% | 6.9% | 99.1% | 0.0% |
| 3 | BoT-SORT (the app) | 53 | 41.5% | 0.811 | 0.17 | 35.9% | 22.6% | 81.1% | 3.8% |
| 3 | MaxN (what the app shows) | 53 | 39.6% | 0.717 | 0.038 | 26.4% | 34.0% | 88.7% | 0.0% |
| 3 | BoT-SORT on human boxes | 53 | 71.7% | 0.358 | 0.094 | 15.1% | 13.2% | 94.3% | 1.9% |
| 4 | BoT-SORT (the app) | 23 | 34.8% | 0.826 | 0.217 | 34.8% | 30.4% | 87.0% | 4.3% |
| 4 | MaxN (what the app shows) | 23 | 39.1% | 0.696 | -0.522 | 8.7% | 52.2% | 91.3% | 0.0% |
| 4 | BoT-SORT on human boxes | 23 | 52.2% | 0.478 | -0.13 | 17.4% | 30.4% | 100.0% | 0.0% |
| 5+ | BoT-SORT (the app) | 17 | 35.3% | 1.176 | -0.235 | 29.4% | 35.3% | 58.8% | 11.8% |
| 5+ | MaxN (what the app shows) | 17 | 11.8% | 1.765 | -1.412 | 17.6% | 70.6% | 41.2% | 23.5% |
| 5+ | BoT-SORT on human boxes | 17 | 52.9% | 0.706 | -0.118 | 17.6% | 29.4% | 88.2% | 11.8% |

### By species, at 2 fps (10 or more clips)

| species | condition | n | exact | mae | signed | over | under | within_1 | catastrophic |
|---|---|---|---|---|---|---|---|---|---|
| chimpanzee | BoT-SORT (the app) | 359 | 60.2% | 0.538 | 0.281 | 29.8% | 10.0% | 90.0% | 3.3% |
| chimpanzee | MaxN (what the app shows) | 359 | 57.9% | 0.507 | 0.167 | 29.0% | 13.1% | 93.6% | 1.7% |
| gorilla | BoT-SORT (the app) | 141 | 58.9% | 0.511 | 0.27 | 31.2% | 9.9% | 91.5% | 1.4% |
| gorilla | MaxN (what the app shows) | 141 | 61.7% | 0.482 | 0.199 | 26.2% | 12.1% | 90.8% | 0.7% |

## Experiment 1b: SA-FARI subset

### SA-FARI: the app's tracker and MaxN per sampling rate

| fps | condition | n | exact | mae | signed | over | under | within_1 | catastrophic |
|---|---|---|---|---|---|---|---|---|---|
| 0.5 | BoT-SORT (the app) | 184 | 54.4% | 0.766 | -0.06 | 18.5% | 27.2% | 84.8% | 4.9% |
| 0.5 | MaxN (what the app shows) | 184 | 69.6% | 0.505 | -0.158 | 8.2% | 22.3% | 90.8% | 3.8% |
| 1.0 | BoT-SORT (the app) | 184 | 60.9% | 0.739 | 0.13 | 20.6% | 18.5% | 87.0% | 6.5% |
| 1.0 | MaxN (what the app shows) | 184 | 68.5% | 0.495 | -0.049 | 12.0% | 19.6% | 91.8% | 2.7% |
| 2.0 | BoT-SORT (the app) | 184 | 60.3% | 0.707 | 0.293 | 25.0% | 14.7% | 88.0% | 3.8% |
| 2.0 | MaxN (what the app shows) | 184 | 65.2% | 0.533 | 0.033 | 16.9% | 17.9% | 91.8% | 3.3% |
| 3.0 | BoT-SORT (the app) | 184 | 63.0% | 0.658 | 0.31 | 25.0% | 12.0% | 90.2% | 4.3% |
| 3.0 | MaxN (what the app shows) | 184 | 66.3% | 0.533 | 0.065 | 17.4% | 16.3% | 90.8% | 3.3% |
| 6.0 | BoT-SORT (the app) | 200 | 59.5% | 0.785 | 0.525 | 31.0% | 9.5% | 83.5% | 6.0% |
| 6.0 | MaxN (what the app shows) | 200 | 63.0% | 0.555 | 0.145 | 22.5% | 14.5% | 91.5% | 3.0% |
| 6.0 | BoT-SORT production run (with camera motion) | 200 | 59.0% | 0.8 | 0.54 | 31.5% | 9.5% | 82.5% | 6.0% |

Detector cost per rate (frames run over the whole set, minutes at the master run's pace, decode included):

| fps | frames | detector_minutes |
|---|---|---|
| 0.5 | 1545 | 6.2 |
| 1.0 | 2979 | 12.0 |
| 2.0 | 5885 | 23.6 |
| 3.0 | 8761 | 35.1 |
| 6.0 | 19810 | 79.4 |

### Other trackers on the same boxes

| fps | condition | n | exact | mae | signed | over | under | within_1 | catastrophic |
|---|---|---|---|---|---|---|---|---|---|
| 1.0 | BoT-SORT (the app) | 184 | 60.9% | 0.739 | 0.13 | 20.6% | 18.5% | 87.0% | 6.5% |
| 1.0 | ByteTrack | 184 | 46.7% | 0.957 | 0.424 | 32.1% | 21.2% | 84.2% | 7.1% |
| 1.0 | ByteTrack, 1 frame | 184 | 45.1% | 0.978 | 0.446 | 33.7% | 21.2% | 84.2% | 7.1% |
| 1.0 | ByteTrack, real time | 184 | 45.6% | 1.0 | 0.467 | 33.1% | 21.2% | 83.2% | 7.1% |
| 1.0 | ByteTrack, real time, 1 frame | 184 | 44.0% | 1.022 | 0.489 | 34.8% | 21.2% | 83.2% | 7.1% |
| 1.0 | OC-SORT | 184 | 46.2% | 0.891 | -0.076 | 19.6% | 34.2% | 87.0% | 7.1% |
| 1.0 | OC-SORT, 1 frame | 184 | 42.4% | 1.163 | 0.446 | 31.0% | 26.6% | 84.2% | 7.6% |
| 1.0 | OC-SORT, real time | 184 | 45.6% | 0.908 | -0.049 | 20.1% | 34.2% | 87.0% | 7.1% |
| 1.0 | OC-SORT, real time, 1 frame | 184 | 41.9% | 1.163 | 0.467 | 31.5% | 26.6% | 84.8% | 8.2% |
| 2.0 | BoT-SORT (the app) | 184 | 60.3% | 0.707 | 0.293 | 25.0% | 14.7% | 88.0% | 3.8% |
| 2.0 | ByteTrack | 184 | 47.8% | 1.0 | 0.652 | 38.0% | 14.1% | 81.5% | 8.7% |
| 2.0 | ByteTrack, 1 frame | 184 | 45.1% | 1.043 | 0.707 | 41.3% | 13.6% | 81.0% | 9.2% |
| 2.0 | ByteTrack, real time | 184 | 47.8% | 1.033 | 0.685 | 38.0% | 14.1% | 79.9% | 9.2% |
| 2.0 | ByteTrack, real time, 1 frame | 184 | 45.1% | 1.076 | 0.739 | 41.3% | 13.6% | 79.9% | 10.3% |
| 2.0 | OC-SORT | 184 | 48.4% | 0.929 | 0.364 | 30.4% | 21.2% | 87.0% | 4.9% |
| 2.0 | OC-SORT, 1 frame | 184 | 41.3% | 1.277 | 0.853 | 41.9% | 16.9% | 78.8% | 13.0% |
| 2.0 | OC-SORT, real time | 184 | 46.7% | 0.929 | 0.364 | 32.1% | 21.2% | 87.5% | 4.3% |
| 2.0 | OC-SORT, real time, 1 frame | 184 | 40.8% | 1.304 | 0.859 | 42.4% | 16.9% | 77.2% | 13.6% |
| 3.0 | BoT-SORT (the app) | 184 | 63.0% | 0.658 | 0.31 | 25.0% | 12.0% | 90.2% | 4.3% |
| 3.0 | ByteTrack | 184 | 53.3% | 1.049 | 0.766 | 36.4% | 10.3% | 81.5% | 12.0% |
| 3.0 | ByteTrack, 1 frame | 184 | 52.7% | 1.082 | 0.799 | 37.0% | 10.3% | 81.5% | 13.0% |
| 3.0 | ByteTrack, real time | 184 | 53.3% | 1.06 | 0.777 | 36.4% | 10.3% | 81.5% | 12.0% |
| 3.0 | ByteTrack, real time, 1 frame | 184 | 52.7% | 1.092 | 0.81 | 37.0% | 10.3% | 81.5% | 13.0% |
| 3.0 | OC-SORT | 184 | 51.6% | 0.902 | 0.391 | 30.4% | 17.9% | 84.8% | 4.3% |
| 3.0 | OC-SORT, 1 frame | 184 | 45.6% | 1.25 | 0.859 | 39.7% | 14.7% | 74.5% | 9.2% |
| 3.0 | OC-SORT, real time | 184 | 51.6% | 0.908 | 0.386 | 29.9% | 18.5% | 84.8% | 4.3% |
| 3.0 | OC-SORT, real time, 1 frame | 184 | 45.6% | 1.261 | 0.859 | 39.7% | 14.7% | 75.0% | 9.2% |

### By true group size, at 2 fps

| size_bin | condition | n | exact | mae | signed | over | under | within_1 | catastrophic |
|---|---|---|---|---|---|---|---|---|---|
| 1 | BoT-SORT (the app) | 117 | 70.9% | 0.419 | 0.248 | 20.5% | 8.6% | 94.9% | 0.9% |
| 1 | MaxN (what the app shows) | 117 | 83.8% | 0.231 | 0.145 | 12.0% | 4.3% | 99.2% | 0.9% |
| 2 | BoT-SORT (the app) | 18 | 50.0% | 1.722 | 1.389 | 33.3% | 16.7% | 83.3% | 11.1% |
| 2 | MaxN (what the app shows) | 18 | 50.0% | 1.0 | 0.556 | 27.8% | 22.2% | 88.9% | 5.6% |
| 3 | BoT-SORT (the app) | 21 | 57.1% | 0.667 | 0.0 | 19.1% | 23.8% | 85.7% | 4.8% |
| 3 | MaxN (what the app shows) | 21 | 23.8% | 0.857 | -0.381 | 23.8% | 52.4% | 90.5% | 0.0% |
| 4 | BoT-SORT (the app) | 15 | 6.7% | 1.667 | -0.6 | 40.0% | 53.3% | 60.0% | 20.0% |
| 4 | MaxN (what the app shows) | 15 | 33.3% | 1.467 | -0.8 | 13.3% | 53.3% | 53.3% | 26.7% |
| 5+ | BoT-SORT (the app) | 13 | 46.2% | 0.846 | 0.692 | 46.2% | 7.7% | 69.2% | 0.0% |
| 5+ | MaxN (what the app shows) | 13 | 23.1% | 1.0 | -0.077 | 38.5% | 38.5% | 76.9% | 0.0% |

### By species, at 2 fps (10 or more clips)

| species | condition | n | exact | mae | signed | over | under | within_1 | catastrophic |
|---|---|---|---|---|---|---|---|---|---|
| trumpeter | BoT-SORT (the app) | 10 | 40.0% | 0.9 | 0.5 | 40.0% | 20.0% | 90.0% | 10.0% |
| trumpeter | MaxN (what the app shows) | 10 | 70.0% | 0.4 | -0.4 | 0.0% | 30.0% | 90.0% | 0.0% |
| white-nosed coati | BoT-SORT (the app) | 13 | 46.2% | 0.923 | 0.154 | 30.8% | 23.1% | 69.2% | 7.7% |
| white-nosed coati | MaxN (what the app shows) | 13 | 23.1% | 1.154 | 0.231 | 38.5% | 38.5% | 76.9% | 7.7% |

## Experiment 2: photo sequences

### Overall, kept iWildCam sequences

| condition | n | exact | mae | signed | over | under | within_1 | catastrophic |
|---|---|---|---|---|---|---|---|---|
| BoT-SORT (the app) | 1530 | 59.7% | 0.578 | 0.196 | 27.1% | 13.2% | 88.3% | 3.9% |
| MaxN (what the app shows) | 1530 | 64.3% | 0.489 | 0.162 | 23.7% | 12.0% | 91.2% | 2.9% |
| sum over photos | 1530 | 0.7% | 15.652 | 15.644 | 98.9% | 0.4% | 2.0% | 93.8% |

### By the median gap between photos

| gap_bin | condition | n | exact | mae | signed | over | under | within_1 | catastrophic |
|---|---|---|---|---|---|---|---|---|---|
| <=0.5 s | BoT-SORT (the app) | 207 | 54.6% | 0.638 | 0.444 | 36.7% | 8.7% | 87.4% | 3.4% |
| <=0.5 s | MaxN (what the app shows) | 207 | 65.2% | 0.449 | 0.188 | 23.7% | 11.1% | 92.3% | 1.0% |
| 0.5-1 s | BoT-SORT (the app) | 1217 | 60.7% | 0.569 | 0.157 | 25.3% | 14.0% | 88.3% | 3.9% |
| 0.5-1 s | MaxN (what the app shows) | 1217 | 64.5% | 0.495 | 0.15 | 23.3% | 12.2% | 91.0% | 3.4% |
| 1-2 s | BoT-SORT (the app) | 64 | 67.2% | 0.406 | 0.25 | 25.0% | 7.8% | 93.8% | 1.6% |
| 1-2 s | MaxN (what the app shows) | 64 | 70.3% | 0.359 | 0.203 | 23.4% | 6.2% | 93.8% | 0.0% |
| 2-5 s | BoT-SORT (the app) | 5 | 60.0% | 0.6 | -0.2 | 20.0% | 20.0% | 80.0% | 0.0% |
| 2-5 s | MaxN (what the app shows) | 5 | 60.0% | 0.4 | 0.4 | 40.0% | 0.0% | 100.0% | 0.0% |
| 5-10 s | BoT-SORT (the app) | 20 | 30.0% | 0.9 | 0.2 | 45.0% | 25.0% | 85.0% | 5.0% |
| 5-10 s | MaxN (what the app shows) | 20 | 45.0% | 0.55 | 0.05 | 30.0% | 25.0% | 100.0% | 0.0% |
| 10-30 s | BoT-SORT (the app) | 17 | 58.8% | 0.706 | -0.118 | 23.5% | 17.6% | 82.3% | 11.8% |
| 10-30 s | MaxN (what the app shows) | 17 | 41.2% | 0.941 | 0.588 | 41.2% | 17.6% | 76.5% | 11.8% |

### Tracking windows: BoT-SORT restarted at every gap over the window

| condition | n | exact | mae | signed | over | under | within_1 | catastrophic |
|---|---|---|---|---|---|---|---|---|
| BoT-SORT, window 10s, max of segments | 1530 | 62.1% | 0.539 | 0.137 | 24.2% | 13.7% | 89.0% | 3.3% |
| BoT-SORT, window 10s, sum of segments | 1530 | 52.0% | 1.18 | 0.955 | 39.7% | 8.3% | 76.2% | 14.6% |
| BoT-SORT, window 1s, max of segments | 1530 | 64.8% | 0.488 | 0.088 | 21.1% | 14.1% | 91.0% | 2.9% |
| BoT-SORT, window 1s, sum of segments | 1530 | 36.7% | 2.424 | 2.299 | 58.6% | 4.8% | 56.9% | 31.6% |
| BoT-SORT, window 2s, max of segments | 1530 | 62.7% | 0.523 | 0.118 | 23.4% | 13.9% | 89.7% | 3.1% |
| BoT-SORT, window 2s, sum of segments | 1530 | 48.5% | 1.612 | 1.416 | 44.1% | 7.4% | 70.9% | 20.2% |
| BoT-SORT, window 30s, max of segments | 1530 | 61.0% | 0.561 | 0.171 | 25.6% | 13.4% | 88.7% | 3.8% |
| BoT-SORT, window 30s, sum of segments | 1530 | 56.9% | 0.714 | 0.394 | 31.8% | 11.3% | 84.1% | 7.1% |
| BoT-SORT, window 5s, max of segments | 1530 | 62.2% | 0.538 | 0.13 | 23.9% | 13.9% | 89.1% | 3.3% |
| BoT-SORT, window 5s, sum of segments | 1530 | 51.1% | 1.346 | 1.133 | 41.0% | 7.9% | 74.7% | 16.4% |
| BoT-SORT, window 60s, max of segments | 1530 | 59.7% | 0.578 | 0.196 | 27.1% | 13.2% | 88.3% | 3.9% |
| BoT-SORT, window 60s, sum of segments | 1530 | 59.7% | 0.578 | 0.196 | 27.1% | 13.2% | 88.3% | 3.9% |

### Other trackers, with and without real time

| condition | n | exact | mae | signed | over | under | within_1 | catastrophic |
|---|---|---|---|---|---|---|---|---|
| ByteTrack | 1530 | 52.8% | 0.776 | 0.467 | 35.6% | 11.7% | 82.2% | 7.4% |
| ByteTrack, 1 frame | 1530 | 52.1% | 0.803 | 0.539 | 37.6% | 10.3% | 81.8% | 8.4% |
| ByteTrack, real time | 1530 | 46.9% | 1.22 | 0.859 | 42.3% | 10.8% | 72.8% | 16.0% |
| ByteTrack, real time, 1 frame | 1530 | 46.5% | 1.248 | 0.922 | 43.7% | 9.8% | 71.8% | 16.7% |
| OC-SORT | 1530 | 51.9% | 0.738 | -0.214 | 18.4% | 29.7% | 84.2% | 5.5% |
| OC-SORT, 1 frame | 1530 | 48.8% | 0.874 | 0.416 | 34.2% | 17.0% | 80.1% | 8.2% |
| OC-SORT, real time | 1530 | 48.6% | 0.907 | -0.238 | 19.1% | 32.3% | 79.6% | 9.2% |
| OC-SORT, real time, 1 frame | 1530 | 42.5% | 1.252 | 0.695 | 39.9% | 17.5% | 72.2% | 15.6% |

### By true count

| size_bin | condition | n | exact | mae | signed | over | under | within_1 | catastrophic |
|---|---|---|---|---|---|---|---|---|---|
| 1 | BoT-SORT (the app) | 340 | 78.5% | 0.318 | 0.282 | 19.7% | 1.8% | 93.2% | 2.1% |
| 1 | ByteTrack, real time, 1 frame | 340 | 59.7% | 0.576 | 0.382 | 30.6% | 9.7% | 89.4% | 3.5% |
| 1 | MaxN (what the app shows) | 340 | 75.0% | 0.397 | 0.362 | 23.2% | 1.8% | 92.1% | 2.9% |
| 2 | BoT-SORT (the app) | 307 | 73.0% | 0.322 | 0.277 | 24.8% | 2.3% | 96.4% | 0.7% |
| 2 | ByteTrack, real time, 1 frame | 307 | 58.6% | 0.805 | 0.629 | 35.5% | 5.9% | 79.5% | 8.5% |
| 2 | MaxN (what the app shows) | 307 | 74.3% | 0.316 | 0.27 | 23.4% | 2.3% | 95.8% | 1.0% |
| 3 | BoT-SORT (the app) | 351 | 66.4% | 0.405 | 0.228 | 25.1% | 8.6% | 94.3% | 0.9% |
| 3 | ByteTrack, real time, 1 frame | 351 | 52.7% | 0.989 | 0.795 | 40.5% | 6.8% | 76.3% | 13.1% |
| 3 | MaxN (what the app shows) | 351 | 77.5% | 0.256 | 0.182 | 18.8% | 3.7% | 97.4% | 0.3% |
| 4 | BoT-SORT (the app) | 259 | 42.5% | 0.784 | 0.158 | 30.5% | 27.0% | 85.7% | 4.6% |
| 4 | ByteTrack, real time, 1 frame | 259 | 30.5% | 1.649 | 1.147 | 54.8% | 14.7% | 61.4% | 25.1% |
| 4 | MaxN (what the app shows) | 259 | 48.6% | 0.66 | 0.166 | 27.8% | 23.5% | 90.0% | 3.1% |
| 5+ | BoT-SORT (the app) | 273 | 29.3% | 1.216 | -0.007 | 38.1% | 32.6% | 67.8% | 12.8% |
| 5+ | ByteTrack, real time, 1 frame | 273 | 23.8% | 2.535 | 1.875 | 62.6% | 13.6% | 45.1% | 38.8% |
| 5+ | MaxN (what the app shows) | 273 | 37.7% | 0.934 | -0.238 | 27.1% | 35.2% | 78.4% | 8.4% |

## Experiment 3: tuning the tracker and the threshold

### The counting threshold for MaxN

MaxN with the box floor moved from the app's 0.2 upwards. Video at 1, 2 and 3 fps; photos as sequences.

| dataset | fps | threshold | n | exact | mae | signed | over | under | within_1 | catastrophic |
|---|---|---|---|---|---|---|---|---|---|---|
| iwildcam |  | 0.2 | 1530 | 64.3% | 0.489 | 0.162 | 23.7% | 12.0% | 91.2% | 2.9% |
| iwildcam |  | 0.3 | 1530 | 68.2% | 0.423 | 0.031 | 17.4% | 14.4% | 92.6% | 2.3% |
| iwildcam |  | 0.4 | 1530 | 70.8% | 0.395 | -0.06 | 12.8% | 16.4% | 92.7% | 2.2% |
| iwildcam |  | 0.5 | 1530 | 71.2% | 0.388 | -0.135 | 9.9% | 18.9% | 92.9% | 2.3% |
| iwildcam |  | 0.6 | 1530 | 71.9% | 0.386 | -0.227 | 6.3% | 21.8% | 92.3% | 2.4% |
| iwildcam |  | 0.7 | 1530 | 68.0% | 0.457 | -0.361 | 3.9% | 28.1% | 89.9% | 2.8% |
| iwildcam |  | 0.8 | 1530 | 60.3% | 0.609 | -0.552 | 2.4% | 37.4% | 85.3% | 4.5% |
| panaf | 1 | 0.2 | 500 | 63.0% | 0.466 | 0.086 | 22.8% | 14.2% | 92.4% | 1.4% |
| panaf | 1 | 0.3 | 500 | 67.0% | 0.422 | -0.074 | 15.0% | 18.0% | 92.8% | 1.2% |
| panaf | 1 | 0.4 | 500 | 67.8% | 0.42 | -0.188 | 10.6% | 21.6% | 92.8% | 1.8% |
| panaf | 1 | 0.5 | 500 | 67.0% | 0.436 | -0.28 | 7.8% | 25.2% | 92.8% | 2.0% |
| panaf | 1 | 0.6 | 500 | 66.0% | 0.476 | -0.404 | 3.6% | 30.4% | 90.4% | 2.6% |
| panaf | 1 | 0.7 | 500 | 61.4% | 0.554 | -0.522 | 1.6% | 37.0% | 88.8% | 3.6% |
| panaf | 1 | 0.8 | 500 | 51.0% | 0.726 | -0.714 | 0.6% | 48.4% | 84.8% | 5.6% |
| panaf | 2 | 0.2 | 500 | 59.0% | 0.5 | 0.176 | 28.2% | 12.8% | 92.8% | 1.4% |
| panaf | 2 | 0.3 | 500 | 66.2% | 0.42 | 0.012 | 18.2% | 15.6% | 93.2% | 1.2% |
| panaf | 2 | 0.4 | 500 | 69.8% | 0.38 | -0.12 | 11.8% | 18.4% | 94.0% | 1.4% |
| panaf | 2 | 0.5 | 500 | 68.2% | 0.408 | -0.212 | 9.4% | 22.4% | 93.4% | 1.6% |
| panaf | 2 | 0.6 | 500 | 67.6% | 0.44 | -0.332 | 5.4% | 27.0% | 91.6% | 2.2% |
| panaf | 2 | 0.7 | 500 | 64.8% | 0.498 | -0.458 | 2.0% | 33.2% | 90.0% | 3.0% |
| panaf | 2 | 0.8 | 500 | 54.4% | 0.658 | -0.646 | 0.6% | 45.0% | 87.0% | 4.6% |
| panaf | 3 | 0.2 | 500 | 57.2% | 0.528 | 0.236 | 31.4% | 11.4% | 92.2% | 1.8% |
| panaf | 3 | 0.3 | 500 | 63.2% | 0.448 | 0.068 | 22.2% | 14.6% | 93.4% | 1.0% |
| panaf | 3 | 0.4 | 500 | 67.6% | 0.404 | -0.076 | 14.4% | 18.0% | 93.6% | 1.2% |
| panaf | 3 | 0.5 | 500 | 69.0% | 0.4 | -0.2 | 9.6% | 21.4% | 93.4% | 1.6% |
| panaf | 3 | 0.6 | 500 | 68.2% | 0.426 | -0.302 | 6.2% | 25.6% | 92.0% | 2.0% |
| panaf | 3 | 0.7 | 500 | 66.4% | 0.47 | -0.43 | 2.0% | 31.6% | 90.6% | 2.6% |
| panaf | 3 | 0.8 | 500 | 57.2% | 0.628 | -0.616 | 0.6% | 42.2% | 87.8% | 5.0% |
| safari | 1 | 0.2 | 184 | 68.5% | 0.495 | -0.049 | 12.0% | 19.6% | 91.8% | 2.7% |
| safari | 1 | 0.3 | 184 | 69.6% | 0.505 | -0.179 | 7.6% | 22.8% | 91.8% | 4.3% |
| safari | 1 | 0.4 | 184 | 71.2% | 0.489 | -0.283 | 2.7% | 26.1% | 90.8% | 4.3% |
| safari | 1 | 0.5 | 184 | 70.1% | 0.505 | -0.332 | 2.2% | 27.7% | 89.1% | 4.3% |
| safari | 1 | 0.6 | 184 | 69.0% | 0.533 | -0.37 | 1.6% | 29.3% | 88.6% | 4.9% |
| safari | 1 | 0.7 | 184 | 66.3% | 0.56 | -0.451 | 1.1% | 32.6% | 87.0% | 5.4% |
| safari | 1 | 0.8 | 184 | 58.7% | 0.679 | -0.614 | 0.5% | 40.8% | 84.8% | 6.5% |
| safari | 2 | 0.2 | 184 | 65.2% | 0.533 | 0.033 | 16.9% | 17.9% | 91.8% | 3.3% |
| safari | 2 | 0.3 | 184 | 67.4% | 0.511 | -0.13 | 10.3% | 22.3% | 92.4% | 3.8% |
| safari | 2 | 0.4 | 184 | 69.0% | 0.495 | -0.223 | 5.4% | 25.5% | 91.8% | 3.8% |
| safari | 2 | 0.5 | 184 | 70.1% | 0.489 | -0.293 | 3.3% | 26.6% | 90.2% | 4.3% |
| safari | 2 | 0.6 | 184 | 71.2% | 0.5 | -0.337 | 1.6% | 27.2% | 89.1% | 4.9% |
| safari | 2 | 0.7 | 184 | 69.0% | 0.522 | -0.413 | 1.1% | 29.9% | 87.5% | 5.4% |
| safari | 2 | 0.8 | 184 | 64.7% | 0.592 | -0.527 | 0.5% | 34.8% | 87.0% | 6.0% |
| safari | 3 | 0.2 | 184 | 66.3% | 0.533 | 0.065 | 17.4% | 16.3% | 90.8% | 3.3% |
| safari | 3 | 0.3 | 184 | 70.1% | 0.473 | -0.06 | 11.4% | 18.5% | 92.9% | 3.3% |
| safari | 3 | 0.4 | 184 | 70.1% | 0.473 | -0.168 | 7.1% | 22.8% | 92.9% | 3.8% |
| safari | 3 | 0.5 | 184 | 72.3% | 0.473 | -0.245 | 3.3% | 24.5% | 90.8% | 3.8% |
| safari | 3 | 0.6 | 184 | 70.7% | 0.484 | -0.31 | 2.2% | 27.2% | 90.8% | 4.9% |
| safari | 3 | 0.7 | 184 | 70.7% | 0.522 | -0.391 | 1.1% | 28.3% | 87.5% | 5.4% |
| safari | 3 | 0.8 | 184 | 64.1% | 0.598 | -0.511 | 1.1% | 34.8% | 86.4% | 6.0% |

### Tracker grid on PanAf train, best ten of the mean over 0.5 to 3 fps

Config names read `h<track start>_l<box floor>_m<match threshold>_b<lost buffer s>_<post-filter>`. The app is `h0.2_l0.01_m0.97_b2_none`.

| config | exact | mae |
|---|---|---|
| h0.3_l0.1_m0.97_b4_life0.5_keep0.7 | 64.3% | 0.445 |
| h0.3_l0.1_m0.97_b4_life1_keep0.5 | 64.3% | 0.439 |
| h0.3_l0.1_m0.97_b4_life1_keep0.7 | 64.2% | 0.439 |
| h0.4_l0.1_m0.97_b4_life1_keep0.5 | 64.2% | 0.453 |
| h0.4_l0.01_m0.97_b4_none | 64.2% | 0.454 |
| h0.4_l0.01_m0.97_b4_life1_keep0.5 | 64.1% | 0.453 |
| h0.3_l0.01_m0.97_b4_life0.5_keep0.7 | 64.1% | 0.447 |
| h0.5_l0.01_m0.97_b4_life1_keep0.5 | 64.1% | 0.454 |
| h0.5_l0.01_m0.97_b4_none | 64.1% | 0.454 |
| h0.4_l0.01_m0.97_b4_life0.5_keep0.7 | 64.1% | 0.454 |

### Frozen configurations on data they never saw

`app` is the tracker as shipped, `app+sharktrack` the shipped tracker with SharkTrack's post-filter, `best_overall` the grid's best single configuration, `best_at_Nfps` the best per rate. Selection used PanAf train only.

**panaf_heldout**

| fps | chosen | config | n | exact | mae | signed | over | under | within_1 | catastrophic |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.5 | app | h0.2_l0.01_m0.97_b2_none | 100 | 56.0% | 0.51 | -0.11 | 19.0% | 25.0% | 95.0% | 1.0% |
| 0.5 | best_at_1fps | h0.2_l0.01_m0.97_b2.0_life0.5_keep0.7 | 100 | 56.0% | 0.51 | -0.11 | 19.0% | 25.0% | 95.0% | 1.0% |
| 0.5 | best_at_0.5fps | h0.2_l0.01_m0.97_b4.0_life0.5_keep0.7 | 100 | 55.0% | 0.52 | -0.12 | 19.0% | 26.0% | 95.0% | 1.0% |
| 0.5 | best_at_3fps | h0.3_l0.1_m0.97_b4.0_sharktrack | 100 | 55.0% | 0.64 | -0.58 | 3.0% | 42.0% | 87.0% | 4.0% |
| 0.5 | best_overall | h0.3_l0.1_m0.97_b4.0_life0.5_keep0.7 | 100 | 54.0% | 0.57 | -0.29 | 13.0% | 33.0% | 92.0% | 1.0% |
| 0.5 | app+sharktrack | h0.2_l0.01_m0.97_b2_sharktrack | 100 | 54.0% | 0.62 | -0.54 | 4.0% | 42.0% | 89.0% | 4.0% |
| 0.5 | best_at_2fps | h0.2_l0.1_m0.97_b2.0_sharktrack | 100 | 54.0% | 0.62 | -0.54 | 4.0% | 42.0% | 89.0% | 4.0% |
| 1.0 | app+sharktrack | h0.2_l0.01_m0.97_b2_sharktrack | 100 | 64.0% | 0.48 | -0.34 | 6.0% | 30.0% | 91.0% | 2.0% |
| 1.0 | best_at_2fps | h0.2_l0.1_m0.97_b2.0_sharktrack | 100 | 64.0% | 0.48 | -0.34 | 6.0% | 30.0% | 91.0% | 2.0% |
| 1.0 | best_at_3fps | h0.3_l0.1_m0.97_b4.0_sharktrack | 100 | 60.0% | 0.53 | -0.35 | 8.0% | 32.0% | 91.0% | 2.0% |
| 1.0 | best_at_0.5fps | h0.2_l0.01_m0.97_b4.0_life0.5_keep0.7 | 100 | 55.0% | 0.53 | 0.11 | 28.0% | 17.0% | 94.0% | 1.0% |
| 1.0 | app | h0.2_l0.01_m0.97_b2_none | 100 | 55.0% | 0.55 | 0.13 | 28.0% | 17.0% | 92.0% | 1.0% |
| 1.0 | best_at_1fps | h0.2_l0.01_m0.97_b2.0_life0.5_keep0.7 | 100 | 55.0% | 0.55 | 0.13 | 28.0% | 17.0% | 92.0% | 1.0% |
| 1.0 | best_overall | h0.3_l0.1_m0.97_b4.0_life0.5_keep0.7 | 100 | 54.0% | 0.55 | -0.03 | 24.0% | 22.0% | 94.0% | 1.0% |
| 2.0 | app+sharktrack | h0.2_l0.01_m0.97_b2_sharktrack | 100 | 68.0% | 0.43 | -0.13 | 11.0% | 21.0% | 92.0% | 2.0% |
| 2.0 | best_at_2fps | h0.2_l0.1_m0.97_b2.0_sharktrack | 100 | 67.0% | 0.44 | -0.14 | 11.0% | 22.0% | 92.0% | 2.0% |
| 2.0 | best_at_3fps | h0.3_l0.1_m0.97_b4.0_sharktrack | 100 | 67.0% | 0.46 | -0.18 | 10.0% | 23.0% | 91.0% | 2.0% |
| 2.0 | best_overall | h0.3_l0.1_m0.97_b4.0_life0.5_keep0.7 | 100 | 58.0% | 0.57 | 0.21 | 29.0% | 13.0% | 90.0% | 3.0% |
| 2.0 | best_at_0.5fps | h0.2_l0.01_m0.97_b4.0_life0.5_keep0.7 | 100 | 55.0% | 0.6 | 0.38 | 37.0% | 8.0% | 90.0% | 4.0% |
| 2.0 | app | h0.2_l0.01_m0.97_b2_none | 100 | 53.0% | 0.64 | 0.42 | 39.0% | 8.0% | 88.0% | 4.0% |
| 2.0 | best_at_1fps | h0.2_l0.01_m0.97_b2.0_life0.5_keep0.7 | 100 | 53.0% | 0.64 | 0.42 | 39.0% | 8.0% | 88.0% | 4.0% |
| 3.0 | best_at_2fps | h0.2_l0.1_m0.97_b2.0_sharktrack | 100 | 71.0% | 0.4 | -0.06 | 13.0% | 16.0% | 94.0% | 3.0% |
| 3.0 | app+sharktrack | h0.2_l0.01_m0.97_b2_sharktrack | 100 | 70.0% | 0.39 | -0.05 | 14.0% | 16.0% | 95.0% | 3.0% |
| 3.0 | best_at_3fps | h0.3_l0.1_m0.97_b4.0_sharktrack | 100 | 70.0% | 0.41 | -0.11 | 12.0% | 18.0% | 93.0% | 3.0% |
| 3.0 | best_overall | h0.3_l0.1_m0.97_b4.0_life0.5_keep0.7 | 100 | 62.0% | 0.51 | 0.31 | 32.0% | 6.0% | 93.0% | 5.0% |
| 3.0 | best_at_0.5fps | h0.2_l0.01_m0.97_b4.0_life0.5_keep0.7 | 100 | 53.0% | 0.61 | 0.43 | 42.0% | 5.0% | 91.0% | 4.0% |
| 3.0 | best_at_1fps | h0.2_l0.01_m0.97_b2.0_life0.5_keep0.7 | 100 | 51.0% | 0.65 | 0.47 | 44.0% | 5.0% | 89.0% | 4.0% |
| 3.0 | app | h0.2_l0.01_m0.97_b2_none | 100 | 49.0% | 0.67 | 0.53 | 46.0% | 5.0% | 87.0% | 3.0% |

**safari**

| fps | chosen | config | n | exact | mae | signed | over | under | within_1 | catastrophic |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.5 | best_overall | h0.3_l0.1_m0.97_b4.0_life0.5_keep0.7 | 184 | 55.4% | 0.739 | -0.152 | 15.2% | 29.3% | 84.8% | 4.3% |
| 0.5 | best_at_0.5fps | h0.2_l0.01_m0.97_b4.0_life0.5_keep0.7 | 184 | 54.9% | 0.766 | -0.082 | 17.9% | 27.2% | 83.7% | 4.9% |
| 0.5 | app | h0.2_l0.01_m0.97_b2_none | 184 | 54.4% | 0.766 | -0.06 | 18.5% | 27.2% | 84.8% | 4.9% |
| 0.5 | best_at_1fps | h0.2_l0.01_m0.97_b2.0_life0.5_keep0.7 | 184 | 54.4% | 0.766 | -0.06 | 18.5% | 27.2% | 84.8% | 4.9% |
| 0.5 | app+sharktrack | h0.2_l0.01_m0.97_b2_sharktrack | 184 | 53.8% | 0.739 | -0.348 | 9.8% | 36.4% | 84.8% | 5.4% |
| 0.5 | best_at_2fps | h0.2_l0.1_m0.97_b2.0_sharktrack | 184 | 53.3% | 0.745 | -0.353 | 9.8% | 37.0% | 84.8% | 5.4% |
| 0.5 | best_at_3fps | h0.3_l0.1_m0.97_b4.0_sharktrack | 184 | 53.3% | 0.755 | -0.375 | 9.8% | 37.0% | 83.7% | 5.4% |
| 1.0 | app+sharktrack | h0.2_l0.01_m0.97_b2_sharktrack | 184 | 62.0% | 0.685 | -0.141 | 10.9% | 27.2% | 87.0% | 6.0% |
| 1.0 | best_overall | h0.3_l0.1_m0.97_b4.0_life0.5_keep0.7 | 184 | 62.0% | 0.696 | 0.0 | 16.3% | 21.7% | 88.0% | 6.0% |
| 1.0 | best_at_2fps | h0.2_l0.1_m0.97_b2.0_sharktrack | 184 | 61.4% | 0.69 | -0.147 | 10.9% | 27.7% | 87.0% | 6.0% |
| 1.0 | best_at_0.5fps | h0.2_l0.01_m0.97_b4.0_life0.5_keep0.7 | 184 | 60.9% | 0.728 | 0.109 | 20.1% | 19.0% | 87.5% | 6.0% |
| 1.0 | app | h0.2_l0.01_m0.97_b2_none | 184 | 60.9% | 0.739 | 0.13 | 20.6% | 18.5% | 87.0% | 6.5% |
| 1.0 | best_at_1fps | h0.2_l0.01_m0.97_b2.0_life0.5_keep0.7 | 184 | 60.9% | 0.739 | 0.13 | 20.6% | 18.5% | 87.0% | 6.5% |
| 1.0 | best_at_3fps | h0.3_l0.1_m0.97_b4.0_sharktrack | 184 | 60.3% | 0.696 | -0.185 | 10.3% | 29.3% | 88.0% | 6.0% |
| 2.0 | best_at_3fps | h0.3_l0.1_m0.97_b4.0_sharktrack | 184 | 65.8% | 0.582 | -0.049 | 12.5% | 21.7% | 91.8% | 4.9% |
| 2.0 | best_overall | h0.3_l0.1_m0.97_b4.0_life0.5_keep0.7 | 184 | 65.8% | 0.63 | 0.174 | 18.5% | 15.8% | 91.8% | 4.3% |
| 2.0 | app+sharktrack | h0.2_l0.01_m0.97_b2_sharktrack | 184 | 65.2% | 0.576 | -0.033 | 13.6% | 21.2% | 92.4% | 4.9% |
| 2.0 | best_at_2fps | h0.2_l0.1_m0.97_b2.0_sharktrack | 184 | 64.1% | 0.603 | -0.016 | 14.1% | 21.7% | 91.8% | 4.9% |
| 2.0 | best_at_0.5fps | h0.2_l0.01_m0.97_b4.0_life0.5_keep0.7 | 184 | 62.0% | 0.663 | 0.239 | 22.8% | 15.2% | 90.2% | 3.8% |
| 2.0 | app | h0.2_l0.01_m0.97_b2_none | 184 | 60.3% | 0.707 | 0.293 | 25.0% | 14.7% | 88.0% | 3.8% |
| 2.0 | best_at_1fps | h0.2_l0.01_m0.97_b2.0_life0.5_keep0.7 | 184 | 60.3% | 0.707 | 0.293 | 25.0% | 14.7% | 88.0% | 3.8% |
| 3.0 | app+sharktrack | h0.2_l0.01_m0.97_b2_sharktrack | 184 | 69.6% | 0.538 | 0.005 | 11.4% | 19.0% | 91.8% | 4.3% |
| 3.0 | best_at_3fps | h0.3_l0.1_m0.97_b4.0_sharktrack | 184 | 69.6% | 0.549 | -0.038 | 9.8% | 20.6% | 92.9% | 4.9% |
| 3.0 | best_at_2fps | h0.2_l0.1_m0.97_b2.0_sharktrack | 184 | 67.9% | 0.565 | 0.033 | 13.0% | 19.0% | 91.8% | 4.3% |
| 3.0 | best_overall | h0.3_l0.1_m0.97_b4.0_life0.5_keep0.7 | 184 | 67.9% | 0.576 | 0.087 | 14.7% | 17.4% | 93.5% | 3.8% |
| 3.0 | best_at_0.5fps | h0.2_l0.01_m0.97_b4.0_life0.5_keep0.7 | 184 | 64.7% | 0.609 | 0.196 | 20.6% | 14.7% | 90.8% | 3.3% |
| 3.0 | best_at_1fps | h0.2_l0.01_m0.97_b2.0_life0.5_keep0.7 | 184 | 64.1% | 0.63 | 0.228 | 21.7% | 14.1% | 91.3% | 4.3% |
| 3.0 | app | h0.2_l0.01_m0.97_b2_none | 184 | 63.0% | 0.658 | 0.31 | 25.0% | 12.0% | 90.2% | 4.3% |

**panaf_train**

| fps | chosen | config | n | exact | mae | signed | over | under | within_1 | catastrophic |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.5 | best_at_0.5fps | h0.2_l0.01_m0.97_b4.0_life0.5_keep0.7 | 400 | 58.8% | 0.54 | -0.21 | 13.5% | 27.8% | 90.8% | 2.2% |
| 0.5 | app | h0.2_l0.01_m0.97_b2_none | 400 | 58.5% | 0.54 | -0.205 | 13.8% | 27.8% | 90.8% | 2.0% |
| 0.5 | best_at_1fps | h0.2_l0.01_m0.97_b2.0_life0.5_keep0.7 | 400 | 58.5% | 0.54 | -0.205 | 13.8% | 27.8% | 90.8% | 2.0% |
| 0.5 | best_overall | h0.3_l0.1_m0.97_b4.0_life0.5_keep0.7 | 400 | 56.8% | 0.568 | -0.323 | 10.5% | 32.8% | 90.2% | 2.5% |
| 0.5 | app+sharktrack | h0.2_l0.01_m0.97_b2_sharktrack | 400 | 53.8% | 0.657 | -0.557 | 4.2% | 42.0% | 86.8% | 4.2% |
| 0.5 | best_at_2fps | h0.2_l0.1_m0.97_b2.0_sharktrack | 400 | 53.5% | 0.655 | -0.56 | 4.2% | 42.2% | 87.0% | 4.0% |
| 0.5 | best_at_3fps | h0.3_l0.1_m0.97_b4.0_sharktrack | 400 | 52.8% | 0.672 | -0.593 | 3.5% | 43.8% | 86.5% | 4.5% |
| 1.0 | app | h0.2_l0.01_m0.97_b2_none | 400 | 62.7% | 0.47 | -0.005 | 19.0% | 18.2% | 92.0% | 1.5% |
| 1.0 | best_at_0.5fps | h0.2_l0.01_m0.97_b4.0_life0.5_keep0.7 | 400 | 62.7% | 0.47 | -0.01 | 18.8% | 18.5% | 92.0% | 1.5% |
| 1.0 | best_at_1fps | h0.2_l0.01_m0.97_b2.0_life0.5_keep0.7 | 400 | 62.7% | 0.47 | -0.005 | 19.0% | 18.2% | 92.0% | 1.5% |
| 1.0 | best_overall | h0.3_l0.1_m0.97_b4.0_life0.5_keep0.7 | 400 | 62.3% | 0.47 | -0.165 | 13.2% | 24.5% | 92.5% | 1.2% |
| 1.0 | app+sharktrack | h0.2_l0.01_m0.97_b2_sharktrack | 400 | 61.3% | 0.525 | -0.42 | 4.8% | 34.0% | 89.8% | 2.8% |
| 1.0 | best_at_2fps | h0.2_l0.1_m0.97_b2.0_sharktrack | 400 | 61.3% | 0.525 | -0.42 | 4.8% | 34.0% | 89.8% | 2.8% |
| 1.0 | best_at_3fps | h0.3_l0.1_m0.97_b4.0_sharktrack | 400 | 60.5% | 0.53 | -0.46 | 3.5% | 36.0% | 90.5% | 3.0% |
| 2.0 | best_at_2fps | h0.2_l0.1_m0.97_b2.0_sharktrack | 400 | 69.2% | 0.375 | -0.195 | 8.0% | 22.8% | 94.8% | 1.5% |
| 2.0 | app+sharktrack | h0.2_l0.01_m0.97_b2_sharktrack | 400 | 68.2% | 0.383 | -0.193 | 8.8% | 23.0% | 94.8% | 1.2% |
| 2.0 | best_at_3fps | h0.3_l0.1_m0.97_b4.0_sharktrack | 400 | 68.0% | 0.385 | -0.23 | 7.0% | 25.0% | 94.8% | 1.2% |
| 2.0 | best_overall | h0.3_l0.1_m0.97_b4.0_life0.5_keep0.7 | 400 | 68.0% | 0.388 | 0.052 | 17.8% | 14.2% | 94.8% | 1.5% |
| 2.0 | best_at_0.5fps | h0.2_l0.01_m0.97_b4.0_life0.5_keep0.7 | 400 | 62.0% | 0.497 | 0.228 | 27.3% | 10.8% | 91.2% | 2.8% |
| 2.0 | app | h0.2_l0.01_m0.97_b2_none | 400 | 61.5% | 0.502 | 0.242 | 28.0% | 10.5% | 91.0% | 2.5% |
| 2.0 | best_at_1fps | h0.2_l0.01_m0.97_b2.0_life0.5_keep0.7 | 400 | 61.5% | 0.502 | 0.242 | 28.0% | 10.5% | 91.0% | 2.5% |
| 3.0 | best_at_3fps | h0.3_l0.1_m0.97_b4.0_sharktrack | 400 | 72.8% | 0.33 | -0.195 | 5.8% | 21.5% | 95.2% | 1.0% |
| 3.0 | best_at_2fps | h0.2_l0.1_m0.97_b2.0_sharktrack | 400 | 71.2% | 0.338 | -0.117 | 9.5% | 19.2% | 95.5% | 0.5% |
| 3.0 | app+sharktrack | h0.2_l0.01_m0.97_b2_sharktrack | 400 | 70.5% | 0.343 | -0.107 | 10.2% | 19.2% | 95.8% | 0.5% |
| 3.0 | best_overall | h0.3_l0.1_m0.97_b4.0_life0.5_keep0.7 | 400 | 70.2% | 0.355 | 0.065 | 16.8% | 13.0% | 94.8% | 0.5% |
| 3.0 | best_at_0.5fps | h0.2_l0.01_m0.97_b4.0_life0.5_keep0.7 | 400 | 61.5% | 0.48 | 0.295 | 30.0% | 8.5% | 91.8% | 1.2% |
| 3.0 | best_at_1fps | h0.2_l0.01_m0.97_b2.0_life0.5_keep0.7 | 400 | 61.0% | 0.497 | 0.307 | 30.8% | 8.2% | 90.8% | 1.5% |
| 3.0 | app | h0.2_l0.01_m0.97_b2_none | 400 | 59.8% | 0.532 | 0.362 | 32.8% | 7.5% | 89.5% | 2.5% |

### Photo sequences: grid on half the locations, scored on the other half

| set | chosen | config | n | exact | mae | signed | over | under | within_1 | catastrophic |
|---|---|---|---|---|---|---|---|---|---|---|
| dev | best_dev | h0.5_l0.01_m0.97_b2.0_life1_keep0.5 | 720 | 68.5% | 0.404 | -0.024 | 15.4% | 16.1% | 93.1% | 1.7% |
| dev | maxn | maxn_0.2 | 720 | 64.7% | 0.463 | 0.232 | 26.0% | 9.3% | 92.2% | 2.1% |
| dev | app | h0.2_l0.01_m0.97_b2_none | 720 | 61.9% | 0.504 | 0.229 | 26.9% | 11.1% | 91.1% | 2.5% |
| heldout | maxn | maxn_0.2 | 810 | 63.9% | 0.512 | 0.1 | 21.7% | 14.3% | 90.4% | 3.7% |
| heldout | best_dev | h0.5_l0.01_m0.97_b2.0_life1_keep0.5 | 810 | 59.9% | 0.572 | -0.1 | 18.4% | 21.7% | 88.9% | 4.2% |
| heldout | app | h0.2_l0.01_m0.97_b2_none | 810 | 57.8% | 0.643 | 0.167 | 27.2% | 15.1% | 85.8% | 5.1% |

## Experiment 4: a second detector, MD1000-SPRUCE on every frame

### PanAf500 with MD1000-SPRUCE: the app's tracker and MaxN per sampling rate

| fps | condition | n | exact | mae | signed | over | under | within_1 | catastrophic |
|---|---|---|---|---|---|---|---|---|---|
| 0.5 | BoT-SORT (the app) | 500 | 50.6% | 0.69 | -0.138 | 20.2% | 29.2% | 86.0% | 4.0% |
| 0.5 | MaxN (what the app shows) | 500 | 55.6% | 0.59 | 0.122 | 26.8% | 17.6% | 89.6% | 3.2% |
| 1.0 | BoT-SORT (the app) | 500 | 50.6% | 0.666 | 0.142 | 29.0% | 20.4% | 87.4% | 3.6% |
| 1.0 | MaxN (what the app shows) | 500 | 51.8% | 0.638 | 0.242 | 33.2% | 15.0% | 88.2% | 3.0% |
| 2.0 | BoT-SORT (the app) | 500 | 49.8% | 0.742 | 0.45 | 38.0% | 12.2% | 84.2% | 6.8% |
| 2.0 | MaxN (what the app shows) | 500 | 46.8% | 0.7 | 0.376 | 40.4% | 12.8% | 86.8% | 3.2% |
| 3.0 | BoT-SORT (the app) | 500 | 47.4% | 0.8 | 0.604 | 44.0% | 8.6% | 81.6% | 7.0% |
| 3.0 | MaxN (what the app shows) | 500 | 45.4% | 0.736 | 0.42 | 42.4% | 12.2% | 85.4% | 3.4% |
| 4.0 | BoT-SORT (the app) | 500 | 47.4% | 0.868 | 0.696 | 45.2% | 7.4% | 78.2% | 8.8% |
| 4.0 | MaxN (what the app shows) | 500 | 41.8% | 0.778 | 0.482 | 46.6% | 11.6% | 84.6% | 3.6% |
| 6.0 | BoT-SORT (the app) | 500 | 41.6% | 1.022 | 0.898 | 53.0% | 5.4% | 74.6% | 11.0% |
| 6.0 | MaxN (what the app shows) | 500 | 40.6% | 0.826 | 0.582 | 50.0% | 9.4% | 82.2% | 4.2% |
| 12.0 | BoT-SORT (the app) | 500 | 33.4% | 1.494 | 1.402 | 62.4% | 4.2% | 62.8% | 21.2% |
| 12.0 | MaxN (what the app shows) | 500 | 38.0% | 0.892 | 0.688 | 54.0% | 8.0% | 79.4% | 5.0% |
| 24.0 | BoT-SORT (the app) | 500 | 24.0% | 2.362 | 2.306 | 73.6% | 2.4% | 48.6% | 35.4% |
| 24.0 | MaxN (what the app shows) | 500 | 35.8% | 0.928 | 0.756 | 57.2% | 7.0% | 79.0% | 5.4% |
| 24.0 | BoT-SORT production run (with camera motion) | 500 | 24.0% | 2.37 | 2.314 | 73.6% | 2.4% | 48.2% | 35.4% |

Detector cost per rate (frames run over the whole set, minutes at the master run's pace, decode included):

| fps | frames | detector_minutes |
|---|---|---|
| 0.5 | 4000 | 2.6 |
| 1.0 | 7500 | 4.8 |
| 2.0 | 15000 | 9.7 |
| 3.0 | 22497 | 14.5 |
| 4.0 | 29997 | 19.3 |
| 6.0 | 44994 | 29.0 |
| 12.0 | 89984 | 58.0 |
| 24.0 | 179957 | 116.0 |

### Detector or tracker: the same tracker on human boxes

| fps | condition | n | exact | mae | signed | over | under | within_1 | catastrophic |
|---|---|---|---|---|---|---|---|---|---|
| 0.5 | BoT-SORT (the app) | 500 | 50.6% | 0.69 | -0.138 | 20.2% | 29.2% | 86.0% | 4.0% |
| 0.5 | MaxN on human boxes | 500 | 80.4% | 0.244 | -0.244 | 0.0% | 19.6% | 96.6% | 1.0% |
| 0.5 | BoT-SORT on human boxes | 500 | 69.0% | 0.402 | -0.306 | 4.4% | 26.6% | 93.4% | 1.8% |
| 1.0 | BoT-SORT (the app) | 500 | 50.6% | 0.666 | 0.142 | 29.0% | 20.4% | 87.4% | 3.6% |
| 1.0 | MaxN on human boxes | 500 | 82.6% | 0.214 | -0.214 | 0.0% | 17.4% | 97.0% | 0.6% |
| 1.0 | BoT-SORT on human boxes | 500 | 78.6% | 0.256 | -0.136 | 5.8% | 15.6% | 96.2% | 0.4% |
| 2.0 | BoT-SORT (the app) | 500 | 49.8% | 0.742 | 0.45 | 38.0% | 12.2% | 84.2% | 6.8% |
| 2.0 | MaxN on human boxes | 500 | 83.6% | 0.196 | -0.196 | 0.0% | 16.4% | 97.6% | 0.6% |
| 2.0 | BoT-SORT on human boxes | 500 | 86.6% | 0.154 | 0.034 | 7.8% | 5.6% | 98.6% | 0.6% |
| 3.0 | BoT-SORT (the app) | 500 | 47.4% | 0.8 | 0.604 | 44.0% | 8.6% | 81.6% | 7.0% |
| 3.0 | MaxN on human boxes | 500 | 84.2% | 0.192 | -0.192 | 0.0% | 15.8% | 97.4% | 0.6% |
| 3.0 | BoT-SORT on human boxes | 500 | 89.6% | 0.124 | 0.064 | 7.4% | 3.0% | 98.6% | 0.4% |
| 4.0 | BoT-SORT (the app) | 500 | 47.4% | 0.868 | 0.696 | 45.2% | 7.4% | 78.2% | 8.8% |
| 4.0 | MaxN on human boxes | 500 | 83.6% | 0.196 | -0.196 | 0.0% | 16.4% | 97.6% | 0.6% |
| 4.0 | BoT-SORT on human boxes | 500 | 90.4% | 0.118 | 0.07 | 7.2% | 2.4% | 98.4% | 0.4% |
| 6.0 | BoT-SORT (the app) | 500 | 41.6% | 1.022 | 0.898 | 53.0% | 5.4% | 74.6% | 11.0% |
| 6.0 | MaxN on human boxes | 500 | 84.2% | 0.19 | -0.19 | 0.0% | 15.8% | 97.6% | 0.6% |
| 6.0 | BoT-SORT on human boxes | 500 | 90.2% | 0.126 | 0.086 | 7.8% | 2.0% | 98.0% | 0.6% |
| 12.0 | BoT-SORT (the app) | 500 | 33.4% | 1.494 | 1.402 | 62.4% | 4.2% | 62.8% | 21.2% |
| 12.0 | MaxN on human boxes | 500 | 84.2% | 0.19 | -0.19 | 0.0% | 15.8% | 97.6% | 0.6% |
| 12.0 | BoT-SORT on human boxes | 500 | 90.4% | 0.132 | 0.096 | 7.8% | 1.8% | 97.4% | 0.6% |
| 24.0 | BoT-SORT (the app) | 500 | 24.0% | 2.362 | 2.306 | 73.6% | 2.4% | 48.6% | 35.4% |
| 24.0 | MaxN on human boxes | 500 | 84.2% | 0.19 | -0.19 | 0.0% | 15.8% | 97.6% | 0.6% |
| 24.0 | BoT-SORT on human boxes | 500 | 90.2% | 0.132 | 0.1 | 8.2% | 1.6% | 97.2% | 0.4% |

### Other trackers on the same boxes

| fps | condition | n | exact | mae | signed | over | under | within_1 | catastrophic |
|---|---|---|---|---|---|---|---|---|---|
| 1.0 | BoT-SORT (the app) | 500 | 50.6% | 0.666 | 0.142 | 29.0% | 20.4% | 87.4% | 3.6% |
| 1.0 | ByteTrack | 500 | 40.8% | 0.942 | 0.646 | 46.0% | 13.2% | 78.4% | 8.0% |
| 1.0 | ByteTrack, 1 frame | 500 | 40.0% | 0.986 | 0.702 | 47.4% | 12.6% | 76.6% | 9.2% |
| 1.0 | ByteTrack, real time | 500 | 40.4% | 0.948 | 0.66 | 46.6% | 13.0% | 78.6% | 8.0% |
| 1.0 | ByteTrack, real time, 1 frame | 500 | 39.2% | 0.996 | 0.716 | 48.2% | 12.6% | 76.8% | 9.2% |
| 1.0 | OC-SORT | 500 | 48.8% | 0.71 | -0.13 | 20.8% | 30.4% | 86.0% | 4.2% |
| 1.0 | OC-SORT, 1 frame | 500 | 41.4% | 0.854 | 0.354 | 38.6% | 20.0% | 83.6% | 5.4% |
| 1.0 | OC-SORT, real time | 500 | 48.8% | 0.716 | -0.124 | 20.8% | 30.4% | 85.6% | 4.0% |
| 1.0 | OC-SORT, real time, 1 frame | 500 | 40.4% | 0.886 | 0.362 | 38.2% | 21.4% | 82.0% | 5.4% |
| 2.0 | BoT-SORT (the app) | 500 | 49.8% | 0.742 | 0.45 | 38.0% | 12.2% | 84.2% | 6.8% |
| 2.0 | ByteTrack | 500 | 34.8% | 1.202 | 1.058 | 59.2% | 6.0% | 71.0% | 14.6% |
| 2.0 | ByteTrack, 1 frame | 500 | 33.8% | 1.238 | 1.098 | 60.4% | 5.8% | 69.8% | 15.2% |
| 2.0 | ByteTrack, real time | 500 | 34.8% | 1.214 | 1.074 | 59.2% | 6.0% | 70.4% | 14.8% |
| 2.0 | ByteTrack, real time, 1 frame | 500 | 33.8% | 1.25 | 1.114 | 60.4% | 5.8% | 69.2% | 15.2% |
| 2.0 | OC-SORT | 500 | 49.0% | 0.754 | 0.37 | 34.8% | 16.2% | 85.4% | 6.0% |
| 2.0 | OC-SORT, 1 frame | 500 | 40.6% | 1.126 | 0.906 | 50.2% | 9.2% | 72.8% | 13.8% |
| 2.0 | OC-SORT, real time | 500 | 48.2% | 0.772 | 0.396 | 35.6% | 16.2% | 85.0% | 6.2% |
| 2.0 | OC-SORT, real time, 1 frame | 500 | 40.0% | 1.156 | 0.936 | 50.6% | 9.4% | 72.4% | 14.6% |
| 3.0 | BoT-SORT (the app) | 500 | 47.4% | 0.8 | 0.604 | 44.0% | 8.6% | 81.6% | 7.0% |
| 3.0 | ByteTrack | 500 | 31.6% | 1.366 | 1.25 | 63.6% | 4.8% | 66.4% | 17.6% |
| 3.0 | ByteTrack, 1 frame | 500 | 31.6% | 1.412 | 1.304 | 63.8% | 4.6% | 64.6% | 18.8% |
| 3.0 | ByteTrack, real time | 500 | 31.6% | 1.372 | 1.256 | 63.6% | 4.8% | 66.0% | 17.8% |
| 3.0 | ByteTrack, real time, 1 frame | 500 | 31.6% | 1.418 | 1.31 | 63.8% | 4.6% | 64.2% | 19.0% |
| 3.0 | OC-SORT | 500 | 48.6% | 0.788 | 0.508 | 39.2% | 12.2% | 84.2% | 6.0% |
| 3.0 | OC-SORT, 1 frame | 500 | 35.0% | 1.326 | 1.182 | 58.4% | 6.6% | 68.4% | 15.2% |
| 3.0 | OC-SORT, real time | 500 | 48.0% | 0.816 | 0.532 | 39.6% | 12.4% | 82.6% | 6.2% |
| 3.0 | OC-SORT, real time, 1 frame | 500 | 34.8% | 1.354 | 1.206 | 58.6% | 6.6% | 67.2% | 16.2% |

### By true group size, at 2 fps

| size_bin | condition | n | exact | mae | signed | over | under | within_1 | catastrophic |
|---|---|---|---|---|---|---|---|---|---|
| 1 | BoT-SORT (the app) | 291 | 58.8% | 0.553 | 0.43 | 35.0% | 6.2% | 90.4% | 3.4% |
| 1 | MaxN (what the app shows) | 291 | 53.3% | 0.581 | 0.533 | 44.3% | 2.4% | 90.4% | 1.7% |
| 1 | BoT-SORT on human boxes | 291 | 94.5% | 0.058 | 0.052 | 5.1% | 0.3% | 99.7% | 0.0% |
| 2 | BoT-SORT (the app) | 116 | 43.1% | 0.853 | 0.543 | 42.2% | 14.7% | 80.2% | 6.9% |
| 2 | MaxN (what the app shows) | 116 | 46.6% | 0.672 | 0.414 | 40.5% | 12.9% | 88.8% | 2.6% |
| 2 | BoT-SORT on human boxes | 116 | 85.3% | 0.155 | 0.017 | 7.8% | 6.9% | 99.1% | 0.0% |
| 3 | BoT-SORT (the app) | 53 | 28.3% | 1.226 | 0.509 | 43.4% | 28.3% | 73.6% | 20.8% |
| 3 | MaxN (what the app shows) | 53 | 30.2% | 1.075 | 0.132 | 32.1% | 37.7% | 75.5% | 9.4% |
| 3 | BoT-SORT on human boxes | 53 | 71.7% | 0.358 | 0.094 | 15.1% | 13.2% | 94.3% | 1.9% |
| 4 | BoT-SORT (the app) | 23 | 39.1% | 0.87 | 0.435 | 39.1% | 21.7% | 78.3% | 4.3% |
| 4 | MaxN (what the app shows) | 23 | 34.8% | 0.826 | -0.217 | 21.7% | 43.5% | 82.6% | 0.0% |
| 4 | BoT-SORT on human boxes | 23 | 52.2% | 0.478 | -0.13 | 17.4% | 30.4% | 100.0% | 0.0% |
| 5+ | BoT-SORT (the app) | 17 | 23.5% | 1.529 | 0.0 | 41.2% | 35.3% | 47.1% | 23.5% |
| 5+ | MaxN (what the app shows) | 17 | 5.9% | 1.588 | -1.0 | 23.5% | 70.6% | 52.9% | 17.6% |
| 5+ | BoT-SORT on human boxes | 17 | 52.9% | 0.706 | -0.118 | 17.6% | 29.4% | 88.2% | 11.8% |

### By species, at 2 fps (10 or more clips)

| species | condition | n | exact | mae | signed | over | under | within_1 | catastrophic |
|---|---|---|---|---|---|---|---|---|---|
| chimpanzee | BoT-SORT (the app) | 359 | 47.3% | 0.816 | 0.526 | 40.7% | 12.0% | 80.8% | 7.8% |
| chimpanzee | MaxN (what the app shows) | 359 | 44.9% | 0.763 | 0.435 | 42.6% | 12.5% | 83.8% | 4.5% |
| gorilla | BoT-SORT (the app) | 141 | 56.0% | 0.553 | 0.255 | 31.2% | 12.8% | 92.9% | 4.3% |
| gorilla | MaxN (what the app shows) | 141 | 51.8% | 0.539 | 0.227 | 34.8% | 13.5% | 94.3% | 0.0% |

## Worst cases to look at

The ten largest overcounts and undercounts of the app's tracker at 2 fps on PanAf500; the full list is `results/worst-cases.csv` on the drive, the clips under `data/panaf500/videos/`.

| kind | video | truth | pred | maxn | species |
|---|---|---|---|---|---|
| over | EyuKOstXhY.mp4 | 1 | 5 | 3 | chimpanzee |
| over | Z4ZwPuzupc.mp4 | 2 | 6 | 6 | chimpanzee |
| over | nrPU7Atyon.mp4 | 1 | 4 | 3 | chimpanzee |
| over | SKHCNVHmft.mp4 | 1 | 4 | 2 | chimpanzee |
| over | Xjb3fltpXq.mp4 | 4 | 7 | 5 | chimpanzee |
| over | vn9uLaFwBc.mp4 | 3 | 6 | 4 | gorilla |
| over | gQLDfMAwO7.mp4 | 2 | 5 | 3 | chimpanzee |
| over | hGZKTKLzXv.mp4 | 2 | 5 | 3 | chimpanzee |
| over | 3m2q70XD8D.mp4 | 2 | 5 | 5 | chimpanzee |
| over | jf1K5KaZSi.mp4 | 1 | 4 | 2 | chimpanzee |
| under | AsisxnkSUK.mp4 | 6 | 3 | 3 | chimpanzee |
| under | MnYV3cYzBq.mp4 | 8 | 5 | 4 | chimpanzee |
| under | xPuolR0EIs.mp4 | 3 | 0 | 2 | gorilla |
| under | HinLZooQYI.mp4 | 2 | 0 | 1 | chimpanzee |
| under | 9Jb6Sv1nJ9.mp4 | 2 | 0 | 1 | chimpanzee |
| under | HCIdN8aetT.mp4 | 6 | 4 | 3 | chimpanzee |
| under | 3QRVeJfsB9.mp4 | 3 | 1 | 2 | chimpanzee |
| under | ujdmSGkG5C.mp4 | 6 | 4 | 5 | gorilla |
| under | 69v9sVUjYV.mp4 | 3 | 1 | 2 | chimpanzee |
| under | DXwFk5ohZm.mp4 | 3 | 1 | 2 | chimpanzee |

## Limits

- Video tonight is apes only (plus the SA-FARI subset if it ran); 720 px footage. Camera-trap mammals at 1080p may behave differently.
- The offline replays skip BoT-SORT's camera-motion compensation; the validation section says what that cost on these fixed cameras.
- ultralytics BoT-SORT never reports a track seen on one sampled frame unless it is the first frame. This is the app's behaviour, and it undercounts animals that cross in under two samples.
- iWildCam counts of 9 were dropped as a suspected 9+ cap (176 sequences); 68 empties counted as 1 dropped. MD5A boxes were used, not the MDv4 boxes shipped with the dataset.
- The Roboflow trackers ran with the app's floors and two-second budget, otherwise package defaults. Experiment 3 tunes only the app's own tracker, on PanAf train and half of iWildCam's locations; SA-FARI is never tuned on.
- PanAf500's licence page was unreachable tonight; the videos came from Bristol's open file server. SA-FARI and MammAlps are CC BY-NC.
- ultralytics is AGPL-3.0 inside an MIT app. Unrelated to the counts, noted for the licence review the plan asks for.
## Decisions, reasons and changes (2026-09-14)

Settled with Peter after the run, one rule per setting. The tuning and the threshold sweep that informed them ran on the same cached MD5A boxes; the numbers below are from data the choice was not tuned on unless stated.

### The six points

| Point | Rule | Why | Change made |
|---|---|---|---|
| Count shown | MaxN over every frame and file of the event, everywhere. No track count. | Track count never beats MaxN on unseen data: as shipped 60% vs 59 to 63% (PanAf), with the filter 68 to 71% vs 70% (MaxN at 0.4). Tracking across photos loses to MaxN out of sample (60% vs 64%). Track count only wins on human boxes (90% vs 84%), so it needs cleaner detections first. | none (what the code does) |
| Sampling rate | `video_fps` default 3 for every detector | With the post-filter on, 3 fps is the peak on both video sets (PanAf held-out 64, 68, 71% at 1, 2, 3 fps; SA-FARI 62, 66, 70%) and SharkTrack's own rate. Without the filter more fps was worse (61% at 1 fps to 43% at 12), because the tracker turned detector flicker into extra tracks. | default 2.0 to 3.0 in `models/project.py`, `schemas/project.py`, `advancedSettingsDefaults.ts` |
| Tracker | ultralytics BoT-SORT as shipped; nothing across photos | The four knobs (track start, box floor, match threshold, lost buffer) barely moved the count in a 128-config grid on PanAf train; every winner kept the shipped values. Roboflow ByteTrack and OC-SORT never beat it (ByteTrack 45 to 53% vs 60% at 2 fps); real-time and one-frame variants worse. The AGPL question is a licence decision, parked in TODO.md. | none |
| Track post-filter | On for every detector, values keyed by data domain: camera traps motion 0.06 / exempt 0.5, underwater 0.08 / exempt 0.7 (SharkTrack's) | The whole tuning gain came from the "nearly static" rule. SharkTrack's values lift PanAf held-out from 49% to 70% exact at 3 fps but empty 9% of clips of a resting animal (the track is deleted, the clip is filed blank, never reviewed), against 2% with no filter. 0.06 / 0.5 keeps 65% (SA-FARI 63 to 67%) and empties 4%. The rule rests on "a real animal moves", true under water and not on a camera trap, which is a domain fact; the underwater values are SharkTrack's published ones, validated on 207 h of BRUV footage, and were not re-measured. | new catalog field `domain` on every detector (replaces `track_filter`); `app/ml/track_filter.py` holds the two value pairs; `tracking_script.py` always filters and takes `--track_filter_motion` / `--track_filter_exempt`; worker passes the detector's domain values |
| Counting threshold | 0.5 everywhere | 0.2 was MegaDetector's *display* default for v5, inherited as the counting floor. MaxN against human counts: PanAf 57% at 0.2 to 69% at 0.5 (3 fps), SA-FARI 66% to 72%, iWildCam photos 64% to 71%; peak flat 0.4 to 0.6, overcount below, undercount above. Underwater unmeasured: SharkTrack starts a track at 0.4 and counts human-confirmed tracks without a threshold, CFD documents no threshold at all. One number by decision; the Mote BRUV sweep is the follow-up. | `DEFAULT_COUNTING_THRESHOLD` 0.2 to 0.5 in `confidence.py` and `confidence.ts`; existing projects keep their value |
| Detector | MD5A stays the camera-trap default | MD1000-SPRUCE on every frame is about 10 points worse at every rate (2 fps: track 50% vs 60%, MaxN 47% vs 59%). REDWOOD and LARCH untested. | none; REDWOOD benchmark on the follow-up list |

### The tuning that informed them

Post-filter sweep on the app's tracker settings at 3 fps (`results/motion_sweep_rows.csv`); `zero` = clips counted 0 with an animal present:

| motion | exempt | PanAf held-out exact | SA-FARI exact | resting-only clips exact | resting-only zero |
|---|---|---|---|---|---|
| none | | 49% | 63% | 55% | 2.2% |
| 0.02 | 0.5 | 59% | 66% | 66% | 4.3% |
| 0.04 | 0.5 | 63% | 67% | 66% | 4.3% |
| 0.06 | 0.5 | 65% | 67% | 69% | 4.3% |
| 0.06 | 0.7 | 70% | 70% | 75% | 7.6% |
| 0.08 | 0.7 (SharkTrack) | 70% | 70% | 73% | 8.7% |

Resting-only clips are the 92 PanAf clips where every ape carries only the `sitting` behaviour label for all 15 s (`results/panaf_resting.json`). The floor of false empties at 0.5 is two clips whose animal never scores above 0.5; no exemption we would accept rescues them.

Photo sequences (iWildCam, grid on half the locations, scored on the other half): the tuned tracker wins where it was tuned (68% vs MaxN 65%) and loses on held-out locations (60% vs MaxN 64%). Tuning does not transfer; MaxN stays.

### Code changes in the repo (uncommitted at the time of writing)

`models.json`, `backend/app/ml/schemas/model_manifest.py`, `backend/app/ml/track_filter.py` (new) and its test, `backend/app/ml/inference/tracking_script.py` (filter parameters, `--raw_detections_json` for benchmark caches), `backend/app/ml/inference/video_detector.py`, `backend/app/workers/detection_worker.py`, `backend/app/core/confidence.py`, `frontend/src/lib/confidence.ts`, `backend/app/models/project.py`, `backend/app/api/schemas/project.py`, `frontend/src/lib/advancedSettingsDefaults.ts`, tests under `backend/tests/ml/` and `tests/models/`, `tests/api/test_project_duplicate.py`, `DEVELOPERS.md`, two user docs pages. No migration: only new projects and folder runs get the new defaults.

### Follow-ups

- Counting threshold sweep on the Mote BRUV clips against a hand count (settles 0.5 for SharkTrack and CFD). The scripts on the drive do it in an hour once a labelled clip set exists.
- MD1000-REDWOOD overnight benchmark with the same scripts (`detect.py` plus a model path).
- The ultralytics AGPL licence (TODO.md).
- If marine users see junk static tracks surviving at 0.5 to 0.7 or the reverse, the domain values are the one place to change.

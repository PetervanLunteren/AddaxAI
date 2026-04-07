# SpeciesNet: official API vs AddaxAI postprocessing comparison

Detailed comparison of how the official SpeciesNet pipeline (`run_md_and_speciesnet` via the `speciesnet` and `megadetector` packages) processes classification results vs how AddaxAI does it. Written 2026-03-30 after a hands-on comparison using 46 images from a Kenya deployment.

## Test setup

- Deployment: `/Users/peter/Downloads/example-data/project_Kenya/Chui River/deployment_001`
- AddaxAI project: `b227f8c6-5c0c-49af-8d1e-b85a972e54dd`
- Country: KEN (Kenya)
- Detection model: MegaDetector v5A
- Classification model: SpeciesNet v4.0.1a (`SPECIESNET-v4-0-1-A-v1`)
- Official API output: `SPPNET_ground_truth.json` (generated with `run_md_and_speciesnet --country KEN`)
- AddaxAI output: `.addaxai/projects/.../results.json` + DB detections

## Current comparison results (2026-04-03, smoothing off, non-label skip disabled)

### Kenya (KEN): 1867 classified detections

| Category | Count |
|----------|-------|
| Exact match | 1863 |
| Confidence-only diff | 3 |
| Label differences | 1 |
| GT only / DB only | 0 |

Match rate: 99.8%. The 1 label difference and 3 confidence differences are all caused by the taxonomy ancestor resolution difference described below (difference #8).

### Netherlands (NLD): 2400 classified detections

| Category | Count |
|----------|-------|
| Exact match | 2372 |
| Confidence-only diff | 27 |
| Label differences | 1 |
| GT only / DB only | 0 |

Match rate: 98.8%. The 1 label difference is `mammalia -> bovidae` (difference #8: AddaxAI picks family, official API picks class). The 27 confidence differences have two causes:
- 3 bovidae rollups where AddaxAI's family sum is higher because more species contribute (difference #8)
- 24 aves rollups where AddaxAI keeps the raw "bird" label at 0.67 while the official API rolls up to "aves" at 0.80 (difference #2: the official API's ensemble combiner triggers rollup for non-species top-1 labels like "bird", AddaxAI does not)

### Raw model output verification (NLD, 841 species-level detections)

Comparing only species-level detections (no rollup involved) between the official API's GT JSON and AddaxAI's results.json: zero label differences, 658 confidence "diffs" that are all just rounding (GT rounds to 3 decimals, AddaxAI to 5). The raw classifier outputs are identical. All label and confidence differences in the comparison above come from the rollup stage, not the model.

## Official SpeciesNet pipeline (3 stages)

Source code is installed at `~/AddaxAI/envs/env-addaxai-base/lib/python3.11/site-packages/speciesnet/`.

### Stage 1: classifier (`speciesnet/classifier.py`)

- Runs the EfficientNet V2 M model on each crop
- Returns **only the top-5 predictions** per detection (line 241: `torch.topk(scores, k=5)`)
- This is the most important difference from AddaxAI, which stores all 2498 predictions

### Stage 2: ensemble combiner (`speciesnet/ensemble_prediction_combiner.py`)

`combine_predictions_for_single_item()` is the core decision engine. It combines detector results (MegaDetector categories: animal/person/vehicle) with classifier results (species predictions) using a series of heuristic thresholds:

1. **Threshold #1a-b (human)**: high-confidence human detection (>0.7) overrides classifier. Mid-confidence human detection (>0.2) + high-confidence human/vehicle classification (>0.5) also returns human.

2. **Threshold #2a-c (vehicle)**: similar heuristics for vehicle detection.

3. **Threshold #3a-b (blank)**: low detection confidence (<0.2) + high blank classification (>0.5) returns blank. Also: extra-high blank classification (>0.99) returns blank regardless of detection.

4. **Threshold #4a-b (confident animal)**: if top-1 classification is an animal species (not blank/human/vehicle):
   - If classification confidence >0.8: call `geofence_animal_classification()` (may geofence + rollup)
   - If classification confidence >0.65 AND detection confidence >0.2: same

5. **Threshold #5a (rollup)**: `roll_up_labels_to_first_matching_level()` with:
   - `target_taxonomy_levels=["genus", "family", "order", "class", "kingdom"]`
   - `non_blank_threshold=0.65`
   - Uses the **top-5 predictions** for rollup sums

6. **Threshold #5b (animal fallback)**: if detection confidence >0.5 and detection class is animal, return "animal"

7. **Unknown fallback**: return unknown with top classification score

### Stage 3: geofence + rollup (`speciesnet/geofence_utils.py`)

Two functions:

**`geofence_animal_classification()`** (line 207):
- Called from step 4a/4b when top-1 is a confident animal species
- Checks if top-1 species IS geofenced (not allowed in country) via `should_geofence_animal_classification()`
- If NOT geofenced (species IS allowed): returns species as-is (line 280)
- If geofenced (species NOT allowed): calls rollup with `non_blank_threshold = scores[0] - 1e-10` (threshold equals the top-1 score, not 0.65)
- The rollup target levels here are `["family", "order", "class", "kingdom"]` (skips species and genus)

**`roll_up_labels_to_first_matching_level()`** (line 106):
- For each target taxonomy level:
  - Groups all predictions by their ancestor at that level (via `get_ancestor_at_level()` using `taxonomy_map`)
  - Sums scores per group
  - Finds the max-scoring group that passes the threshold AND is not geofenced
  - If found, returns it
- Returns `None` if no level crosses the threshold

**`should_geofence_animal_classification()`** (line 34):
- Checks geofence_map for the species' full class string
- If species is NOT in geofence_map at all: returns `False` (do NOT geofence; allowed by default)
- If species IS in geofence_map: checks "allow" and "block" rules for the country
- Returns `True` (should geofence) if country is not in the allow list

## AddaxAI pipeline (current, after session fixes)

### Phase 6: DB load (`backend/app/ml/json_pipeline.py`)

- Stores **all 2498 predictions** per detection in the JSON (raw model output)
- Uses `filter_and_rollup_classifications()` to redirect excluded species' scores to taxonomy ancestors (exclusion rollup) for DB label assignment
- Uses `should_skip_detection()` to skip detections whose top-1 (after exclusion) is a non-label class

### Phase 7: postprocessing (`backend/app/ml/postprocessing.py`)

Order of operations in `run_postprocessing_for_deployment()`:
1. `apply_label_exclusion_to_results()`: removes excluded species from classification lists (simple drop, no redirection to ancestors, no renormalization). NON_LABEL classes (blank, bait, etc.) are kept.
2. `apply_taxonomic_rollup_to_results()`: for detections with top-1 confidence < 0.65, sums top-5 predictions' confidences at each taxonomy level. Picks the most specific level above 0.65. Returns None if nothing crosses threshold.
3. `strip_non_label_from_results()`: removes blank/bait/etc. from classification lists.
4. Event smoothing (MegaDetector subprocess, non-fatal).
5. `update_database_from_smoothed_results()`: writes final labels to DB.

### Rollup implementation (`backend/app/ml/taxonomic_rollup.py`)

`rollup_single_detection()`:
- Triggers when top-1 confidence < 0.65 AND top-1 is in taxonomy
- Uses only top-5 predictions for rollup sums (matches official SpeciesNet API)
- Sums top-5 scores at each level (species, genus, family, order, class)
- Walks from most specific to broadest, picks first level >= 0.65
- Returns None if nothing crosses threshold

## Key differences

### 1. Predictions stored vs used in rollup

| | Official API | AddaxAI |
|---|---|---|
| Predictions stored | Top-5 only | All 2498 |
| Predictions used in rollup sums | Top-5 | Top-5 (trimmed before summing) |

Both systems now use top-5 for rollup sums. AddaxAI stores all 2498 in the JSON (needed for Phase 6 exclusion rollup which redirects scores to ancestors), but `rollup_single_detection()` trims to top-5 before computing level sums.

### 2. Decision tree vs simple threshold

| | Official API | AddaxAI |
|---|---|---|
| When rollup triggers | Complex heuristic: only when species conf < 0.65 AND detection heuristics don't match earlier | Simple: when top-1 conf < 0.65 |
| Blank handling | Heuristic: detection conf < 0.2 + blank cls > 0.5 | Non-label skip: top-1 is blank after exclusion |
| Human/vehicle | Heuristic thresholds using detection confidence | Not classified (person/vehicle detections skip classification) |

The official API's ensemble combiner uses BOTH detection and classification confidence in a complex decision tree. AddaxAI separates these concerns: MegaDetector handles detection categories, and classification handles species.

### 3. Geofence check on rollup results

| | Official API | AddaxAI |
|---|---|---|
| Geofence on rollup result | Yes: each rollup candidate is checked against geofence (line 190-193 in `roll_up_labels_to_first_matching_level`) | No: any taxonomy ancestor can be a rollup result |

The official API ensures that the rolled-up ancestor label is itself allowed in the country. If "bovidae family" were geofenced in a country, the rollup would skip it and try order level instead. AddaxAI doesn't perform this check.

### 4. Rollup threshold when geofenced vs general

| | Official API | AddaxAI |
|---|---|---|
| General rollup threshold | 0.65 (step #5a) | 0.65 (hardcoded `ROLLUP_THRESHOLD`) |
| Geofence rollup threshold | top-1 score (e.g. 0.38) | N/A (no separate geofence rollup) |

When the official API geofences a species (step 4 → `geofence_animal_classification`), the rollup uses `non_blank_threshold = scores[0] - 1e-10`, meaning any ancestor sum that exceeds the original top-1 score qualifies. This is a LOWER threshold than 0.65.

### 5. Taxonomy levels checked

| | Official API (step #5a) | Official API (geofence rollup) | AddaxAI |
|---|---|---|---|
| Levels | genus, family, order, class, kingdom | family, order, class, kingdom | species, genus, family, order, class |

The official API skips species level in step #5a rollup (starts at genus). The geofence rollup skips both species and genus (starts at family). AddaxAI starts at species and doesn't include kingdom.

### 6. Label exclusion approach

| | Official API | AddaxAI (Phase 6 DB load) | AddaxAI (Phase 7 postprocessing) |
|---|---|---|---|
| Excluded species | Geofence per species at output time | Exclusion rollup: redirect scores to taxonomy ancestors | Simple drop: remove from classification list |

The official API doesn't pre-filter the classification list. It runs the model on all 2498 classes, takes top-5, then applies geofencing at the output stage. AddaxAI pre-filters during both DB load (with ancestor redirection) and postprocessing (simple drop).

### 7. Fallback when nothing crosses threshold

| | Official API | AddaxAI |
|---|---|---|
| Rollup finds nothing above threshold | Returns None → ensemble falls through to "animal" (if detection > 0.5) or "unknown" | Returns None → keeps original top-1 label |

### 8. Taxonomy ancestor resolution (primary source of remaining differences)

| | Official API | AddaxAI |
|---|---|---|
| How ancestors are found | `get_ancestor_at_level()` looks up the ancestor key in `taxonomy_map` (built from the model's `.labels.txt` file). Returns `None` if the ancestor has no label in the model. | Groups by taxon value from `taxonomy.csv`. Every family/order/class that appears in the taxonomy is a valid rollup target. |
| Effect | Can only roll up to levels that have a dedicated label in the model's training data. 172 out of 280 families have no label, so rollup skips them and falls through to order or class. | Can roll up to any taxonomy level that exists in the taxonomy CSV, producing more specific labels. |

The official API's `roll_up_labels_to_first_matching_level()` calls `get_ancestor_at_level()` (in `taxonomy_utils.py`) which constructs a 5-part key (e.g., `mammalia;cetartiodactyla;bovidae;;`) and looks it up in `taxonomy_map`. If no entry exists for that key, the function returns `None` and the species does not contribute to the family-level sum. For 172 out of 280 families (e.g., callitrichidae, pycnonotidae), there is no family-level entry in the model's `.labels.txt`, so the rollup cannot stop at family level and falls through to order or class.

AddaxAI's `rollup_single_detection()` groups by taxon value from the taxonomy CSV (e.g., `entry["family"] == "bovidae"`). Every species with a family value contributes to that family's sum, regardless of whether the model has a family-level label. This produces more specific, taxonomically correct rollup results.

This is an intentional design difference. Rollup targets should be based on taxonomy, not on what labels happened to be in the model's training data. AddaxAI's approach gives users more informative labels (e.g., "callitrichidae" instead of "mammalia").

## Geofence data

Both systems use the same geofence file: `geofence_release.2025.02.27.0702.json` in the model directory. Format: `{ "taxonomy_key": { "allow": { "KEN": [], "USA": ["CA","FL"], ... } } }`.

- Total model labels: 2493 (from `.labels.txt`)
- Geofence-allowed for Kenya: 573
- Geofence-excluded for Kenya: 1920
- User's `excluded_classes`: 1920 (identical to geofence-excluded)
- Labels not in geofence at all (allowed by default): 8
- Extra entries in `classification_categories` not in labels file: 5 (`malabaricus`, `arnee`, `bres`, `stella`, `cinnamomeus`, `inornata`)

The GT's 23 `classification_categories` are just the distinct top-1 labels that appear in the output across all 46 images. They are NOT a species filter. 7 of the 23 are not geofence-allowed for Kenya (e.g. "eastern gray squirrel", "pronghorn") because the official API keeps the raw top-1 when it doesn't trigger any of the ensemble heuristics.

## Taxonomy note

Both systems use the same taxonomy data. There is a split in the order level: 7 species use `artiodactyla` (e.g. domestic sheep, dromedary camel) while 162 use `cetartiodactyla` (e.g. domestic cattle, impala). This is a taxonomy convention difference (Artiodactyla vs merged Cetartiodactyla). Both systems have the same mapping; it does not cause any comparison differences.

## Fixes applied in this session

These changes are in the working tree (not yet committed):

1. **Removed renormalization** from `filter_classifications()` in `label_exclusion.py`. Confidences keep raw values after filtering.

2. **Added descriptions** for exclusion rollup ancestor categories in `filter_and_rollup_classifications()`. Fixes `KeyError: '2566'` crash in MegaDetector smoothing.

3. **Moved NON_LABEL stripping to after rollup**. `apply_label_exclusion_to_results()` no longer strips blank/bait before taxonomic rollup. New `strip_non_label_from_results()` helper is called after rollup. Matches official API behavior where blank is present during rollup.

4. **Simplified Phase 7 exclusion**. `apply_label_exclusion_to_results()` now drops excluded species without redirecting to ancestors (simple `filter_classifications`). Only Phase 6 DB load uses `filter_and_rollup_classifications()` for ancestor redirection. Matches official API which doesn't redistribute excluded scores.

5. **Removed rollup fallback**. When no taxonomy level crosses the 0.65 threshold, `rollup_single_detection()` returns None (keep original label) instead of returning the broadest available level. Matches official API.

6. **Isolated smoothing failure from rollup**. Smoothing crash in `run_postprocessing_for_deployment()` now returns rollup-only results instead of propagating the exception. Phase 7 in `detection_worker.py` is fatal (rollup must succeed).

## Remaining known differences

### Taxonomy ancestor resolution (intentional, not a bug)

All remaining label and confidence differences stem from difference #8 above. AddaxAI produces more specific rollup labels because it resolves taxonomy ancestors from the taxonomy CSV rather than requiring a label in the model's training data. This is by design.

### Ensemble heuristics

The official API uses a complex decision tree that combines detection confidence with classification confidence (human/vehicle/blank heuristics). AddaxAI separates these concerns: MegaDetector handles detection categories, classification handles species. Not planned to change.

### `included_ancestor_taxa` check (intentional AddaxAI addition)

AddaxAI checks that a rollup ancestor has at least one non-excluded descendant species. The official API only checks if the ancestor itself is geofenced. This prevents useless labels (e.g., "canidae" when all canidae species are excluded). Kept as a deliberate improvement.

## Files reference

### Official SpeciesNet (installed package)
- `~/AddaxAI/envs/env-addaxai-base/lib/python3.11/site-packages/speciesnet/classifier.py` (top-5 prediction)
- `~/AddaxAI/envs/env-addaxai-base/lib/python3.11/site-packages/speciesnet/ensemble.py` (ensemble orchestrator)
- `~/AddaxAI/envs/env-addaxai-base/lib/python3.11/site-packages/speciesnet/ensemble_prediction_combiner.py` (heuristic decision tree)
- `~/AddaxAI/envs/env-addaxai-base/lib/python3.11/site-packages/speciesnet/geofence_utils.py` (geofence check + rollup)
- `~/AddaxAI/envs/env-addaxai-base/lib/python3.11/site-packages/speciesnet/taxonomy_utils.py` (ancestor lookup)

### AddaxAI
- `backend/app/ml/label_exclusion.py` (exclusion, filtering, non-label skip)
- `backend/app/ml/taxonomic_rollup.py` (rollup algorithm)
- `backend/app/ml/postprocessing.py` (Phase 7 orchestrator)
- `backend/app/ml/json_pipeline.py` (Phase 6 DB load)
- `backend/app/ml/geofence.py` (geofence data loading)
- `backend/app/workers/detection_worker.py` (pipeline phases)

### Model data
- `~/AddaxAI/models/cls/SPECIESNET-v4-0-1-A-v1/taxonomy.csv` (taxonomy mapping)
- `~/AddaxAI/models/cls/SPECIESNET-v4-0-1-A-v1/geofence_release.2025.02.27.0702.json` (geofence rules)
- `~/AddaxAI/models/cls/SPECIESNET-v4-0-1-A-v1/always_crop_99710272_22x8_v12_epoch_00148.labels.txt` (2493 labels with taxonomy)

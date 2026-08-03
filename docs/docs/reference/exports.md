---
sidebar_position: 1
title: Exports
---

# Exports

What is in the files AddaxAI writes. Most of this page covers the columns in the CSV and XLSX tables, which hold the same columns either way; XLSX just puts each table on its own sheet. The recognition file is described at the end.

AddaxAI exports four tables. Which one you want depends on the question you are asking. A folder run has no sites, no deployments and no confirmed counts, so two of them are projects only.

| Table | One row is | Use it for | Available in |
|---|---|---|---|
| Counts | one species in one event | ecological analysis. Start here | Projects |
| Detections | one box on one photo | model checking, bounding boxes | Both |
| Files | one photo or video | one label per file, file lists, finding blanks | Both |
| Deployments | one camera period | effort, trap nights, locations | Projects |

## Counts

One row per species per event, with the count. Each row is one [observation](../understanding/detections-events-observations.mdx), so an animal is counted once per event instead of once per photo. This is the analysis-ready table.

| Column | Meaning |
|---|---|
| `event_id` | Identifier of the event |
| `deployment_id` | Which camera period it came from |
| `event_start` | Time of the first photo in the event, camera local time |
| `event_end` | Time of the last photo in the event |
| `category` | animal, person or vehicle |
| `classification_label` | The species label as used by the model |
| `taxon_class` | Class, for example mammalia |
| `taxon_order` | Order, for example carnivora |
| `taxon_family` | Family, for example canidae |
| `taxon_genus` | Genus, for example vulpes |
| `taxon_species` | Species |
| `scientific_name` | Scientific name for display |
| `common_name` | Common name for display |
| `count` | Number of individuals. Your confirmed number if you set one, otherwise the AI's highest number seen in a single photo |
| `is_confirmed` | TRUE if you signed off the count for this event |

## Detections

One row per box. Use it when you care about individual boxes. Blank files do not appear here.

| Column | Meaning |
|---|---|
| `detection_id` | Identifier of the box |
| `file_id` | Which file it is on |
| `deployment_id` | Which camera period |
| `event_id` | Which event, empty if not grouped |
| `detection_category` | animal, person or vehicle |
| `detection_confidence` | How sure the detector was there is something there |
| `classification_label` | The current species label. May be your correction |
| `classification_confidence` | Score for the current label. Always 1.0 when a human set it |
| `ai_classification_label` | The label the app showed before you touched it, kept even after you relabel |
| `ai_classification_confidence` | Score for that label |
| `classification_method` | machine or human, who set the current label |
| `is_verified` | TRUE if you checked this detection |
| `taxon_class` to `taxon_species` | Taxonomy, broad to specific |
| `scientific_name` | Scientific name |
| `common_name` | Common name |
| `frame_number` | Frame index for videos, empty for photos |
| `bbox_x`, `bbox_y` | Top left corner of the box, 0 to 1 |
| `bbox_width`, `bbox_height` | Size of the box, 0 to 1 |

Box positions are fractions of the image, not pixels. Multiply by the image width and height to get pixels.

To see where the AI was wrong, compare `ai_classification_label` with `classification_label` on rows where `is_verified` is TRUE.

One thing to know before you read too much into that. `ai_classification_label` is the label after cleanup, the one the app put in front of you, not the model's raw output. That raw call stays in the `results.json` on disk. So the comparison scores the whole pipeline, model plus rollup plus smoothing. See [how labels get cleaned up](../understanding/label-cleanup.md).

## Files

One row per photo or video, whether or not anything was found.

| Column | Meaning |
|---|---|
| `file_id` | Identifier of the file |
| `deployment_id` | Which camera period |
| `event_id` | Which event, empty if not grouped |
| `file_type` | image or video |
| `relative_path` | Path inside the deployment folder |
| `absolute_path` | Full path on the machine that ran the analysis |
| `datetime` | Capture time, camera local time. Empty if the file had no readable date |
| `observation_type` | What the file holds, taken from its strongest box: the one you checked yourself, or else the one the detector scored highest. For a video, only boxes on the one frame AddaxAI saved count. The value is whatever the detector called it, so animal, person or vehicle for MegaDetector, or blank when no box passed |
| `detection_confidence` | How sure the detector was there is something there, for that same box. A verified box counts whatever its score, so this can sit below your [counting threshold](../understanding/confidence-and-verification.md) |
| `classification_label` | The species of that same strongest box, not the most confident species on the file. Empty for a person, a vehicle, or an animal that was never classified |
| `classification_confidence` | Score for that species. Always 1.0 when a human set the label. Empty when there is no species |
| `taxon_class` to `taxon_species` | Taxonomy of that species, broad to specific. Only a species label fills all five, so check these before you group by `classification_label` |
| `scientific_name` | Scientific name of that same box |
| `common_name` | Common name of that same box. Never empty on a file that holds something: it reads `Person`, `Vehicle` or `Animal` when there is no species |
| `is_verified` | TRUE if every box on the file was checked |
| `notes` | Your own notes |

`observation_type` is the only place "blank" appears. Use it to count empty files.

## Deployments

One row per camera period. This is your effort table.

| Column | Meaning |
|---|---|
| `deployment_id` | Identifier of the camera period |
| `site_name` | Name of the location |
| `latitude`, `longitude` | Location in decimal degrees |
| `site_elevation_m` | Elevation in metres, if you entered it |
| `site_habitat` | Habitat type, if you entered it |
| `site_notes` | Your notes on the site |
| `site_tags` | Your tags on the site |
| `deployment_start` | First day of the period |
| `deployment_end` | Last day, empty if the camera is still out |
| `trap_nights` | How long the camera was out: the days from its first file to its last, counting both ends. Not the number of days that produced photos. See [how trap nights are counted](../understanding/trap-nights.md) |
| `deployment_notes` | Your notes on this deployment |
| `deployment_tags` | Your tags on this deployment |

## Folder runs

In the two tables a folder run writes, `deployment_id` is dropped because there is no deployment, and `notes` because nothing ever fills it.

`event_id` stays. Files and detections from the same burst share one, so you can still group by visit. What you cannot do is look the event up, because the counts table is projects only. In a project that column points at a row in counts; in a folder run it is only a grouping key.

## Recognition file (JSON)

A folder run also writes `addaxai-recognitions.json`. This is the file Timelapse reads.

It follows the MegaDetector output format, version 1.6, which is [documented here](https://microsoft.github.io/MegaDetector/output_format/). Four things are worth knowing on top of that spec:

1. The `info` block carries an extra `addaxai` section with the app version and the settings the run used, such as smoothing, rollup and the independence interval. So the file records how it was produced.
2. Each detection keeps only the top classification, not the full list the format allows.
3. Nothing is filtered by a threshold. Every detection AddaxAI stored is in the file.
4. File paths are relative to the folder the JSON sits in.

## Spatial

Projects can also export point layers for GIS tools such as QGIS and ArcGIS, as GeoJSON, Shapefile or GeoPackage. You get two layers: one point per camera, and one point per camera per species. Deployments with no site coordinates are left out.

## Camtrap DP

Projects can also export Camtrap DP, a standard format for camera trap data. It uses its own column names and a fixed structure, so other tools and archives can read your data without knowing anything about AddaxAI. Use it when you share data or deposit it in a repository.

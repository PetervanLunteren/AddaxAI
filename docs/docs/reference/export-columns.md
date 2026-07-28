---
sidebar_position: 1
title: Export columns
---

# Export columns

What every column in the CSV and XLSX files means. The two formats hold the same columns; XLSX just puts each table on its own sheet.

There are four tables. Which one you want depends on the question you are asking.

| Table | One row is | Use it for |
|---|---|---|
| Observations | one species in one event | ecological analysis. Start here |
| Detections | one box on one photo | model checking, bounding boxes |
| Files | one photo or video | file lists, finding blanks |
| Deployments | one camera period | effort, trap nights, locations |

:::tip Which table should I use?

For "how many animals did I see", use **observations**. It counts each animal once per event instead of once per photo. See [detections, events and observations](../understanding/detections-events-observations.mdx).

:::

## Observations

One row per species per event, with the count. This is the analysis-ready table.

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

One row per box. Use it when you care about individual boxes rather than ecology. Blank files do not appear here.

| Column | Meaning |
|---|---|
| `detection_id` | Identifier of the box |
| `file_id` | Which file it is on |
| `deployment_id` | Which camera period |
| `event_id` | Which event, empty if not grouped |
| `detection_category` | animal, person or vehicle |
| `detection_confidence` | How sure the detector was there is something there |
| `classification_label` | The current species label. May be your correction |
| `classification_confidence` | Score for the current label |
| `ai_classification_label` | What the AI originally said, kept even after you change it |
| `ai_classification_confidence` | Score for the AI's original label |
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
| `observation_type` | What the file holds overall: animal, human, vehicle or blank |
| `is_verified` | TRUE if every box on the file was checked |
| `notes` | Your own notes |

`observation_type` is the only place "blank" appears. Use it to count empty photos.

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
| `trap_nights` | Days the camera produced photos. See [how trap nights are counted](../understanding/trap-nights.md) |
| `deployment_notes` | Your notes on this deployment |
| `deployment_tags` | Your tags on this deployment |

## Folder runs

A folder run writes the same tables, minus the columns that need a project. There is no `deployment_id` and no notes column, because a folder run has no sites or deployments.

It also writes a recognition file for Timelapse and a text file describing the models and settings used.

## Camtrap DP

Projects can also export Camtrap DP, a standard format for camera trap data. It uses its own column names and a fixed structure, so other tools and archives can read your data without knowing anything about AddaxAI. Use it when you share data or deposit it in a repository.

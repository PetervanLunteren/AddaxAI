---
sidebar_position: 3
title: Choose a workflow
---

# Choose a workflow

AddaxAI has two ways of working. They use the same AI models, so the species labels are the same. What differs is how much the app remembers, and which results you can get.

Pick one before you start. You can turn a folder run into a project later, but not the other way around.

## Analyse a folder

You point at one folder. The app runs the AI, you check the labels, then it writes files into that folder.

Good when you have one batch of photos and you want tables or sorted folders.

**You get:** detection tables, a recognition file for Timelapse, species separated folders, images with the animals marked.

**You do not get:** a map, activity charts, trap night rates, or any history. There is no site, no camera location, and no record of when the camera was out.

## Build a project

You create sites and deployments, then add folders to them. The app keeps everything: detections, your corrections, and the counts you confirm.

Good when you run several cameras, or the same cameras over more than one season.

**You get:** everything a folder run gives, plus a dashboard, a map, activity charts, trap night rates, a verification history, and Camtrap DP export.

**It asks more from you:** a location for each camera, and a start and end date for each deployment.

## How this changes your results

This is the part that matters for analysis.

| | Analyse a folder | Build a project |
|---|---|---|
| Species labels | Same | Same |
| Grouped into events | No | Yes |
| Counts per species | Per file | Per event, using the highest number seen in one frame |
| Trap nights | Not counted | Counted per deployment |
| Rates per 100 trap nights | Not available | Available |
| Map | No | Yes |
| Your corrections are kept | Until you delete the run | Yes, with a history |

A folder run counts detections. A project counts **observations**, which is a species within an event. Ten photos of one deer in two minutes are ten detections, but one observation. For most ecological work the observation is what you want. See [detections, events and observations](../understanding/detections-events-observations.mdx).

## Still not sure

Use a project. It does everything a folder run does, and you can ignore the parts you do not need. The only cost is entering a location and dates for each camera.

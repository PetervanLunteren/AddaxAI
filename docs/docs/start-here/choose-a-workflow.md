---
sidebar_position: 3
title: Choose a workflow
---

# Choose a workflow

AddaxAI has two ways of working. Both run the same AI, so the detections and the species labels are identical. What differs is what happens next: a folder run hands you the raw results, a project helps you interpret them.

Nothing is locked in. A folder run can become a project later. You keep every detection and every correction, and nothing is analysed again.

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

## What each one gives you

The AI output is the same in both. The difference is the layer AddaxAI builds on top of it.

| | Analyse a folder | Build a project |
|---|---|---|
| Detections and species labels | Same | Same |
| Grouped into events | No | Yes |
| Counts per species | You work them out yourself | Per event, using the highest number seen in one frame |
| Trap nights | Not counted | Counted per deployment |
| Rates per 100 trap nights | Not available | Available |
| Map | No | Yes |
| Your corrections are kept | Until you delete the run | Yes, with a history |

A folder run hands you detections. A project turns them into **observations**, which is a species within an event. Ten photos of one deer in two minutes are ten detections, but one observation. For most ecological work the observation is what you want. See [detections, events and observations](../understanding/detections-events-observations.mdx).

## Why a folder run has no counts

This is a deliberate split, not a missing feature.

A folder run stops after the AI. It finds the animals, names them, lets you correct labels, then writes the raw results out: a table of files, a table of detections, and a recognition file. It never groups photos into events and never decides how many individuals were there.

That is the point. You take those tables into the tool you already use, such as Timelapse, R or Zooniverse, and do the interpreting there.

A project goes further. It groups photos into events, counts individuals per event, and works out rates per 100 trap nights. Counts need events, and events only exist in projects.

So: use a folder run when another program will do the analysis. Use a project when you want AddaxAI to do it.

## Still not sure

Start with a folder run. It is the quickest way to see what the AI found, and you can promote it to a project whenever you want more, without losing anything or analysing again.

Go straight to a project if you already know you want counts, rates or a map, and you have the camera locations and dates to hand.

---
sidebar_position: 2
title: FAQ
---

# Frequently asked questions

## Getting started

### Which workflow should I use?

Use a project if you want AddaxAI to do the interpreting: counts, rates, charts and maps. Use a folder run if another program will do the analysis, and you only want the AI to find and name the animals.

Both run the same AI, so the detections and species are the same either way. If you are unsure, start with a folder run: it can become a project later without losing anything. See [choose a workflow](../start-here/choose-a-workflow.md).

### Is it free?

Yes. AddaxAI is free and open source.

### Which systems does it run on?

Windows, macOS (Apple Silicon) and Linux. Intel Macs are not supported.

## Privacy and data

### Does my data get uploaded anywhere?

No. Everything runs on your own computer. Your photos and videos never leave your machine. AddaxAI only goes online to download models and check for updates.

## Requirements

### Do I need a GPU?

No. AddaxAI runs on any computer. It is faster with an NVIDIA GPU or Apple Silicon, but a GPU is not required.

### It does not use my GPU. What can I do?

On Windows and Linux you need an NVIDIA GPU and a recent driver, version 555 or newer. Update the driver from NVIDIA's site. Without a supported GPU the app falls back to the processor and still works, just slower.

### Does it work on video?

Yes. AddaxAI handles both photos and videos.

## Understanding the results

### What is the difference between a detection and an observation?

A detection is one box on one photo. An observation is one species within one event, with a count.

One deer that triggers the camera 40 times gives 40 detections but a single observation. For most ecological work you want observations. See [detections, events and observations](../understanding/detections-events-observations.mdx).

### Why is my trap night count different from what I expected?

Trap nights come from your photos, not from the dates you type in. AddaxAI counts the days between the first and last photo of a deployment, including both ends.

So a camera that died halfway through counts only up to its last photo. See [how trap nights are counted](../understanding/trap-nights.md).

### Why are labels and counts not the same thing?

Labels answer which species. Counts answer how many animals. They are checked on different pages and tracked separately, which is why "labels verified" and "counts confirmed" on the dashboard usually show different numbers.

Do labels first, then counts. See [confirm the counts](../guides/confirm-counts.mdx).

### Why does my folder run have no counts?

A folder run stops after the AI and does not interpret the results. It gives you tables of files and detections to analyse elsewhere. Counts need events, and events only exist in projects. See [choose a workflow](../start-here/choose-a-workflow.md).

### Some photos have no date. What happens to them?

They are still analysed, so you keep the detections and the species. But they drop out of anything that needs a date: events, trap nights, and the trend and activity charts.

AddaxAI never guesses a date from the file. See [dates, times and timezones](../understanding/capture-times.md).

### What does each column in the CSV mean?

Every column of all four tables is described in [export columns](../reference/export-columns.md). Start with the observations table, it is the one most analyses need.

## Using AddaxAI

### Can I correct the AI?

Yes. You can check and relabel any detection. Your edits always take priority over the AI and are never overwritten.

### Which species can it recognise?

It depends on the model. Each one covers a region or a group of animals. See the [model zoo](../reference/model-zoo.mdx).

### Why do some images have no boxes during review?

The detector runs at a low threshold and keeps everything. The review grid then hides detections below your chosen confidence, so an image can look empty even though it has a weak detection. Lower the threshold to see them.

### What is the difference between the detection models?

The detector finds animals, people and vehicles. MegaDetector is the default and works almost everywhere. See the [model zoo](../reference/model-zoo.mdx) for the full list.

## Getting help

### Something went wrong

Open the Help menu and click **Export diagnostic report**, then email it with a short description of what happened. See [troubleshooting](./index.md).

### How do I cite AddaxAI?

Citation details are on the [Addax Data Science site](https://addaxdatascience.com/addaxai/). Please cite the underlying models too: MegaDetector, SpeciesNet, and any species model you used.

### I found a bug. What should I do?

Export a diagnostic report and email it to [peter@addaxdatascience.com](mailto:peter@addaxdatascience.com) with what you were trying to do, what you expected, and what actually happened. See [troubleshooting](./index.md).

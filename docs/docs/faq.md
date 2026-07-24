---
sidebar_position: 8
title: FAQ
slug: /faq
---

# Frequently asked questions

## Privacy and data

### Does my data get uploaded anywhere?

No. Everything runs on your own computer. Your photos and videos never leave your machine. AddaxAI only goes online to download models and check for updates.

### Is it free?

Yes. AddaxAI is free and open source.

## Requirements

### Which systems does it run on?

Windows, macOS (Apple Silicon), and Linux. Intel Macs are not supported.

### Do I need a GPU?

No. AddaxAI runs on any computer. It is faster with an NVIDIA GPU or Apple Silicon, but a GPU is not required.

### It does not use my GPU. What can I do?

On Windows and Linux you need an NVIDIA GPU and a recent driver (version 555 or newer). Update the driver from NVIDIA's site. Without a supported GPU the app falls back to the processor and still works, just slower.

### Does it work on video?

Yes. AddaxAI handles both photos and videos.

## Using AddaxAI

### Can I correct the AI?

Yes. You can check and relabel any detection. Your edits always take priority over the AI and are never overwritten.

### Which species can it recognise?

Many, through a growing set of models for different regions and animal groups. See the [model zoo](./models/model-zoo.mdx).

### Why do some images have no boxes during review?

The detector runs at a low threshold and keeps everything. The review grid then hides detections below your chosen confidence, so an image can look empty even though it has a low-confidence detection. Lower the threshold to see them.

### What is the difference between the detection models?

The detector finds animals, people, and vehicles. MegaDetector is the default and works almost everywhere. See the [model zoo](./models/model-zoo.mdx) for the full list.

## Getting help

### How do I cite AddaxAI?

Citation details are on the [Addax Data Science site](https://addaxdatascience.com/addaxai/). Please cite the underlying models (MegaDetector, SpeciesNet, and any species model you used) as well.

### I found a bug. What should I do?

Open the Help menu and click **Export diagnostic report**, then email it to [peter@addaxdatascience.com](mailto:peter@addaxdatascience.com) with a short description of what went wrong. See [Troubleshooting](./troubleshooting/index.md).

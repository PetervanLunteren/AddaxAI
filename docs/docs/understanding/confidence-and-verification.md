---
sidebar_position: 3
title: Confidence and verification
---

# Confidence and verification

Every label the AI gives has a confidence score between 0 and 1. This page explains which scores are used where, and what happens when you correct something.

## Three different numbers

People often think there is one confidence setting. There are three, and they do different jobs.

### 1. What gets stored

The detector keeps everything it finds above 0.005, which is nearly everything, including very weak boxes. You cannot change this, and you do not need to. It means nothing is thrown away, so you can always lower a threshold later and see more without running the AI again.

### 2. Which boxes get a species (classification gate, default 0.1)

Only animal boxes above this score are sent to the species model, so weak boxes are left as plain "animal". Raise it to save time on a big folder, or lower it if you think real animals are being missed. This one applies while the AI runs, so changing it only affects new analyses.

### 3. What you see and what gets counted (default 0.2)

This is the one you will actually use. Detections below it are hidden from the grid, the charts and the counts. Change it any time: nothing is deleted, the app just shows more or less.

## The rule that matters

Anything you verified always counts, whatever its score. If you confirm a bobcat at 0.04 confidence, it stays in your counts and charts even though the threshold is 0.2. Your judgement beats the model's score, and this is why your numbers do not drop when you raise the threshold after checking things.

## Two kinds of checking

The app tracks two separate things, and the dashboard shows both. **Labels verified** means you looked at a detection and either accepted the species or changed it, counted per file. **Counts confirmed** means you looked at an event and signed off how many animals were there, counted per event. You can do one without the other: checking labels does not confirm counts.

## What happens when you correct a label

The app keeps both versions: what the AI said, and what you changed it to. Reprocessing and changing a setting skip verified detections, so your correction stays. One thing does wipe it, and that is re-running the AI, which deletes the results and analyses the folder again from scratch, including your corrections. The app warns you first.

Keeping both versions is also what makes the confusion matrix work. It compares your confirmed label against the label the app showed you before you touched it, using only detections you actually verified. That label is the one after cleanup, not the raw model output, so the matrix scores the whole pipeline: model plus rollup plus smoothing. See [how labels get cleaned up](./label-cleanup.md).

## When the model is unsure

Sometimes the model cannot choose between similar species but is confident about the group. In that case it can fall back to a higher rank, for example "felidae" instead of a specific cat. You will see these in the label list alongside species names. See [species names and taxonomy](./species-names.md).

## Practical advice

- Leave the threshold at the default while you check labels.
- Check the common species first. They drive most of your numbers.
- Do not chase every weak detection. Verify what you will report on.
- If you raise the threshold later, your verified detections stay.

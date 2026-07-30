---
sidebar_position: 4
title: How labels get cleaned up
---

# How labels get cleaned up

The label you see is not always what the model said. Between the model and your screen, AddaxAI cleans up the results in three steps. All three are on by default.

This page explains what they do, so a label that surprises you still makes sense.

## 1. Empty results are dropped

Some models can answer "nothing here", with classes like blank, empty or false detection. When that is all a detection got, the detection is dropped. If every detection in a file is dropped, the file is marked blank.

This keeps false triggers out of your counts. The model output stays in the file on disk, so nothing is lost for good.

## 2. Unsure species roll up to a group

When the model cannot pick one species with enough confidence, it falls back to a broader group: genus, family, order or class. So you get "felidae" instead of a coin flip between two cats.

AddaxAI adds up the confidence of the species in each group, and uses the most specific group that passes. You can still relabel these to a species yourself. See [species names](./species-names.md).

## 3. Labels are smoothed within an event

This is the step that surprises people.

Inside one event the same animal usually triggers many photos. The model can label 11 of them deer and 1 of them elk. Smoothing treats the odd one out as a mistake, and changes it to the label that dominates the event.

| Model said | Photos |
|---|---|
| deer | 11 |
| elk | 1 |
| After smoothing | 12 deer |

Smoothing works the other way too. If most of an event is confidently "lion" and one crop only made it to "felidae" in step 2, smoothing can turn that group back into the species.

This removes a lot of noise and it is usually right. But it can also hide a real second species in a busy event. So if you work with mixed herds, or with two similar species at the same site, check those events yourself.

Smoothing needs events, so it works per event. See [detections, events and observations](./detections-events-observations.mdx).

## What this means for you

Two things follow from the three steps above.

**Your own corrections are never touched.** Smoothing and rollup skip anything you verified. Your label always wins.

**The AI's score is measured after cleanup.** The confusion matrix compares your confirmed label against the label the app showed you, which is the cleaned-up one, not the raw model output. See [confidence and verification](./confidence-and-verification.md).

## Changing it

All three steps are project settings. You can make smoothing milder or stronger, or turn it off, and you can turn rollup off. See [settings](../reference/settings.mdx).

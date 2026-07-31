---
sidebar_position: 4
title: How labels get cleaned up
---

# How labels get cleaned up

The label you see is not always what the model said. Between the model and your screen, AddaxAI cleans up the results in three steps. This page explains what they do, so a label that surprises you still makes sense.

## 1. Empty results are dropped

Some models can answer "nothing here", with classes like blank, empty or false detection. When that is the model's top answer, the detection is dropped. So a box that would have read "Animal 68%, false detection 80%" never reaches your results at all. If every detection in a file is dropped, the file is marked blank.

## 2. Taxonomic rollup

When the model cannot name one species with enough confidence, it labels the animal with a broader group instead: genus, family, order or class.

It works by adding up. A model that is unsure between two deer species might give each one 40%, which is weak on its own. Added together the genus reaches 80%, which clears the 65% that AddaxAI asks for. It walks up the tree like this and stops at the most specific level that clears the bar, so you get "deer" rather than a coin flip between two species. If nothing clears it, even at class level, the label stays as the model's best guess.

That 65% is fixed. It is the same for every model and every project, and you cannot change it.

It is a trade-off: a broader label tells you less, but it is more likely to be right. You can still relabel it to a species yourself if you can tell them apart. Rollup is on by default and you can turn it off in the project settings. See [species names](./species-names.md).

## 3. Labels are smoothed

It runs twice: first within a single image, across the boxes in it, then across all the files in an event. Inside one event the same animal usually triggers many photos. If the model labels 11 of them deer and 1 of them elk, smoothing treats the odd one out as a mistake and makes all 12 deer.

Smoothing works the other way too. If most of an event is confidently "lion" and one crop only made it to "felidae" in step 2, smoothing can turn that group back into the species. Vague labels give way easily: "animal" or "unknown" needs far less support to be overwritten than a real species does. This removes a lot of noise and it is usually right, but it can also hide a real second species in a busy event. So if you work with mixed herds, or with two similar species at the same site, check those events yourself. See [detections, events and observations](./detections-events-observations.mdx) for how events are formed.

## What this means for you

Two things follow from the three steps above. Your own corrections are never touched, because smoothing and rollup skip anything you verified, so your label always wins. And the AI's score is measured after cleanup: the confusion matrix compares your confirmed label against the label the app showed you, which is the cleaned-up one, not the raw model output. See [confidence and verification](./confidence-and-verification.md).

## Changing it

Steps 2 and 3 are project settings. You can turn rollup off, and you can make smoothing milder or stronger or turn it off. Step 1 always runs. See [settings](../reference/settings.mdx).

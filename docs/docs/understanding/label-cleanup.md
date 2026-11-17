---
sidebar_position: 5
title: How labels get cleaned up
---

# How labels get cleaned up

The label you see is not always what the model said. Between the model and your screen, AddaxAI cleans up the results in a few steps. This page explains what they do, so a label that surprises you still makes sense.

## 1. Empty results are set aside, not thrown away

Some models can answer "nothing here", with labels like blank, empty or false detection. When that is the model's top answer, the box is kept with that label, but it is not counted: it draws no box on the photo, it is not an observation on the Counts page, it is in no chart, and a file with nothing else on it is marked blank. So a box that reads "Animal 68%, false detection 80%" is a box the detector found and the species model rejected.

The model can be wrong about that, so the box is not hidden from you. It is a card on the Labels page like any other, with the label the model gave it. Press <kbd>Enter</kbd> to agree, or relabel it if there is an animal after all, and it counts from then on. The detections table and the recognition file carry these boxes too, with the model's label, so nothing the AI found is lost.

## 2. One species per animal in a video

The AI follows each animal through a video and gives it one card. The card does not take the species from one frame: it takes what all the animal's frames agree on, weighted by how clearly each frame saw it. A sharp frame counts fully, a blurry frame where the animal is half out of the picture barely counts. So one odd frame cannot name the animal, and the card says what the video as a whole shows. Each frame keeps its own label underneath, which is what lets you see when the AI switched animals halfway. See [check the labels](../guides/check-labels.mdx).

## 3. Taxonomic rollup

When the model cannot name one species with enough confidence, it labels the animal with a broader group instead: genus, family, order or class.

It works by adding up. A model that is unsure between two deer species might give each one 40%, which is weak on its own. Added together the genus reaches 80%, which clears the 65% that AddaxAI asks for. It walks up the tree like this and stops at the most specific level that clears the bar, so you get "deer" rather than a coin flip between two species.

If nothing clears it, not even at class level, there is one last step. Everything the model recognised as some kind of animal is added up, and if that total clears the bar the label becomes the kingdom: "Animal" in common names, "Animalia" in scientific names. That says only that an animal was there. If even the kingdom does not clear it, the label stays as the model's best guess.

That 65% is fixed. It is the same for every model and every project, and you cannot change it.

It is a trade-off: a broader label tells you less, but it is more likely to be right. You can still relabel it to a species yourself if you can tell them apart. Rollup is on by default and you can turn it off in the project settings. See [species names](./species-names.md).

## 4. Your own taxa list (SpeciesNet only)

Instead of a country, you can give SpeciesNet a taxa list: a .csv that maps SpeciesNet's taxa to the labels you use. Pick "Custom taxa list (.csv)" in the country dropdown. It is the file MegaDetector's `restrict_to_taxa_list` reads, and AddaxAI runs it the same way. The file needs two columns: `input_latin`, a SpeciesNet taxon (`aepyceros`, `accipitridae`), and `user_taxon_name`, the label you want for it ("impala", "bird of prey"). Several rows can share a label. Other columns are ignored.

Each prediction walks up the tree from the species until it reaches a taxon in your list, and gets that row's label. So with `aepyceros` mapped to "impala", every impala prediction reads "impala", and with `accipitridae` and `strigiformes` both mapped to "bird of prey", hawks, eagles and owls all read "bird of prey". A prediction can also stop at a group above your taxa: with only birds of prey in the list, a stork reads "bird". A prediction with nothing from your list above it reads "animal". People are never changed.

A file loads completely or not at all: if a row names a taxon SpeciesNet does not know, or lists one twice, you get the list of bad rows and nothing changes. The list replaces the country and your species selection, so there is no "Edit species" with a list loaded. In the label filter and the exports, a label gets the lowest taxon its rows share: "impala" on the genus is a genus, "bird of prey" is the class Aves.

## 5. Labels are smoothed

It runs twice: first within a single photo, across the boxes in it, then across all the files in an event. Inside one event the same animal usually triggers many photos. If the model labels 11 of them deer and 1 of them elk, smoothing treats the odd one out as a mistake and makes all 12 deer.

Smoothing works the other way too. If most of an event is confidently "lion" and one detection only made it to "felidae" in step 3, smoothing can turn that group back into the species. Vague labels give way easily: "animal" or "unknown" needs far less support to be overwritten than a real species does. This removes a lot of noise and it is usually right, but it can also hide a real second species in a busy event. So if you work with mixed herds, or with two similar species at the same site, check those events yourself. See [detections, events and observations](./detections-events-observations.mdx) for how events are formed.

## What this means for you

Two things follow from the steps above. Your own corrections are never touched, because smoothing and rollup skip anything you verified, so your label always wins. And the AI's score is measured after cleanup: the confusion matrix compares your verified label against the label the app showed you, which is the cleaned-up one, not the raw model output. See [confidence and verification](./confidence-and-verification.md).

## Changing it

Steps 3 to 5 are project settings. You can turn rollup off, load or remove a taxa list, and make smoothing milder or stronger or turn it off. Steps 1 and 2 always run.

Rollup also decides what happens to species you left out of your selection. With rollup on, a photo the model reads as a species you excluded is moved up to the nearest group you did allow, with the model's full confidence. With rollup off, the excluded species are simply dropped and the next best species from your selection becomes the label, at the score the model gave it. If that score is below 1%, the animal stays unlabelled instead.

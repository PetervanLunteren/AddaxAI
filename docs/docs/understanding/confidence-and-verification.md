---
sidebar_position: 4
title: Confidence and verification
---

# Confidence and verification

The AI gives two scores between 0 and 1. The detector says how sure it is that a box holds an animal, and the species model how sure it is about the species. In this photo, the box is 90% animal and 98% coyote.

<img src="/img/two-confidences.webp" alt="A coyote at night, with the box labelled Animal 90% and Coyote 98%" style={{maxWidth: '620px', width: '100%', display: 'block'}} />

<p style={{fontSize: '0.75rem', color: 'var(--ifm-color-emphasis-600)', marginTop: '-0.5rem'}}>
Photo: <a href="https://lila.science/datasets/ena24detection">ENA24-detection dataset</a> (CDLA-Permissive).
</p>

## Two settings

Both settings work on the detector's score.

| Setting | What it does | Default | Takes effect |
|---|---|---|---|
| Classify detections above | Only boxes above it get a species. Weaker ones stay "animal" | 0.1 | On the next analysis |
| Count detections above | Detections below it are hidden from the grid, the charts and the counts | 0.5 | Right away |

Raise Classify detections above to save time on a big folder, or lower it if real animals are missed. Leave Count detections above at the default while you check labels. Lowering it later shows more, without running the AI again.

## Species confidence

Low species confidence doesn't hide a detection. It changes the label to a broader group, for example "felidae" instead of a specific cat. You can turn this off in the project settings. See [how labels get cleaned up](./label-cleanup.md).

Under More filters on the Labels page, two sliders filter on the detector and the species score. They only change what the grid shows.

## Your corrections win

Anything you verified always counts, whatever its score. A bobcat you verified at 0.04 stays in your counts and charts, even with the threshold at 0.5.

Reprocessing and changing a setting skip what you verified, so your corrections stay. Re-running the AI is the exception: it analyses the folder again from scratch and deletes your corrections. The app warns you first.

The app keeps both the AI's label and yours. The [confusion matrix](./insights.mdx#confusion-matrix-and-class-performance) uses them to score the AI against your verified labels.

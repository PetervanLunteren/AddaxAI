---
sidebar_position: 6
title: Species names and taxonomy
---

# Species names and taxonomy

## Two names for the same animal

Every species has a common name and a scientific name. "Virginia opossum" and *Didelphis virginiana* are the same animal. You can switch which one the app shows, and the choice is yours alone: it changes the display, not the data. Both names go into the exports, in separate columns. Use scientific names if you will publish, or if you work across languages. Common names are easier to scan while checking photos.

## Labels are grouped into a tree

Species are organised the standard way, from broad to specific:

```
class > order > family > genus > species
```

So a red fox sits under mammalia > carnivora > canidae > vulpes > vulpes vulpes. The filter on the Labels and Insights pages uses this tree, so picking "carnivora" selects every carnivore at once, instead of ticking twenty species by hand.

## When you see a group instead of a species

Sometimes a label is a family or an order rather than a species, for example "felidae" or "aves". That means the model was not sure enough about any single species, so it gave you the group instead. You can relabel these to a species yourself if you can tell them apart. See [how labels get cleaned up](./label-cleanup.md).

## When you see "animal"

The box was found but never given a species. Two reasons:

- the project has no species model, only a detector
- the detection scored below the classification gate, so it was never sent to the species model

These still count as animals in the totals. They just have no species. See [confidence and verification](./confidence-and-verification.md).

## Which species a model can recognise

Every model has its own fixed list, so a European model does not know African species. Check the [model zoo](../reference/model-zoo.mdx) for what each one covers. Some models also use the country you set for the project, to rule out species that do not occur there, so set the country correctly or you may lose valid species.

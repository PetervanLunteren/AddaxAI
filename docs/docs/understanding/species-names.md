---
sidebar_position: 5
title: Species names and taxonomy
---

# Species names and taxonomy

## Two names for the same animal

Every species has a common name and a scientific name. "Virginia opossum" and *Didelphis virginiana* are the same animal.

You can switch which one the app shows. The choice is yours alone: it changes the display, not the data. Both names go into the exports, in separate columns.

Use scientific names if you will publish, or if you work across languages. Common names are easier to scan while checking photos.

## Labels are grouped into a tree

Species are organised the standard way, from broad to specific:

```
class > order > family > genus > species
```

So a red fox sits under mammalia > carnivora > canidae > vulpes > vulpes vulpes.

The filter on the Labels and Insights pages uses this tree. Picking "carnivora" selects every carnivore at once, instead of ticking twenty species by hand.

## When you see a group instead of a species

Sometimes a label is a family or an order rather than a species, for example "felidae" or "aves".

This happens when the model is not confident enough about any single species, but the candidates all sit in the same group. Rather than guess, the app gives you the group. A confident "felidae" is more useful than a coin flip between two cats.

You can relabel these to a species yourself if you can tell them apart.

## When you see "animal"

The box was found but never given a species. Two reasons:

- the project has no species model, only a detector
- the detection scored below the classification gate, so it was never sent to the species model

These still count as animals in the totals. They just have no species. See [confidence and verification](./confidence-and-verification.md).

## Blank photos

Some classifications mean "nothing here", such as blank or empty. When the model returns only these, the detection is dropped and the file is marked blank.

This keeps false triggers out of your counts. The original result stays in the file on disk, so nothing is lost for good.

## Which species a model can recognise

Every model has its own fixed list. A European model does not know African species. Check the [model zoo](../reference/model-zoo.mdx) for what each one covers.

Some models also use the country you set for the project, to rule out species that do not occur there. Set the country correctly or you may lose valid species.

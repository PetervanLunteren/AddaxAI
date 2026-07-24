---
sidebar_position: 2
title: How trap nights are counted
---

# How trap nights are counted

A trap night is one camera working for one day. It is how you turn raw counts into a rate you can compare between sites.

This page explains exactly how AddaxAI works the number out, because it surprises people.

## The short version

**Trap nights come from your photos, not from the dates you type in.**

For each deployment, AddaxAI takes the first and last photo, and counts the days between them, including both ends:

```
trap nights = (date of last photo) - (date of first photo) + 1
```

A deployment whose first photo is 1 September and last photo is 16 September counts as 16 trap nights.

## Why photos and not the dates you enter

The app can only be sure a camera was working when it produced a file. A start date in a form is a plan. A photo is proof.

This also means a deployment with no photos counts as zero trap nights, even if you entered dates for it.

## What this means in practice

Two cases catch people out.

**Your camera ran longer than its photos suggest.** You put the camera out on 1 September and collected it on 30 September, but the last photo is from 16 September because the battery died. AddaxAI counts 16 trap nights, not 30. That is the honest number: the camera was not working after the 16th, so it should not count as effort.

**Your camera took photos on only a few days.** First photo 1 September, last photo 16 September, but nothing in between. It still counts 16 trap nights. The camera was out and working that whole window, it just saw nothing. Empty days are effort too.

## Cameras pulled more than once

If a deployment folder has subfolders, AddaxAI treats each subfolder as a separate stretch and adds them up. This matches how people work: one subfolder per card swap.

So a deployment with two subfolders, one covering 10 days and one covering 6, counts as 16 trap nights.

## Rates per 100 trap nights

Most charts show a rate, not a raw count:

```
rate = observations / trap nights x 100
```

This lets you compare a camera that was out for two weeks with one that was out for two months. A site with 40 observations over 20 trap nights (200 per 100 trap nights) is busier than a site with 60 observations over 60 trap nights (100 per 100 trap nights), even though the second has more observations.

## Where to check the number

The Deployments page shows the period and the file count for every deployment. The Deployment timeline shows the same thing as bars, and how many cameras ran at once.

If a number looks wrong, check the first and last photo of that deployment. Photos with no date do not count at all. See [capture times](./capture-times.md).

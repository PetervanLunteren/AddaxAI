---
sidebar_position: 1
title: Overview
---

# Insights

Insights are the in-depth analytical views inside a project. They go deeper than the glanceable Dashboard, and each one is a scientifically grounded visualisation with its own filter bar. 

## Available views

### Map

Per-deployment observation rate per 100 trap nights, plotted on a base map. Three spatial views: trap-night-normalised rate, absolute observation count, and a heat-style density layer.

### Activity overlap

A 1- or 2-species temporal activity comparison. Pick one species to see its daily activity pattern, or two to compare them. With two species selected, the page also shows the overlap coefficient with a bootstrap confidence interval, and classifies each species as diurnal, nocturnal, crepuscular, or cathemeral.

### Deployment timeline

When each camera was active and what it recorded, over time.

### Confusion matrix and per-class performance

Where verifications exist, these views compare the model's labels against your confirmed labels, so you can see where the model is strong and where it needs a human eye.

:::note

Insights need capture times and, for the map, site coordinates. Files with no readable timestamp are still detected and classified, but they drop out of time-based views.

:::

---
sidebar_position: 5
title: Dates, times and timezones
---

# Dates, times and timezones

Camera trap analysis lives or dies on time. This page explains where AddaxAI gets the time, and what it does when the time is missing.

## Where the time comes from

From the file itself. For photos it reads the capture time written by the camera. For videos it reads the equivalent field.

The time is stored exactly as the camera recorded it. Nothing is converted.

## You always see camera time

The app shows the time the camera wrote, not the time on your computer.

This matters if you analyse data from another country. A photo taken at 23:00 in Kenya shows as 23:00, even if you are sitting in the Netherlands. If it were converted to your local time, every night photo would move into the afternoon and your activity charts would be wrong.

## What the project timezone is for

A project has a timezone setting. It is used only to work out when the sun rises and sets at your site, for the day and night bands on the activity charts.

It never changes the times shown or stored. Set it to match how the cameras were configured.

If your cameras stay on winter time all year, choose a fixed offset zone rather than a country, so the app does not apply summer time.

## Photos with no date

Some files have no readable capture time. Cameras get reset, files get edited, some formats lose it.

AddaxAI does not guess. It does not use the file modification date, because that is usually the date you copied the files, not the date the animal walked past.

A file with no time is still analysed. You keep the detections and the species, and they still count in your species totals, because the file becomes an event of its own.

But it drops out of anything that needs a date:

- it does not count towards trap nights
- it does not appear in the trend or activity charts
- it is never grouped with other files, even ones taken at the same moment

The app tells you how many files this affected after an analysis. If the number is large, fix the camera clock settings and analyse again.

## Wrong camera clocks

If a camera had the wrong date set, the photos carry that wrong date. AddaxAI stores what the camera said.

You can correct it with a time shift, either when you add the deployment or later by editing it. Editing it later is safe: AddaxAI shifts the stored times and updates the dates, events and charts. You do not have to analyse again.

## Quick checks

- Compare the time on a photo with what the app shows. They should match exactly.
- Look at the activity chart. Real animals have a pattern. A flat line across all 24 hours usually means a broken clock somewhere.
- Check the deployment period on the Deployments page against your field notes.

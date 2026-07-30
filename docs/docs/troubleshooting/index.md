---
sidebar_position: 1
title: Troubleshooting
---

# Troubleshooting

## Export a diagnostic report

If something goes wrong, the fastest way to get help is a diagnostic report. Open the Help menu and click **Export diagnostic report**. It builds a zip of logs and saves it to your Downloads folder. Attach that to an email to [peter@addaxdatascience.com](mailto:peter@addaxdatascience.com), along with what you were trying to do, what you expected, and what actually happened. A screenshot or short screen recording helps a lot.

## If the app won't open

Zip the logs folder by hand and email it.

| OS | Logs folder |
|---|---|
| macOS | `/Users/<username>/AddaxAI/logs/` |
| Windows | `C:\Users\<username>\AddaxAI\logs\` |
| Linux | `/home/<username>/AddaxAI/logs/` |

Nothing is ever uploaded automatically. Sharing logs is always an explicit action you take.

## Common questions

### Some images show no date or time

A file with no readable capture timestamp is still detected and classified, but it drops out of time-based views (trap nights, activity, trend charts) and is never grouped with other files. Fix the source data and re-run if you need the timestamps.

### My GPU isn't being used

On Windows and Linux you need an NVIDIA GPU and a driver version 555 or newer. Update the driver from NVIDIA's site. Without a supported GPU the app falls back to the CPU and still works, just slower.

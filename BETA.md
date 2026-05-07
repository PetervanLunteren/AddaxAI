# AddaxAI beta

Thanks for trying the new AddaxAI. It is a full rewrite from scratch, with stronger metadata analysis and verification options. It works as an additive system: keep adding deployments to a project and the analyses, dashboards, and insights update with them.

## Download

| OS | Download |
|---|---|
| Windows | [AddaxAI-Setup.exe](https://github.com/PetervanLunteren/AddaxAI-WebUI/releases/latest/download/AddaxAI-Setup.exe) |
| macOS (Apple Silicon) | [AddaxAI-arm64.dmg](https://github.com/PetervanLunteren/AddaxAI-WebUI/releases/latest/download/AddaxAI-arm64.dmg) |

Linux is still in development, so the beta will not run there yet. Intel Macs are not supported, only Apple Silicon (M1, M2, M3, M4, M5).

## Install

#### macOS
Open the dmg, drag AddaxAI into Applications.

#### Windows
Run the setup `.exe` and follow the installer.

## Timelapse integration

Windows only, because Timelapse itself is also Windows only. Install AddaxAI first using the steps above, then open the AddaxAI-Timelapse integration window:

- From AddaxAI: hamburger menu in home page > `Timelapse integration`.
- From Timelapse: choose AddaxAI from the menu `Recognitions` > `AddaxAI Image Recognizer` > `Run AddaxAI recognizer on a folder...`.

When the run finishes, AddaxAI writes `timelapse_recognition_file.json` next to the chosen folder. In Timelapse, go to Recognition > Import recognition data for this image set and pick that file.

## What I would like to learn from you

I want to hear about anything that feels wrong, weird, or could be better. You do not need a polished bug report. One sentence already helps.

- bugs and crashes
- buttons or controls in places that feel weird
- text or labels that read strange (typos, weird English, unclear wording)
- features, plots, or filters that are missing
- workflow steps that are confusing or take too many clicks
- anything else you notice

If you are unsure whether it is a bug, send it anyway, I would rather see one too many than miss something.

## How to send me a bug report

Send everything to [peter@addaxdatascience.com](mailto:peter@addaxdatascience.com). Please give me as much detail as you can so I can reproduce it. Most useful:

- what you were trying to do
- what you expected to happen
- what actually happened
- a screenshot, or a short screen recording
- a diagnostics report (see below)

#### Export diagnostic report

Click the bug icon in the top-right page header (tooltip says "Export diagnostic report"). It builds a zip of logs, attach that to the email. If the app won't open, zip the logs folder by hand:

| OS | Logs folder |
|---|---|
| macOS | `/Users/<username>/AddaxAI/logs/` |
| Windows | `C:\Users\<username>\AddaxAI\logs\` |

Also share these if they exist next to `logs/`:

- `crash-dumps/` (zip the full folder)
- `timelapse-runs/<timestamp>__<job_id>/` (Timelapse bugs only, share every `.json` file inside the run subfolder, skip the `video_frames/` subfolder, it can be many GB)

## What is not built yet

Here is what is missing on purpose. None of these are forgotten, they are just not in this beta yet:

- full-image classification models support (AHDRIFT-v1 and similar)
- multi-language support
- depth estimation 
- postprocess batch results from MegaDetector
- proper documentation
- repeat detection elimination
- Wildbook integration
- the full model zoo (only a subset is shipped for now)

## Uninstall

#### macOS
Drag AddaxAI from Applications to the Trash. To also remove logs, models, and the database, delete the folder at `/Users/<username>/AddaxAI`.

#### Windows
Open Windows Settings, then Apps, find AddaxAI, click Uninstall. The uninstaller asks whether to also remove your user data.

## Contact

[peter@addaxdatascience.com](mailto:peter@addaxdatascience.com)

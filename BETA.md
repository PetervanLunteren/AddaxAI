# AddaxAI beta

This is the new AddaxAI, a full rewrite from scratch. It has stronger metadata analysis and verification options, and works as an additive system: keep adding deployments to a project and the analyses, dashboards, and insights update with them.

## Download

| OS | Download |
|---|---|
| Windows | [AddaxAI-Setup.exe](https://github.com/PetervanLunteren/AddaxAI-WebUI/releases/latest/download/AddaxAI-Setup.exe) |
| macOS (Apple Silicon) | [AddaxAI-arm64.dmg](https://github.com/PetervanLunteren/AddaxAI-WebUI/releases/latest/download/AddaxAI-arm64.dmg) |

Linux is still in development, so the beta will not run there yet. Intel Macs are not supported, only Apple Silicon (M1, M2, M3, M4, M5).

## Install

**macOS**: open the dmg, drag AddaxAI into Applications. First launch may take a few seconds while Gatekeeper checks the Apple notarisation.

**Windows**: run the setup `.exe` and follow the installer. If SmartScreen still warns, click "More info" then "Run anyway", the installer is signed.

## What I want from you

Anything that is wrong, off, or could be better:

- bugs and crashes
- buttons or controls in places that feel weird
- text or labels that read strange (typos, weird English, unclear wording)
- features, plots, or filters that are missing
- workflow steps that are confusing or take too many clicks
- anything else you notice

Even small notes help. If you are not sure whether it is a bug, send it anyway.

## How to send me a bug report

Email everything to [peter@addaxdatascience.com](mailto:peter@addaxdatascience.com).

Please give me as much detail as you can so I can reproduce it. Most useful:

- what you were trying to do
- what you expected to happen
- what actually happened
- a screenshot, or a short screen recording if the bug is dynamic

**If the app opens**: click the bug icon in the page header (tooltip says "Export diagnostic report"), or open Settings, then Diagnostics, and click "Export diagnostic report". Both build the same zip in your Downloads folder with logs, system info, installed models, environment state, and recent jobs. Please attach it to the email.

**If the app does not open**: zip the logs folder by hand and attach that.

| OS | Logs folder |
|---|---|
| macOS | `~/AddaxAI/logs/` |
| Windows | `%USERPROFILE%\AddaxAI\logs\` |

To zip on macOS: open Finder, press `Cmd+Shift+G`, paste the path, right-click the folder, Compress.
To zip on Windows: open File Explorer, paste the path, right-click the folder, Send to, Compressed folder.

Native crash dumps live at `~/AddaxAI/crash-dumps/` (macOS) or `%USERPROFILE%\AddaxAI\crash-dumps\` (Windows). Include those if present.

## What is not built yet

These are on the roadmap but not in this beta:

- timelapse integration (will be a standalone app)
- full-image classification models support (AHDRIFT-v1 and similar)
- multi-language support
- depth estimation 
- postprocess batch results from MegaDetector
- proper documentation
- repeat detection elimination
- Wildbook integration
- the full model zoo (only a subset is shipped for now)

<details>
<summary><strong>Reset (open if something is stuck)</strong></summary>

Inside the app: open Settings, find Reset, follow the prompts. The app quits, next launch starts from scratch.

If the app does not open at all and you cannot reach Settings, delete the user data folder by hand:

- macOS: `rm -rf ~/AddaxAI`
- Windows: delete `%USERPROFILE%\AddaxAI` in File Explorer (or `rmdir /s /q %USERPROFILE%\AddaxAI` in cmd)

Then reopen the app. The setup wizard runs again.

</details>

## Contact

[peter@addaxdatascience.com](mailto:peter@addaxdatascience.com)

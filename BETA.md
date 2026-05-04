# AddaxAI beta

This is the new AddaxAI, a full rewrite of the legacy desktop app. It is still in active development. Please help me test it.

## Download

The latest installers are here: [github.com/PetervanLunteren/AddaxAI-WebUI/releases/latest](https://github.com/PetervanLunteren/AddaxAI-WebUI/releases/latest)

| OS | File |
|---|---|
| Windows | `AddaxAI Setup *.exe` |
| macOS (Apple Silicon) | `AddaxAI-*-arm64.dmg` |

Linux is not packaged yet. Intel Macs are not packaged either, only Apple Silicon (M1, M2, M3, M4).

## Install

**macOS**: open the dmg, drag AddaxAI into Applications. First launch may take a few seconds while Gatekeeper checks the Apple notarisation.

**Windows**: run the setup `.exe`. Windows SmartScreen will probably say "Windows protected your PC". Click "More info", then "Run anyway".

A short note on the SmartScreen warning: the installer is signed and the signature is valid. But Microsoft has a separate reputation system on top of the signature. A new code-signing certificate starts with zero reputation, so SmartScreen warns until many people have installed the app without problems. The warning will go away on its own over time. It is annoying, not a security issue.

## First launch

The app downloads around 1.9 GB on first launch (Python environment and default ML models). It needs about 7 GB of free disk space. This takes 10 to 30 minutes depending on your internet.

## What I want from you

Anything that is wrong, off, or could be better:

- bugs and crashes
- buttons or controls in places that feel weird
- text or labels that read strange (typos, off English, unclear wording)
- features, plots, or filters that are missing
- workflow steps that are confusing or take too many clicks
- anything else you notice

Even small notes help. If you are not sure whether it is a bug, send it anyway.

## How to send me a bug report

Email everything to [peter@addaxdatascience.com](mailto:peter@addaxdatascience.com). I will reply when there is a fix and tell you which version to download next.

**If the app opens**: go to Settings, then Diagnostics, click "Export diagnostic report". This builds a zip with logs, system info, installed models, environment state, and recent jobs. Save it to Downloads, attach it to the email.

**If the app does not open**: zip the logs folder by hand and attach that. Add a screenshot or a short description of what you saw.

| OS | Logs folder |
|---|---|
| macOS | `~/AddaxAI/logs/` |
| Windows | `%USERPROFILE%\AddaxAI\logs\` |

To zip on macOS: open Finder, press `Cmd+Shift+G`, paste the path, right-click the folder, Compress.
To zip on Windows: open File Explorer, paste the path, right-click the folder, Send to, Compressed folder.

Native crash dumps live at `~/AddaxAI/crash-dumps/` (macOS) or `%USERPROFILE%\AddaxAI\crash-dumps\` (Windows). Include those if present.

## What is not built yet

These are on the roadmap but not in this beta:

- timelapse standalone app
- full-image classification models (AHDRIFT-v1 and similar)
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

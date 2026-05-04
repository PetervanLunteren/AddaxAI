# AddaxAI beta

Thanks for trying the new AddaxAI. It is a full rewrite from scratch, with stronger metadata analysis and verification options. It works as an additive system: keep adding deployments to a project and the analyses, dashboards, and insights update with them. Things are still rough in places, that is what this beta is for.

## Download

| OS | Download |
|---|---|
| Windows | [AddaxAI-Setup.exe](https://github.com/PetervanLunteren/AddaxAI-WebUI/releases/latest/download/AddaxAI-Setup.exe) |
| macOS (Apple Silicon) | [AddaxAI-arm64.dmg](https://github.com/PetervanLunteren/AddaxAI-WebUI/releases/latest/download/AddaxAI-arm64.dmg) |

Linux is still in development, so the beta will not run there yet. Intel Macs are not supported, only Apple Silicon (M1, M2, M3, M4, M5).

## Install

#### macOS
Open the dmg, drag AddaxAI into Applications. First launch may take a few seconds while Gatekeeper checks the Apple notarisation.

#### Windows
Run the setup `.exe` and follow the installer. If SmartScreen still warns, click "More info" then "Run anyway", the installer is signed.

## What I would like to learn from you

I want to hear about anything that feels wrong, weird, or could be better. You do not need a polished bug report. One sentence already helps.

- bugs and crashes
- buttons or controls in places that feel weird
- text or labels that read strange (typos, weird English, unclear wording)
- features, plots, or filters that are missing
- workflow steps that are confusing or take too many clicks
- anything else you notice

Even one line is useful. If you are unsure whether it is a bug, send it anyway, I would rather see one too many than miss something.

## How to send me a bug report

Email is old tech but it works. Send everything to [peter@addaxdatascience.com](mailto:peter@addaxdatascience.com). I read every message and reply when there is a fix, and I will tell you which version to download next.

Please give me as much detail as you can so I can reproduce it. Most useful:

- what you were trying to do
- what you expected to happen
- what actually happened
- a screenshot, or a short screen recording
- a diagnostics report (see below)

#### Export diagnostic report

Click the bug icon in the page header (tooltip says "Export diagnostic report"). This builds a zip file with logs, system info, states, jobs, etc. Please attach it to the email. If the app does not open the button is not available. In that case, please zip the logs folder by hand and attach that (right-click the folder > compress).

| OS | Logs folder |
|---|---|
| macOS | `/Users/<username>/AddaxAI/logs/` |
| Windows | `C:\Users\<username>\AddaxAI\logs\` |

Please also zip and share the `\crash-dumps` folder if present.

## What is not built yet

Here is what is missing on purpose. None of these are forgotten, they are just not in this beta yet:

- timelapse integration (will be a standalone app)
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
Open Settings, then Apps, find AddaxAI, click Uninstall. To also remove logs, models, and the database, delete the folder at `C:\Users\<username>\AddaxAI`.

## Contact

[peter@addaxdatascience.com](mailto:peter@addaxdatascience.com)

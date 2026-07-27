# AddaxAI beta

Thanks for trying the new AddaxAI. It is a full rewrite from scratch, with stronger metadata analysis and verification options.

There are two ways to work:

- **Analyse a folder**: a one-off run in three steps (setup, labels, save). Run the AI, clean up labels if you want, and get files out: complete detection tables, a recognition file for Timelapse, species-separated folders, visualised or blurred images. No ecological interpretation, you do your own analysis with the files.
- **Build a project**: a stored workspace for tracking cameras over time. Confirm species counts, keep verification history, watch dashboards and maps, and export to Camtrap DP. It works as an additive system: keep adding deployments and the analyses update with them. A folder run can be turned into a project after saving.

## Download

| OS | Download |
|---|---|
| Windows | [AddaxAI-Setup.exe](https://github.com/PetervanLunteren/AddaxAI-WebUI/releases/latest/download/AddaxAI-Setup.exe) |
| macOS (Apple Silicon) | [AddaxAI-arm64.dmg](https://github.com/PetervanLunteren/AddaxAI-WebUI/releases/latest/download/AddaxAI-arm64.dmg) |
| Linux | [AddaxAI-amd64.deb](https://github.com/PetervanLunteren/AddaxAI-WebUI/releases/latest/download/AddaxAI-amd64.deb) |

Intel Macs are not supported, only Apple Silicon (M1, M2, M3, M4, M5).

## Install

#### macOS
Open the dmg, drag AddaxAI into Applications.

#### Windows
Run the setup `.exe` and follow the installer.

#### Linux
Double-click the downloaded `.deb`. The App Center opens and shows it as `addaxai` from an unknown publisher with a "potentially unsafe" warning. That is how Ubuntu presents every app installed outside its own store. It is expected, click Install. After installing, find AddaxAI in the app grid to open the app.

## Removing the old AddaxAI

The new AddaxAI installs in a different place than the old one, so after installing you have two separate AddaxAI programs on your computer. The old one takes up a lot of space (often 10 GB or more) and you do not need it any more.

The first time you open the new app it checks for the old version and offers to remove it. Your images, videos and any results the old version wrote next to them are never touched. Skipping is fine, nothing breaks either way, and you can run it later from the Help menu under "Remove old AddaxAI".

If you want the old version back at any point, install it again from the AddaxAI website.

## Updating to a new beta version

Your projects, verifications, and settings live in the AddaxAI user data folder and survive updates. Just install the new version over the old one:

- **macOS**: open the new dmg, drag AddaxAI into Applications, click Replace.
- **Windows**: run the new setup `.exe`, it replaces the old version.
- **Linux**: the App Center shows a greyed-out "Installed" button for the new file and offers no update, that is an Ubuntu limitation for third-party packages. Update via terminal instead: `sudo apt install ./AddaxAI-amd64.deb` from the download folder.

## GPU acceleration

On Apple Silicon the GPU is always used and there is nothing to install. On Windows and Linux with an NVIDIA GPU, AddaxAI uses it automatically. The models now ship with CUDA 12.8 builds, so the newest cards (RTX 50-series) run at full speed instead of a slow fallback. This needs a reasonably recent NVIDIA driver, version 555 or newer; if yours is older, update it from NVIDIA's site. Without a supported GPU the app still runs, just on the CPU.

## Timelapse integration

Windows only, because Timelapse itself is also Windows only. Install AddaxAI first using the steps above. In Timelapse, choose AddaxAI from the menu `Recognitions` > `AddaxAI Image Recognizer` > `Run AddaxAI recognizer on a folder...`. AddaxAI opens "Analyse a folder" with that folder filled in; run it and keep the recognition file enabled in the save step.

When the run finishes, AddaxAI writes `addaxai-recognitions.json` into the analysed folder itself, next to your images. In Timelapse, go to Recognition > Import recognition data for this image set and pick that file.

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

Open the Help menu and click Export diagnostic report. It builds a zip of logs and saves it to your Downloads folder, attach that to the email. If the app won't open, zip the logs folder by hand:

| OS | Logs folder |
|---|---|
| macOS | `/Users/<username>/AddaxAI/logs/` |
| Windows | `C:\Users\<username>\AddaxAI\logs\` |
| Linux | `/home/<username>/AddaxAI/logs/` |

## What is not built yet

Here is what is missing on purpose. None of these are forgotten, they are just not in this beta yet:

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

#### Linux
Run `sudo apt remove addaxai`. To also remove logs, models, and the database, delete the folder at `/home/<username>/AddaxAI`.

## Contact

[peter@addaxdatascience.com](mailto:peter@addaxdatascience.com)

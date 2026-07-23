---
sidebar_position: 1
title: Installation
---

# Installation

AddaxAI is a desktop app. Download the installer for your operating system and run it. Intel Macs are not supported, only Apple Silicon (M1, M2, M3, M4, M5).

| OS | Download |
|---|---|
| Windows | [AddaxAI-Setup.exe](https://github.com/PetervanLunteren/AddaxAI-WebUI/releases/latest/download/AddaxAI-Setup.exe) |
| macOS (Apple Silicon) | [AddaxAI-arm64.dmg](https://github.com/PetervanLunteren/AddaxAI-WebUI/releases/latest/download/AddaxAI-arm64.dmg) |
| Linux | [AddaxAI-amd64.deb](https://github.com/PetervanLunteren/AddaxAI-WebUI/releases/latest/download/AddaxAI-amd64.deb) |

## Install

### macOS

Open the dmg, drag AddaxAI into Applications.

### Windows

Run the setup `.exe` and follow the installer.

### Linux

Double-click the downloaded `.deb`. The App Center opens and shows it as `addaxai` from an unknown publisher with a "potentially unsafe" warning. That is how Ubuntu presents every app installed outside its own store. It is expected, click Install. After installing, find AddaxAI in the app grid to open the app.

## GPU acceleration

On Apple Silicon the GPU is always used and there is nothing to install. On Windows and Linux with an NVIDIA GPU, AddaxAI uses it automatically. The models ship with CUDA 12.8 builds, so the newest cards (RTX 50-series) run at full speed. This needs a reasonably recent NVIDIA driver, version 555 or newer. Without a supported GPU the app still runs, just on the CPU.

## Updating

Your projects, verifications, and settings live in the AddaxAI user data folder and survive updates. Install the new version over the old one:

- **macOS**: open the new dmg, drag AddaxAI into Applications, click Replace.
- **Windows**: run the new setup `.exe`, it replaces the old version.
- **Linux**: update via terminal with `sudo apt install ./AddaxAI-amd64.deb` from the download folder.

## Uninstall

- **macOS**: drag AddaxAI from Applications to the Trash. To also remove logs, models, and the database, delete `/Users/<username>/AddaxAI`.
- **Windows**: Settings → Apps → AddaxAI → Uninstall. The uninstaller asks whether to also remove your user data.
- **Linux**: `sudo apt remove addaxai`. To also remove user data, delete `/home/<username>/AddaxAI`.

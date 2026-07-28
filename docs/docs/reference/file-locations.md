---
sidebar_position: 4
title: Where your files live
---

# Where your files live

AddaxAI keeps three things in three places. You need all three to move your work to another computer.

## 1. Your photos and videos

Wherever you put them. AddaxAI never moves, changes or deletes your originals. It only reads them.

The app stores the path to each file. If you move or rename a folder, the app can no longer find those files and the deployment shows as offline. Point it at the new location to fix it.

## 2. Results next to your photos

Inside each analysed folder AddaxAI writes a hidden `.addaxai` folder. It holds the raw model output and, for videos, the single best frame per clip.

Keep it. It lets the app redo work after you change a setting, without running the AI again. Delete it and that shortcut is gone.

## 3. The database

One file holding every detection, label, correction and confirmed count.

| System | Location |
|---|---|
| macOS and Linux | `~/AddaxAI/` |
| Windows | `%USERPROFILE%\AddaxAI\` |

Inside that folder:

| Item | What it is |
|---|---|
| `addaxai.db` | The database. All your work |
| `backups/` | Automatic copies of the database |
| `logs/` | Log files, useful for bug reports |
| `models/` | Downloaded model files |
| `envs/` | Software the models need |

## Backups

The app copies the database into `backups/` once a day when it starts, and again before any update that changes the database. You can also make one yourself from the menu.

Only the database is backed up. Your photos are not, they are far too large. Back those up the way you normally would.

## Moving to another computer

You need the photos, the `.addaxai` folders, and the database.

The photos and the `.addaxai` folders travel together, so an external drive handles both. The database is separate: back it up, copy the file across, then restore it on the new machine.

If the drive letter or the path changes, the app will report the deployments as offline. Point it at the new location.

## What is safe to delete

| Item | Safe to delete |
|---|---|
| `logs/` | Yes |
| `backups/` | Yes, but you lose your safety net |
| `.addaxai` folders | Yes, but reprocessing then needs a full re-run |
| `models/` | Yes, they download again when needed |
| `addaxai.db` | No. This is your work |

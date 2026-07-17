---
name: run-addaxai
description: Run, launch, screenshot, or smoke-test the AddaxAI desktop GUI from a dev checkout without the platform installer; also drive/test the Gundi upload feature against the stage API. Use when asked to run the app, verify a UI change, or test Gundi/EarthRanger uploads.
---

# Run AddaxAI (dev checkout, no installer)

AddaxAI is a tkinter/customtkinter desktop GUI (`AddaxAI_GUI.py`, single file).
All paths below are relative to the repo root. The GUI is driven by
`.claude/skills/run-addaxai/driver.py` — an in-process probe (tkinter has no
external automation surface), which launches the real app, switches it to
advanced mode, clicks the Gundi checkbox, verifies the options frame appears,
and saves a window screenshot. ML **deployment** (running detection models) is
NOT covered here — it needs the platform installer's conda envs. Everything
else (full UI, post-processing, Gundi upload) works from a plain venv.

## Prerequisites (one-time)

The app expects sibling repos and dirs NEXT TO the repo checkout
(`AddaxAI_files` = parent directory of the repo):

```bash
cd ..   # parent of the AddaxAI repo
mkdir -p envs models/det/"MegaDetector 5a" models/cls
git clone --depth 1 https://github.com/PetervanLunteren/visualise_detection.git visualise_detection
git clone --depth 1 https://github.com/agentmorris/MegaDetector.git cameratraps
cd AddaxAI
```

Python deps (a `uv`-created `.venv` ships without pip — bootstrap it first if
`python -m pip` fails: `.venv/bin/python -m ensurepip --upgrade`):

```bash
.venv/bin/python -m pip install requests Pillow piexif GPSPhoto exifread \
  opencv-python folium numpy pandas customtkinter seaborn tqdm plotly \
  CTkTable RangeSlider matplotlib pytest jsonpickle
```

(`pytest` and `jsonpickle` look wrong but are imported at module level by the
cloned MegaDetector repo — the GUI won't start without them.)

## Run (agent path) — the driver

```bash
.venv/bin/python .claude/skills/run-addaxai/driver.py
```

Prints `OK`/`FAIL` per probe; exit 0 = app launched and the Gundi UI wiring
works. Saves `screenshot-addaxai.png` next to the driver (gitignored). Add
`KEEP_OPEN=1` to leave the window open afterwards. Extend `probe()` to drive
other widgets — find labels by text, widgets by grid row (see the helpers).

## Run (human path)

```bash
ADDAXAI_GUNDI_ENV=stage .venv/bin/python AddaxAI_GUI.py
```

Window opens; Ctrl-C in the terminal (or close the window) to quit. ALWAYS set
`ADDAXAI_GUNDI_ENV=stage` when testing Gundi uploads — the default endpoint is
**production**, and unknown values exit with an error by design.

## Testing the Gundi upload end-to-end in the UI

Generate a GPS-tagged image + pre-baked MegaDetector recognition JSON (skips
the ML deployment step entirely):

```bash
.venv/bin/python .claude/skills/run-addaxai/setup_fixtures.py
mkdir -p ../gundi-test-images
cp .claude/skills/run-addaxai/test_data/* ../gundi-test-images/
```

Then in the app (launched with `ADDAXAI_GUNDI_ENV=stage`): advanced mode →
step 1 select `../gundi-test-images` → step 4 tick "Upload events to Gundi" →
enter a stage API key → Start post-processing. Expect a success dialog and the
event (with photo) in EarthRanger.

Regenerate fixtures before EVERY re-run — EarthRanger discards events whose
data is identical to a previous one; `setup_fixtures.py` stamps a fresh
timestamp + jittered GPS each time.

## Direct invocation (no GUI)

- `test_gundi_upload.py` — standalone replica of the upload payload/POST logic
  fired at the stage API: `GUNDI_API_KEY=... .venv/bin/python .claude/skills/run-addaxai/test_gundi_upload.py`
  (or `GUNDI_DRY_RUN=1` to just print payloads). Run `setup_fixtures.py` first.
- `test_retry_logic.py` — 9-case mock test of the event-POST retry loop's
  error accounting: `.venv/bin/python .claude/skills/run-addaxai/test_retry_logic.py`

## Gotchas

- The app writes its settings into the repo's own tracked `global_vars.json`
  on every run (dev checkout = runtime dir). Don't commit those runtime edits.
- The Gundi API key is persisted to `../gundi-api-key.txt` (outside the repo).
- Step-4 widgets start **disabled** until a source folder is selected, and
  `toggle_gundi_frame` additionally guards on the *label's* state — a driver
  must enable both the checkbox and its label before `invoke()` works.
- The app starts in **simple mode**; the Gundi controls live in the advanced
  pane. `winfo_ismapped()` is useless for widgets in a hidden pane — check
  `winfo_manager()` instead.
- customtkinter's CTk root misreports `winfo_width/height` (stays 200x200);
  the driver pads the screenshot region to compensate.
- Gundi's API returns HTTP **200** (not 201) on success for both events and
  attachments.
- Attachments don't appear in Gundi's Activity Log — check the destination
  (EarthRanger) to confirm an image landed.

## Troubleshooting

- `ModuleNotFoundError: pytest` (from `megadetector/utils/path_utils`) or
  `jsonpickle` → `.venv/bin/python -m pip install pytest jsonpickle`
- `screencapture ... could not create image from display` when run from a
  plain terminal → grant the terminal Screen Recording permission (System
  Settings → Privacy & Security). The driver's in-app capture usually works
  without it.
- App exits immediately with `Unknown ADDAXAI_GUNDI_ENV value` → you typo'd
  the env var; valid values are `prod` and `stage` (this is intentional —
  typos must not silently hit production).

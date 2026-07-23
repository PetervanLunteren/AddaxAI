# Send a draft model to a tester

Use this when you have a draft classification model (built and validated, but not
published: not in `models.json`, not on the public HuggingFace org) and want a
beta tester or the model author to try it in their installed AddaxAI. The
recipient is usually non-technical, so a real part of this job is producing
dead-simple, copy-pasteable instructions for them.

This skill only packages and delivers a model. Build and validate it first with
the `add-classification-model` skill.

## Why sideloading works (the mechanism)

- The app builds its model picker by scanning
  `~/AddaxAI/models/{det,cls,emb}/*/manifest.json` (`ManifestManager`), so it
  lists whatever model dirs are present locally, not only the remote catalog.
- Catalog sync never overwrites or deletes existing local model dirs, so a
  sideloaded model survives future syncs.
- So a complete model dir dropped into `~/AddaxAI/models/cls/<id>/` shows up in
  the picker and runs, while never being in the public catalog or on HF.

## Prerequisite that silently bites: the env

The app can only build the envs whose yamls are bundled in the recipient's app
version. Sideloading cannot add a new env. So confirm the recipient's AddaxAI
version provides the env this model needs before sending. A model that needs a
new env (for example DINOv3 needs the py3.11 `pytorch` env) requires them to be
on the release that ships it; otherwise it fails on their machine with no obvious
cause.

## Steps (the LLM does these)

1. Work from a COPY of the validated dir, so the real one is untouched:
   `cp -r ~/AddaxAI/models/cls/<id> /tmp/<id>-draft`.
2. Make the copy a draft:
   - rename the folder to `<id>-draft`.
   - in its `manifest.json`: `model_id` to `<id>-draft`, `friendly_name` to
     `"... (DRAFT)"`.
   - `min_app_version`: set it to the version that first ships the env this model
     needs (the same value the eventual public entry will use). It only warns, it
     does not block, so it is an honest safety-belt: a compatible tester sees
     "you're good to go", an older one sees a version note.
3. Confirm the dir is COMPLETE. This is the number one safety item: if any file
   is missing, the app marks the model "not ready" and tries to download it from
   a HF repo that does not exist, which fails confusingly. Required:
   - the weights file (`manifest.model_fname`)
   - `inference.py`
   - `taxonomy.csv`
   - `manifest.json` (the app needs it to list the model)
   - vendored arch source (`hubconf.py` + the package dir) for torch.hub models
   - `LICENSE`
   Delete `__pycache__` and `.DS_Store` first.
4. Zip it so the archive holds the `<id>-draft` folder at its root:
   `cd /tmp && zip -r <id>-draft.zip <id>-draft -x '*/__pycache__/*' '*.DS_Store'`.
5. Upload the zip to a file-transfer service (WeTransfer, Google Drive) and get a
   link. Weights run from 100s of MB to over 1 GB, past email limits.
6. Write the recipient message (next section).

## The recipient message (the LLM writes this, in plain language)

Give them: the link, where to put the folder for their OS, how to find that
folder, how to restart, which model to pick, and what to send back. Keep only the
OS line that applies to them if you know it.

> Hi [name], here is a test model to try in AddaxAI.
>
> 1. Download and unzip this: [link]. You get a folder called `<id>-draft`.
> 2. Put that whole folder inside your AddaxAI models folder:
>    - macOS: `/Users/<you>/AddaxAI/models/cls/` (in Finder press Cmd+Shift+G, paste `~/AddaxAI/models/cls`, press Enter)
>    - Windows: `C:\Users\<you>\AddaxAI\models\cls\` (in a File Explorer window, click the address bar, paste `%USERPROFILE%\AddaxAI\models\cls`, press Enter)
>    - Linux: `/home/<you>/AddaxAI/models/cls/`
> 3. Fully quit AddaxAI and open it again.
> 4. Start an analysis and, in the model list, pick "[friendly_name] (DRAFT)". The
>    first run may take a few minutes while it sets things up.
> 5. Send me a screenshot of the results, or of any error message. Thank you.

## Gotchas

- **Completeness is everything** (step 3): a missing file surfaces as a failed HF
  download, not an obvious "file missing" error.
- **The env must already exist in their app version** (see the prerequisite);
  sideloading cannot add one.
- **Use the `-draft` id and "(DRAFT)" name** so it cannot collide with, or be
  mistaken for, the published model.
- **It persists**: catalog syncs will not remove it. Tell them to delete the
  `<id>-draft` folder when done, or when the real model ships.
- **They run your code**: `inference.py` executes in the env subprocess. Fine for
  a model you send directly; do not forward third-party drafts blindly.
- **Restart, not just a new window**: the picker is scanned at app startup.

## When it graduates to public

Rename `<id>-draft` back to the real `<id>`, drop the "(DRAFT)", set the real
`min_app_version`, and follow `add-classification-model` (HF upload plus a
`models.json` entry). Ask the tester to delete their `<id>-draft` folder so they
pick up the published model instead.

## Key files

- `backend/app/ml/manifest_manager.py` (scans local dirs to build the picker)
- `backend/app/ml/catalog_updater.py` (never overwrites or prunes local dirs)
- `backend/app/ml/model_storage.py` (`check_weights_ready`, the "not ready" path)
- `skills/add-classification-model/SKILL.md` (build and validate the model first)

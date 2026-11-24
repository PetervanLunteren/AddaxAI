# Fast with a large database

Overarching plan, written 2026-11-20 after an investigation of the code, the query plans and how other tools handle the same problem. Phase 0, phase A and most of phase B are built; "Status and next steps" below says where the work stands. Each phase gets its own detailed plan in its own session; this file is the map they start from, so nobody has to redo the investigation.

The short version: the app is slow on a large database mainly because of its own habits, not because of the size. It opens a new database connection for every request, runs heavy queries where they block every other request, and does full-database work at every launch. Those are cheap to fix with standard practice (phase A). A second group is query plans that start from the whole database where they should start from the project (phase B). Deleting data does not make a project's own pages faster, it saves disk space and backup time and shortens the project list, so a clean-up dialog and an archive option are not part of this plan. They are in `TODO.md` under "Nice to haves" (CLEAN UP DATABASE, ARCHIVE A PROJECT), with their design.

## Status and next steps (2026-11-24)

**Start here in a new session.** Phase 0, phase A and the phase B fixes below are on `main`. Nothing of it is released yet; phase A and B go out together.

Built in phase B, details under "Measured facts (2026-11-22, phase B baseline)":

- A verify refreshes only what it changed, and no longer scans every box (2 to 7 s per click down to under 0.1 s).
- `labels/stats`, the label-tree count and the Verification card start from the project and count in fewer passes.
- The dashboard card and the Counts pill ask only for the two event numbers they show; the card fills in part by part.
- The Labels sort starts without waiting for the embedding count.
- The Files tab's "Sort by event" looks each file's event up once.
- The label lists: one request for what exists (`filter-options`), one for the counts (`label-counts`). The filter dialog opens at once and says "counting…" until the counts are in.
- Three duplicate indexes dropped (migration `b9c3d5e7f9a1`).

### Next, in this order

1. **Final measurement round.** Done 2026-11-24, see "Measured facts (2026-11-24, final round)". It also found the sort worker on the Elliot seed slower and far more variable than on 11-22, cause not found; decide under step 3 whether that holds up the release.
2. **The project list.** For every project it loads every file's capture time into Python to count trap nights (`crud/project.py`, `get_trap_nights` per project; `crud/trap_nights.py`, `compute_intervals_for_deployments`). 4.5 s on the Simon seed, 7 s on the Elliot seed, and Simon has 36 projects on a slow machine. Look at the fix first; do it only if it is simple and keeps the trap-night rule. Not yet investigated.
3. **Decide from step 1, simple fixes only:** the dashboard cards of a big project (overview, animal photos, progress per label, the Labels bar; 15 to 25 s each, every one a pass over all boxes) and the info sheets of a big deployment or site (22 s and 10 s on "Large"; the two confidence averages can be one query).
4. **The "not yet tested" pass, before any release:**
   - **The first start after the update on a big database.** The release carries a migration, so the first start makes a pre-upgrade backup, migrates and runs a full `ANALYZE` before the app opens. On the dev Mac 19.6 s for the 4.6 GB Simon seed and 6.7 s for the 3.4 GB Elliot seed (final round); on Simon's machine (6 to 12 times slower per row, 3.6 GB) an estimated one to a few minutes. Check what the user sees meanwhile, and say it in the release notes.
   - **The folder-run save step.** Its label picker was rewired for the label lists (`useMediaOptions.ts`, `SaveShared.tsx`) and only the typecheck covers it. Click through one real folder run.
   - **A project without embeddings.** The Labels sort start changed for it (`LabelsTab.tsx`, `emptyCauseUnknown`); never tried live.
   - **Windows.** Everything was measured on the Mac. One upgrade run of the packaged build with a large database on the Windows machine (`/windows-linux-test-partner`); it covers the first point too.
5. **Release** with `/release-partner`. The notes say plainly what got faster and that the first start after the update takes longer once.
6. **Drafts to Simon and Elliot** (drafts only, Peter sends): try the new version, say what still feels slow. For Simon also: is the AddaxAI folder on an internal disk or an external drive?

### Known and left as it is

- One pass over all boxes of a 2.5M-box project still takes 10 to 15 s on a fast disk. Making a pass itself fast needs a project column on `detections`, rejected below as too large.
- A count the browser has abandoned keeps running on the backend. How often that happens in normal use is not measured.
- Files tab with a species filter: 21 to 45 s per page on the Elliot seed in the final round (about 7 s on 11-23), the count and the page about half each. The species filter is the cost: an `EXISTS` over the boxes of every file.
- The Files tab remembers its sort per browser, so one visit with "Sort by event" keeps a large project on the slower sort.
- No server-side cache of the label lists (see the note under the phase B measured facts).
- `SortSelector.tsx` says the random seed "lives in URL state"; on the Files tab it does not.

### After this release

- **Folder runs** (under "Later" below): a new run for the same folder replaces the hidden duplicate. Small, its own plan.
- **Clean up database** and **Archive a project** moved to `TODO.md` under "Nice to haves" on 2026-11-24, with their design. They save disk space and backup time, not page speed.

### Measuring in a new session

The golden databases stay on the external SSD: `/Volumes/2TB-SSD/addaxai-speed/golden` (Simon shape) and `golden-elliot`. Everything else lived in the old session's scratchpad and is gone: rebuild the 3.11 venv and the `sqlite345` link as in "How we measure" (a few minutes), and write the SQL tracer again if statement timings are needed (described there). Only one database copy fits on the internal disk at a time.

Three things learned in the final round:

- **The golden copies are at migration `a8b2c4d6e8f0`**, so the first start on every copy migrates, backs up and runs `ANALYZE`. Restart once before measuring.
- **Chrome freezes the tab between Claude's tool calls.** A streamed response, the sort, is then not read, so a sort timed in the browser can take minutes. Time the sort worker from a script that posts the page's body to `labels/sort` and reads the stream to its `result` line. In dev mode `await import('/src/lib/query-client.ts')` gives the page's own query client, and `isFetching()` tells when a page has settled.
- **Delete `backups/.last-rolling-utc-date` between seeds**, or the second seed skips its daily backup.

## Why

Reported by Simon Kravis (2026-10-04, v7.2.1), two symptoms:

1. A 923K-photo project (4 GB database) takes about 30 s to open and the dashboard Verification card hangs on "Loading...".
2. A small 27K-photo project in the same database is slow on the Labels page with a species filter. Crops read from his USB drive are a separate cost, bound by the drive, and not something this plan can fix.

Elliot Johnston has the same class of problem (1.2M-photo project, https://forum.addaxai.com/t/removing-blank-image-originals-from-project/362). Tell both when a phase ships.

## How we work on this

- **Bang for buck, not a fixed target.** The aim is roughly "opens in 5 s, pages of a small project in 2 s" on the synthetic database, but that number does not decide the work. A fix goes in when it is simple, robust and clearly worth it. A fix that buys a second at the cost of a maintenance burden stays out. KISS, DRY, YAGNI.
- **Measure before and after.** No timing exists today; every cost below is read from the code. Each fix is measured on the synthetic database, and anything that turns out not to matter is left alone.
- **Established solutions only.** Every fix below follows SQLite's or FastAPI's own documentation, or what comparable apps do. Sources are at the bottom.
- **Two releases.** Phase A ships first (low risk, helps everyone), phase B after it is checked against Simon's real statistics.

## Every phase starts with a plan

Phase 0, phase A, phase B and each later feature is a large change. Before any code is touched, the session switches to plan mode and writes a detailed plan together with Peter, following the instructions below word for word. This file says what to build and why; the detailed plan says how.

> ## plan
>
> Instructions:
> * Switch to plan mode, I want this task to be done with "plan mode on"
> * Codex will review your output once you are done, so make sure you exceed his expectations
> * do not sugar coat, be honest and clear
> * Read all MD file in root to get a understanding of the project.
> * If something is unclear at any point, stop and ask before continuing.
> * Prioritize simplicity and clarity over perfection. The code must be clean, easy to read, and understandable for collaborators. Avoid unnecessary complexity.
> * I'm not in a rush. Please be precise and do the task thoroughly.
> * Follow the KISS principle (Keep It Simple Stupid). Keep things as simple as possible.
> * Follow the DRY principle (Don't Repeat Yourself). Avoid duplication and maintain a single source of truth.
> * Follow the YAGNI principle (You Aren't Gonna Need It). Do not build functionality until it is actually needed.
> * Before changing code, explain the current architecture and the intended change in your own words.
> * Make the smallest safe change that solves the problem.
> * Prefer simple, boring, maintainable code over clever abstractions.
> * You can do web queries if you wish.
> * A minimalist set that caters for all is a good design.
> * Do not introduce new dependencies unless there is a clear benefit.
> * Preserve existing behaviour unless explicitly asked to change it.
> * Update or add tests for the changed behaviour.
> * Please do webqueries if you wish to clarify things. I'm not in a rush and have enough tokens.
> * I do not want complex heuristics designed to handle every possible edge case. Prefer a simple, consistent rule that works well across all situations.
> * Run the relevant checks, tests, linters, and type checks before finishing (backend only).
> * Document any important trade-offs, risks, or follow-up work.
> * If you find unrelated issues, mention them separately, but do not fix them unless asked.
> * Use a clear mental model that makes the feature easy for users to understand. Good UX should feel simple.
> * make sure you test it properly and only stop when everything passes. I want you to test it e2e. You have the crome claude extention.
> * Please ask me any question for clarification. I would rather that you ask too many questions than assume certain details.
> * KISS, DRY, YAGNI. Be critical and honest. Prefer simple, robust solutions. Be thorough and don't cut corners. A minimal set that caters for all is good design. Research established solutions and industry best practices when needed. Learn from proven implementations. Don't reinvent the wheel. Separate concerns. Follow coding best practices.
> * at the top of the plan file show a ballpark figure of the proposed (estimated) Line changes (separated by code/comments/docs/tests), the net line changes (+ / - (net)) and the number of files.
> * At the top of the plan file below the ballpark figure, it should have a plain English summary of maximum a few sentences with n easy to follow plain English words and sentences that describes the problem and how to solve it.
> * Ask me clarifying questions before beginning. Based on the conventions set out in the MD files in root and your knowledge, give your recommended solution to each questions you ask me. The minimum number of questions (in interactive mode with selectable predefined multiple choice answers, the first always being your recommended answer) to ask me is 4

## Phase 0: measure

- **Seed script** in `backend/scripts/`, the generated database goes in the scratchpad, never in the repo. Simon's shape: about 1M image files, one huge deployment of about 900K next to a small 27K project and a few normal ones, detections stored down to 0.01, embeddings for the gated ones. Built through the normal models so the schema is always the current one.
- **Slow-request log.** One middleware that logs every request over 1 s with its path and duration. It stays in the product: it is what makes the next report from a user readable.
- **Check on SQLite 3.45.** The packaged backend carries 3.45.1 (inside CPython 3.11). The dev venv has 3.53 and the worker env 3.54, and plans differ between them. Timings and `EXPLAIN QUERY PLAN` that decide anything are taken on 3.45.
- **Simon's real numbers.** Sent 2026-11-20: `backend/scripts/addaxai-db-stats.py` (see "Numbers from a user's database" in `DEVELOPERS.md`). He sent his `addaxai-db-stats.txt` on 2026-11-24, after most of phase B; see "Simon's database (2026-11-24)" under the measured facts. The seed was kept as it is: his real shape is milder than it. Elliot got the same script and sent his on 2026-11-21 (see the addition under phase B).

**Built (2026-11-20):** the seed script (`backend/scripts/seed_large_image_database.py`), the slow-request log plus a duration line for every startup step and every `ANALYZE` (see "Logging & Debugging" in `DEVELOPERS.md`), and the 3.45 runtime described under "How we measure". The baseline is under "Measured facts".

## How we measure

Every phase measures this way, before and after, so the numbers compare.

1. **Runtime: SQLite 3.45.1, the build users have.**
   - A throwaway venv from a Python 3.11 (`python3.11 -m venv $SCRATCH/venv311`, then `pip install -r backend/requirements.txt`).
   - A folder `$SCRATCH/sqlite345` with a symlink to the installed app's `Contents/Resources/backend/_internal/python3.11/lib-dynload/_sqlite3.cpython-311-darwin.so`.
   - `PYTHONPATH=$SCRATCH/sqlite345` puts that module ahead of the venv's own.
   - Check first: `python -c "import sqlite3; assert sqlite3.sqlite_version == '3.45.1'"`.
   - The installed app cannot run the seeded database itself once `main` has a newer migration than the release.
2. **Data.**
   - Seed once onto the external SSD with the 3.45 venv: `ADDAXAI_USER_DATA_DIR=/Volumes/2TB-SSD/addaxai-speed/golden`, `--classification-model SPECIESNET-v4-0-2-A`, and `--taxonomy-csv` set to that model's `taxonomy.csv`.
   - For every measurement, check that at least 8 GB is free on the internal disk.
   - Copy `addaxai.db` into `$SCRATCH/run/`.
   - Symlink `models`, `envs` and `bin` from `~/AddaxAI` into it. The sort worker needs `env-addaxai-base`, and the setup check must pass.
   - Delete `$SCRATCH/run/` afterwards, backups included.
3. **Run.**
   - Backend, from `backend/`: `ADDAXAI_USER_DATA_DIR=$SCRATCH/run ADDAXAI_API_PORT=8123 ADDAXAI_DISABLE_MODEL_UPDATES=1 ADDAXAI_SLOW_REQUEST_SECONDS=0 PYTHONPATH=$SCRATCH/sqlite345 $SCRATCH/venv311/bin/python -m uvicorn app.main:app --port 8123`.
   - Front end, from `frontend/` on Node 20: `ADDAXAI_BACKEND_PORT=8123 npm run dev`. The dev backend on 8000 is never touched.
4. **Pages, in Chrome.** Load each page cold once (the first load after a fresh backend start) and warm once:
   1. the project list
   2. the dashboard of "Large"
   3. the dashboard of "Small"
   4. the Labels page of "Small" with one species filter
   5. the deployment info and site info of "Small"
   6. the similarity sort on that Labels page
   7. on the Labels page of "Small" and of the Elliot seed: one verify in the grid (click a card, Enter), one in the detail view (double-click a card, Enter), and Next on the Files tab. Mark the log line count before each click, and read every request after it until the log goes quiet.
5. **Read off.**
   - From the log: the startup lines, the background-task lines, and the `Slow request:` lines for each page.
   - The browser's Performance API is no use here: Vite's dev modules fill its 250-entry buffer before the page asks anything. Read the times from the log.
   - The sort is a worker process: its request returns at once, so time it from the `Streaming labels subprocess` line to the first crop request.
   - A species filter goes in the URL as `?lbl_labels=<taxonomy id>`, so it can be reloaded without clicking.
   - One table per phase, under "Measured facts".
   - For the cause of a slow request, time its statements. A `sitecustomize.py` on `PYTHONPATH` that wraps `sqlite3.connect` (and `sqlite3.dbapi2.connect`, which SQLAlchemy calls) with a cursor that times execute plus fetches gives one line per statement; `EXPLAIN QUERY PLAN` on the 3.45 runtime then gives its plan. Kept out of the repo on purpose.
   - Seed the Elliot shape with `--shape elliot` into its own folder. Only one copy fits on the internal disk next to the 8 GB that must stay free, so measure one shape at a time, and point `$SCRATCH/run/backups` at the SSD so the daily backup does not fill the disk.

Absolute numbers come from a Mac with an internal NVMe disk, not from a user's laptop. The comparison between phases is what counts.

## Phase A: stop the work that hurts (release 1)

Ordered by expected gain. Decisions already taken are marked **Decided**.

1. **One engine per process.** `get_db` calls `get_session_factory()`, which calls `get_engine()`, which runs `create_engine` every time (`backend/app/db/base.py:73-103`, `:150-164`). Every request opens a new SQLite connection, sets the pragmas again, parses the schema again and starts with an empty 64 MB page cache. Same in the startup tasks, `_seed_builtin_labels`, the job reconcile and `websocket.py`. Fix: one module-level engine and session factory. Check what the tests and `ADDAXAI_USER_DATA_DIR` overrides need.
2. **No database work on the event loop.** **Decided: all async endpoints that touch the database, plus a test that fails when a new one is added.** Heavy ones that cause the hangs: `labels/progress` (`routers/labels.py:320`), `labels/files` (`:209`), `events/filter-options` (`routers/events.py:156`), `deployments/{id}/info` (`routers/deployments.py:917`), `sites/{id}/info` (`routers/sites.py:205`). About 20 light ones, including the eleven event edits (`events.py:558-723`), which can freeze the loop for the 30 s busy timeout when another write holds the lock. The detailed plan lists them all again from a grep of `async def` in `api/routers/`; `statistics.py`, `detections.py`, `tracks.py`, `backup.py`, `calibration.py`, `logs.py` and `undo.py` have none.
   - An endpoint that returns no observational datetimes becomes plain `def` (`labels/progress`, and `deployment info` / `site info`, which never set the timezone and have no serializer for it, whatever their docstrings say).
   - An endpoint that needs the timezone stays `async def`, sets the ContextVar on the loop, then runs the work through `asyncio.to_thread` or `run_in_threadpool`. Both copy the context into the thread; a `set` inside the thread does not come back. The pattern is already used in `files.py:183`, `events.py:263` and `export.py:105`.
   - Not database, same bug: `update_model_catalog` does a synchronous `urlopen` on the loop (`ml/catalog_updater.py:144`), and `GET /api/ml/models/{id}/status` runs `subprocess.run(timeout=10)` on the loop (`routers/ml_models.py:214`).
3. **Startup does only what it must before `/health`.**
   - **Daily backup. Decided: in a background thread right after start, always, with its duration logged.** Today `ring_buffer_backup` runs synchronously in the lifespan (`main.py:477`). Same once-per-day rule, same five kept. The copy uses almost no CPU, and writes keep working during it. Known cost, accepted: on a hard disk it competes with image reading for a minute or more, and the WAL cannot be checkpointed while it runs. The pre-upgrade backup stays where it is, before the migration.
   - **ANALYZE. Decided: measure first.** `refresh_query_statistics` runs a full `ANALYZE` at every launch (`base.py:224`) and after every analysis, embedding, postprocessing and import job. If the measured time is small, leave it. If not, the candidates are: skip it at launch unless a migration ran or statistics are missing, and set `PRAGMA analysis_limit` so it samples. Whatever happens, it keeps going through `refresh_query_statistics` (see "Never run a bare ANALYZE" in `DEVELOPERS.md`).
4. **Remove the warm-up scans. Decided.** `_warm_up_query_caches` (`main.py:141-222`) runs about 18 whole-database counts into a private cache that is thrown away. With one engine the cache stays warm after the first page load.
5. **Card pictures only when missing. Decided.** `auto_generate_thumbnails` (`main.py:63-88`) re-picks a picture for every project at every launch, with a query per project, an image read and a commit. A project gets a picture once and keeps it.
6. **Legacy video-frame cleanup runs once.** `_reclaim_legacy_video_frames` (`main.py:225-316`) loads every video row at every launch through a full table scan. Detailed plan decides between a done-marker and removing it.

**Built (2026-11-21):** all six, plus a seventh found while planning. Decisions taken in the detailed plan:

- **A2 has no exceptions.** Every endpoint that takes a database session is a plain `def`. The project timezone no longer forces `async def`: a middleware puts a fresh holder object in the ContextVar per request and the endpoint's thread writes into it (`start_request_timezone`). `ws_manager.register_start` hops onto the loop by itself. `tests/api/test_no_db_on_event_loop.py` fails on any new `async def` endpoint with a session. See "Endpoints and the event loop" in `DEVELOPERS.md`.
- **A3 ANALYZE:** skipped at launch unless a migration ran or `sqlite_stat1` is empty. The bad `idx_files_source_video` row is still deleted at every launch. Snapshots are written under a `.partial` name and renamed when complete, because the daily one now runs while the app is in use.
- **A6:** removed, not marked done.
- **A7, jobs do not freeze the app:** jobs run on the same event loop as the requests. Event generation and the closing `ANALYZE` in the analysis, embedding and postprocessing workers now run in a thread.
- **Found while measuring A1:** SQLite reads the statistics once per connection. With a pool, a connection would keep old statistics forever, and the one that ran `ANALYZE` keeps the deleted `idx_files_source_video` row in memory (the 6.5-hour project delete). `refresh_query_statistics` now replaces every pooled connection.

The numbers are under "Measured facts (2026-11-21, after phase A)".

## Phase B: queries that start from the project (release 2)

**Decided scope: project-first plans, worker cache, duplicate indexes.** Keyset paging and counter tables wait until a measurement asks for them.

1. **Sort worker, project first.** There is no project column on `detections`, `files` or `detection_embeddings`; the project filter arrives only through `deployments` (`ml/inference/similarity_script.py:136-144`). `sqlite_stat1` holds one average per index for the whole database, so one huge deployment shapes the plan for every small project. Confirmed with `EXPLAIN QUERY PLAN` on a backup copy: with a label filter the plan starts from `idx_detections_label_taxonomy` (every detection with that label in the database), without one it scans all of `detection_embeddings` including the vector blobs. Fix: fixed join order with `CROSS JOIN` (deployments, files, detections), which SQLite documents for exactly this and advises to keep to the few queries that need it.
   **Corrected 2026-11-22: not as a fixed rule.** Project first is wrong for a big project with a rare species: on the seed, "Large" with a 14.8K-box species took 5 to 6 s starting from the species and 66 to 102 s forced project first. "Small" with the same filter went the other way (6 s against 4 s, 15 s against 4 s for a common species). Without a species filter the sort worker's queries already start from the project. The packaged SQLite has no `STAT4`, so it cannot see the skew itself. The choice for species-filtered queries is taken after the stage 3 measurements; see "Measured facts (2026-11-22, phase B baseline)".
2. **Count folded into the load.** `_load_metadata` and `_load_embeddings` run `SELECT COUNT(*) FROM (sql)` and then the same query again (`similarity_script.py:369-379`, `:410-422`). One pass with `count(*) OVER ()`.
3. **Worker connections.** Each helper opens its own read-only connection with SQLite's 2 MB default cache, up to four per run (`:366`, `:403`, `:457`, `:476`). One connection per run, with `cache_size` and `temp_store` set.
4. **Duplicate indexes.** `idx_files_deployment` (prefix of `idx_files_deployment_path`), `idx_detection_embeddings_detection` (prefix of the unique index) and `idx_event_files_event` (prefix of the primary key). Each costs space and insert time and buys nothing. Check each against the foreign-key cascade rule in `DEVELOPERS.md` before dropping.
5. **Statistics, if plans still flip.** Idea to test, not verified: make `idx_files_source_video` a partial index (`WHERE source_video_id IS NOT NULL`). SQLite can use it for `= ?` lookups, and it may remove the reason the statistic has to be deleted after every `ANALYZE`. The documented fallback is shipping fixed `sqlite_stat1` rows.
6. **Other project-scoped queries.** The thumbnail pick (`ORDER BY confidence DESC LIMIT 10`) may walk the confidence index across the whole database; after A5 it runs rarely. Check the dashboard and Labels queries on the synthetic database and fix only what shows up in the slow-request log.

**Added 2026-11-22, after Elliot's numbers.** The goal of phase B stays the goal of this whole plan: the app is slow on anything with a large database, so make everything as fast as it can be within the scope, with simple and robust fixes (KISS, DRY). The two cases below are not the only pain points. They are the two that users have named, so each gets a test project in the seed and its own lines in the measured table, and they decide the order of the work.

- **A small project next to a big one** (Simon's complaint). The query does more work than the project needs: it starts from the whole database because one huge deployment shapes the statistics. Items 1 to 6 above are aimed at this, and item 1's project-first join order applies to every project query that shows the problem, not only the sort worker (`label-tree` and `labels/stats` of "Small" took 64 s and 93 s in the baseline, 24 s and 34 s after phase A).
- **One big project** (Elliot's complaint: 1.27M photos and 2.57M boxes in one project, 58 deployments of 2K to 35K photos, 20K events). The query is right, but counting the project's own rows takes long however it is planned. His pain is a lag after every click on the Labels page: verifying, scrolling through blanks, changing a filter. Two steps, in this order:
  1. **Fewer counts per click.** Measure first what one verify and one filter change on the Labels page request. If a click refetches the project-wide totals (`labels/progress`, `labels/stats`, `label-tree`), refetch only what the click changed, or once the user stops clicking. Probably the biggest gain for this case, and mostly front-end.
  2. **Cheaper counts.** Merge counts that read the same rows into one pass, and check that the indexes cover them. Counter tables stay rejected unless both steps fall short.
- **Seed.** Add an "Elliot shape" next to the current one: one project, 58 deployments of 2K to 35K photos, about two boxes per photo with 21% at 0.2 or higher, few large events (about 62 photos each), ViT-S embeddings. His `addaxai-db-stats.txt` (v7.16, Windows, 3.8 GB) is the source. Simon's numbers came on 2026-11-24 and did not change the seed (see "Simon's database (2026-11-24)").
  **Built (2026-11-22):** `--shape elliot` copies each of his 58 deployments (photos, boxes, share at 0.2 or more), about 270 species weighted 1/rank, events of about 62 photos, no legacy tail. Full scale gave 1,271,059 photos, 2,568,006 boxes, 763,734 embeddings and 20,669 events (his: 1,271,059, 2,571,682, 772,337, 20,555), 3.4 GB, in 20 minutes. Golden copy: `/Volumes/2TB-SSD/addaxai-speed/golden-elliot`.
- **Measuring.** "How we measure" gets the Labels page clicks: the time from a verify, a scroll through blanks and a filter change until the page is settled, on both shapes.

## Later: folder runs

Not a speed fix. Small, can go with any release.

A folder run is a hidden project with all its data, kept forever, no cap (`routers/folder_runs.py:524-560`). When several runs point at the same folder only the newest is listed, so the older ones are invisible and cannot be deleted (`:603-609`, `:386-387`). `delete_folder_run` also leaves the job rows.

- **Decided:** a new run for the same folder replaces the hidden duplicate.
- **Decided:** existing hidden duplicates are not deleted automatically. The clean-up dialog (CLEAN UP DATABASE in `TODO.md`) lists them.
- **Rejected:** keeping only the last ten runs. It deletes data without the user noticing.

The clean-up dialog and the project archive were worked out here on 2026-11-20 and moved to `TODO.md` under "Nice to haves" on 2026-11-24.

## What we rejected (so nobody rebuilds it)

- **Pruning as a speed fix.** B-tree depth grows with the logarithm of the row count: millions of rows are four or five levels, and removing 30% of them almost never removes a level. An indexed lookup costs the same. What scales with size is counts, full scans, ANALYZE, VACUUM and backups. Lightroom users report the same about splitting catalogs.
- **One database per project.** Timelapse and Camelot work this way. For us it is a near rewrite: no foreign keys across files, no atomic commit across attached files in WAL mode, a default limit of ten attached databases.
- **Integer primary keys in place of UUID strings.** Would shrink every index (each key is 36 bytes, repeated in all twelve indexes on `detections`), but the migration touches every table and every bundle.
- **Counter tables.** The standard answer to slow `COUNT(*)`, but triggers or bookkeeping in every write path. Only if counts are still slow after A and B.
- **Keyset paging.** The official alternative to `OFFSET`, but it touches paging, find and prefetch in the frontend. Only if the slow-request log shows deep pages are a problem.
- **Embeddings in a separate file.** The digiKam pattern; would roughly halve the main file and its backups. A medium refactor of the sort worker and the bundles. Only if backup time is still a problem.
- **`mmap_size`.** On Windows a memory-mapped file cannot be truncated, and an I/O error becomes a crash.
- **A database server**, as digiKam offers. Too heavy for a desktop installer.
- **Backup in small steps.** Every write by the app restarts an incremental copy from zero, so during an analysis it may never finish.

## Measured facts (2026-11-20, backup copy of a dev database, 218 MB, mostly video)

- `detections`: about 286 B per row in the table and about 287 B across its twelve indexes.
- `detection_embeddings`: float16 vectors, 768 B (ViT-S) or 2048 B (ViT-L). With ViT-L one row fits per 4 KB page, so about 3.9 KB on disk per embedding. 45% of the table's pages were unused space left by deletes.
- No `sqlite_stat4`, so the planner sees averages only.
- Not measured: any duration, and anything on a real large database.

## Measured facts (2026-11-20, seeded database, SQLite 3.45.1)

**Event generation was quadratic. Fixed the same day.** `generate_events_for_project` holds every file of a deployment in its session. `calculate_max_n_for_event` runs once per event and deleted the event's old observations with an ORM `delete` whose default `synchronize_session="evaluate"` tests the `WHERE` against every object in the session. So one event cost one pass over all files. The fix is `synchronize_session="fetch"`, which expires only the rows SQLite returns as deleted (`crud/event_observation.py`).

| Photos in one deployment | Events | Before | After |
|---|---|---|---|
| 9K | 3,044 | 8 s | |
| 27K | 9,043 | 43 s | 13.6 s |
| 54K | 17,934 | 168 s | 27.8 s |
| 900K | 300,408 | about 13 h (extrapolated; the seed was stopped after 2 h) | 23 min (database on a USB SSD) |

At 900K the fixed version waits on the disk, not on Python: the whole deployment is one transaction, it outgrows the 64 MB page cache, and SQLite spills pages to the WAL and reads them back (`pagerStress` in a stack sample, about 75% of the time in `pread`/`pwrite`). A candidate for later, not measured on an internal disk yet.

This runs after every analysis and postprocessing job, so a user with one large deployment and many events paid it every time. Simon's time-lapse folders make a few giant events and escaped most of it; Elliot's 1.2M project probably did not. Tell both when it ships.

**Baseline before phase A (`main` at 567ed76d plus phase 0).** Seeded database of 4.6 GB: 987K photos, 2.8M boxes, 1.48M embeddings, 329K events. It runs from the internal disk of a MacBook. The file had just been copied, so it sat in the OS cache, and a real cold start after a reboot is slower. Times are from the slow-request log at threshold 0.

| What | Cold | Warm |
|---|---|---|
| Startup until `/health` answers | 41 s: `ANALYZE` 34 s, daily backup 7 s, migrations and the rest under 1 s | |
| Project list (`GET /api/projects`) | 7.3 s | 7.3 s |
| Dashboard "Large", most cards | about 23 s | about 18 to 22 s |
| Dashboard "Large", Verification card (`/api/events/verification-stats`) | 204 s | 232 s |
| Dashboard "Small", slowest request | 2.8 s (`verification-stats`) | 3.8 s |
| Labels "Small": `label-tree` | 64 s | 70 s |
| Labels "Small": `labels/stats` | 93 s | 105 s |
| Labels "Small", grid until sorted crops show | about 95 s, because the sort starts only after `labels/stats` returns | |
| Labels "Small" with one species filter (`?lbl_labels=<taxonomy id>`), sort until first crops | about 7 s | |
| Deployment info, "Small" / "Large" | 0.6 s / 26.8 s | 0.2 s / 21.2 s |
| Site info, "Small" / "Large" | 0.1 s / 9.6 s | 0.1 s / 8.3 s |

What it shows:

- **Startup.** It is mostly `ANALYZE`, which makes A3 clearly worth doing.
- **Requests stuck behind one blocker.** On the "Large" dashboard, five requests finish within 0.1 s of each other (`labels/progress`, `species`, `timeline`, `animal-photos`, `capture-date-coverage` at 23.4 s). Requests that end together like that were not each slow, they waited behind one blocking request. That is A2, and `labels/progress` is one of the `async` endpoints A2 lists.
- **The Verification card hangs.** This reproduces Simon's report.
- **The small project is slow because of the big one.** "Small" has 27K photos, but `label-tree` and `labels/stats` take more than a minute. That is the phase B problem, and both endpoints should join B6's list.
- **Warm loads are no faster.** The cost is query plans and work, not cold caches.

## Measured facts (2026-11-21, after phase A)

Same seeded database, runtime and pages as the baseline. The baseline-code column was measured the same day with the same procedure, for the rows where it was taken, because the machine was not in the same state as the day before.

| What | Baseline (11-20) | Baseline code (11-21) | Phase A |
|---|---|---|---|
| Startup until `/health` answers | 41 s | 43 s | 0.2 s (no migration, so no `ANALYZE`; backup 25 s in the background) |
| Project list (`GET /api/projects`) | 7.3 s | 6.7 s | 4.5 s |
| Dashboard "Large", cards | about 23 s, five ending together | 27.9 s, five ending together | each on its own: species 1.4 s, date coverage 2.0 s, timeline 4.5 s, overview 11 s, animal photos 16 s, `labels/progress` 23 s |
| Dashboard "Large", Verification card | 204 s | 177 s | 125 s, see below |
| Dashboard "Small", slowest request | 2.8 s | | 2.3 s |
| Labels "Small": `label-tree` / `labels/stats` | 64 s / 93 s | | 24 s / 34 s |
| Labels "Small", sort starts after | about 95 s | | 33 s |
| Deployment info, "Small" / "Large" (cold) | 0.6 s / 26.8 s | | 0.3 s / 23.5 s |
| Site info, "Small" / "Large" (cold) | 0.1 s / 9.6 s | | 0.05 s / 10.0 s |
| Longest `/health` wait during a reprocess of "Small" (55 s job, 26 s of it `ANALYZE`) | | 45 s; the job WebSocket dropped on a missed keepalive | 0.76 s |

What it shows:

- **Startup and jobs are fixed.** Startup was `ANALYZE` and the backup; neither waits for `/health` now. A job no longer stops the app.
- **Nothing waits behind anything else.** The dashboard cards now finish in their own time. What is left slow is slow on its own: `labels/progress`, `verification-stats`, `label-tree`, `labels/stats` and the info sheets of "Large". Those are phase B.
- **One engine (A1) buys little by itself.** Project list 6.7 to 5.3 s, "Large" info sheets about a third faster. A controlled check (the same query alternating a brand-new and a pooled connection) shows no penalty either way.
- **The Verification card is disk-bound and its time is noise.** The same request took 106, 125, 177, 346, 454 and 473 s across runs, and 669 s alone on a brand-new connection outside the app. That last run spent 16% of its time on CPU, mostly in system calls. It depends on what the OS happens to cache, not on the code around it. Its three `count(DISTINCT …)` queries likely spill temporary B-trees to disk. Phase B looks at the query.
- **The card-picture pass still runs while a project has none.** On the seed no photo exists, so every launch spends 19 s in the background trying again. With real photos a project gets its picture once.

## Measured facts (2026-11-22, phase B baseline)

`main` at 6a836cce (all of phase A), same runtime and procedure as before, on both seeds. Every SQL statement over 50 ms was also timed from inside the backend with its 3.45 query plan, through a throwaway `sitecustomize` tracer (not in the repo). Times are cold unless marked.

| What | Simon seed | Elliot seed |
|---|---|---|
| Project list | 5.4 s | 7.1 s |
| Dashboard, Verification card | "Large" 86 s, warm 93 s; "Small" 2.2 s | 61 s |
| Dashboard, other slow cards | "Large": `labels/progress` 32 s, animal photos 25 s, progress per label 25 s, overview 24 s, `label-colors` 15 s, timeline 11 s; warm about half | `labels/progress` 22.5 s, animal photos 16 s, overview 15 s, progress per label 15 s, `label-colors` 11 s |
| Labels, open the page | "Small": `labels/stats` 26 s, `label-tree` 15 s, sort starts at 26 s, grid after about 60 s; the same with one species filter | `labels/stats` 38 s, `filter-options` 37 s, `label-tree` 28 s, `labels/progress` 22 s, sort starts at 38 s |
| Sort worker alone | | similarity 24 s (13 s of it the count before the load), by event 23 s |
| Verify one card in the grid | `bulk-verify` 3.7 s, then `filter-options` 1.8 s and `labels/progress` 0.7 s | `bulk-verify` 2.3 s, then `labels/progress` 17 s and `filter-options` 30 s |
| Verify in the detail view | `bulk-verify` 5.8 s, a full re-sort, `label-tree` 22 s | `bulk-verify` 5.6 s, a full re-sort, `label-tree` 31 s, `labels/progress` twice (23 s each), `filter-options` 47 s, `label-colors` 10 s |
| Files tab, open / next page | | `label-tree` 48 s and `filter-options` 51 s on open; `labels/files` 13 to 15 s per page, and the prefetched page after it the same |
| Deployment / site info | "Small" 0.4 / 0.1 s; "Large" 22 / 9 s | biggest deployment 1.0 / 0.2 s |

What it shows:

- **Elliot's lag is the number of counts, not their plans.** Every slow statement on his seed starts from the project, which is the whole database. One pass over his 2.5M boxes costs 12 to 15 s on an NVMe disk whatever the plan, because each box is a row read. Opening the Labels page and one verify ran the "which species are present" query 7 times (84 s), the events-per-species count 4 times (83 s), each of the two progress counts 6 times (72 s and 60 s), and a `min(label_confidence)` 4 times (34 s). A verify refetches `labels/progress` and `filter-options`, which a verify does not change; one in the detail view also re-sorts and refetches four more.
- **Two queries start from the whole database without any species filter** (Simon's case). The embedding count in `labels/stats` scans every row of `detection_embeddings` (up to 41 s for "Small"). The label-tree count walks every box in the database in species order when the "Unverified" filter is on (up to 29 s).
- **The Verification card re-runs one subquery seven times.** Each of its seven statements recomputes the filtered events, with an `EXISTS` per event through `event_files`, `files` and `detections`.
- **The rest already start from the project and are slow because "Large" has 2.5M boxes**, read several times per page (`labels/progress`, overview, animal photos, the info sheets).
- **`bulk-verify` itself takes 2 to 6 s.** Not yet looked at.

**After the decided fixes (same day, Elliot seed):**

- **A verify refreshes only what it changed, once the clicks stop** (`invalidateAfterVerify` in `frontend/src/lib/invalidate-label-queries.ts`). Three quick verifies in the grid: one `labels/progress` (20 s) and one `label-tree` (29 s), where each verify used to refetch `labels/progress` (17 s) and `filter-options` (30 s). Two verifies in the detail view: the same pair once, with no re-sort, no `filter-options`, no `label-colors`. The label tree is now refreshed after a verify when the Verified filter is on; before, the grid left its counts stale.
- **The three duplicate indexes are dropped** (migration `b9c3d5e7f9a1`). On 3.45 the cascades seek the longer index, and deleting a 27K-photo project took 2.5 s with and without them.
- **A verify no longer scans every box.** `bulk-verify` looked its boxes up with `id IN (...) OR track_id IN (...)`, and 3.45 answered with `SCAN detections` because `track_id` is NULL on every image box, so its `sqlite_stat1` row read "any value matches the whole table". The row is now dropped after every `ANALYZE`, like the one for `idx_files_source_video`. Elliot seed: 2.3 to 6.8 s per verify, now 0.02 to 0.09 s.
- **Three more fixes, timed old against new on the same requests** (each twice; same answers):

  | Request | Simon seed, old | new | Elliot seed, old | new |
  |---|---|---|---|---|
  | `labels/stats` ("Small" for Simon) | 37 / 21 s | 2.0 / 0.3 s | 43 / 22 s | 11 / 11 s |
  | `label-tree`, Unverified ("Small") | 11.5 / 9.2 s | 0.2 / 0.2 s | 10.4 / 9.5 s | 8.6 / 7.8 s |
  | Verification card ("Large" for Simon) | 65 / 67 s | 42 / 46 s | 47 / 44 s | 26 / 27 s |

  `labels/stats` counts its three numbers in one pass that starts from the project; counting embeddings from `detection_embeddings` read every embedding in the database. The label-tree count fixes its join order with `CROSS JOIN`, project first: it never carries a label filter, so that start is always right, and left to itself 3.45 walked every box in the database in label order (315 to 423 s for "Small" off the USB SSD, 3.2 s cold now). The Verification card merges the counts that read the same rows, seven statements to four.
- **Measured and left alone.** `labels/progress`: the plain file count it could merge never took 50 ms; its two expensive counts read different rows. The Files tab: per page the count is 1.3 to 1.6 s and the page itself 11 to 14 s, because the event order sorts every file of the project through correlated subqueries on every page. Fixing that means walking events in index order, a rewrite for a separate decision.
- **Follow-up, 2026-11-23** (after a review of the above):
  - **The Verification card asks only for what it shows.** The dashboard card and the Counts page pill read two numbers, events and confirmed events, but `verification-stats` computed nine, five of which nothing read. `/api/events/count` now returns `confirmed` next to `count` in one statement, and those two screens, the folder-run page and the run README use it. `verification-stats` keeps four numbers for the reprocess dialog. "Large": 42 s to 7 s alone, 19 s while the rest of the dashboard loads. The card also shows each of its three parts when that part's request is back; it used to show "Loading" until the slowest one was.
  - **The Labels sort no longer waits for the embedding count.** It only used that count as "is anything embedded". "Small" on the Simon seed: grid on screen after about 10 s, was about 60 s.
  - **The index drop, timed at scale.** Deleting "Large" (900K photos) through `delete_project` on 3.45: 263 s without the three indexes, 277 s with them.
  - **Tried and dropped:** counting "events with something to count" from `event_observations` (2 s against 18 to 27 s, same number on the seed). The rules differ: observations also hold species a person added by hand, and they are not rebuilt when the threshold changes.
- **The Files tab is slow under "Sort by event" only (2026-11-23).** The earlier "11 to 14 s per page" was measured with that sort on; it is not a paging problem. On the Elliot seed (3.45, "Unverified", count included) a page takes 0.5 to 0.6 s sorted by path (the default) or time, 0.7 to 1.6 s random, also 1,000 pages deep, so keyset paging stays out. Event sort looked up each file's event through three correlated subqueries. It now joins files to their event once (one event per file is what generation writes; `test_every_file_is_in_one_event_at_most`). Same pages before and after on both seeds. Elliot seed: a page 5.5 to 9.4 s to 2.2 to 2.9 s, the `find` lookup behind "Open in files view" 18.3 s to 6.8 s; in the app 2.5 s per Next. With a species filter a page still takes about 7 s, where the species filter is the cost, not the sort. The sort is remembered per browser, so one visit with event sort keeps the tab on it.
- **The label lists (2026-11-24).** Opening the Labels page read every box of the project seven times for the label filter and the species colours: `filter-options` four times (two of them for an events-per-label count no screen read, one for the slider limit), `label-colors` once, `label-tree` twice. The same "which species are present" query ran three times. Now:
  - `filter-options` answers everything about the labels a project has in one pass: the labels, both names, the slider limit (now the lowest score among the boxes the pages can show, not among all boxes), the colours and the filter tree without counts.
  - `label-counts` answers the counts under a page's filters; a filter change or a verify refetches only that.
  - `label-tree` and `label-colors` are gone.

  | Reads of every box | Before | After |
  |---|---|---|
  | Open the Labels page | 7 | 2 |
  | Change a filter, or verify with "Unverified" on | 2 | 1 |
  | Relabel | 3 or more | 2 |

  Identical answers on both seeds (labels, both name maps, colours, the whole tree, and the counts for all three units with and without "Unverified"; the slider limit unchanged at 0.3 on both). Elliot seed, each request alone: `filter-options` 55.8 s to 14.4 s, and it now also carries the colours (`label-colors` was 8.3 s) and the tree. In the app the Labels page opened with `filter-options` 15.4 s and `label-counts` 10.5 s, where it had `filter-options` 36.5 s, `label-tree` 28 s and `label-colors` 10.7 s. One pass is still 10 to 15 s on 2.5M boxes; making a pass itself fast needs a project column on `detections` (rejected above).
- **Not built: a server-side cache of the label lists.** `PRAGMA data_version` would make it safe (tested on 3.45.1: it changes after every commit from any connection or process, never after a read). Left out because the test suite runs in one open transaction on an in-memory database, which the signal cannot see, so the cache would go untested; and after the split it would save one pass on later opens that the browser's five-minute cache mostly covers. The fallback if first opens are still too slow.
- **Item 2 and 3 above measured to nothing, so they are left out.** Folding the count into the load with `count(*) OVER ()` was slower (17.8 s against 10.5 to 16 s), because the load alone walks the capture-time index and stops at the cap in about 2 s, while the window needs every row. One shared connection with a 64 MB cache gave 17 to 22 s against 17 to 19 s with four.

## Measured facts (2026-11-24, final round)

`main` at 3b474944, SQLite 3.45.1, both seeds copied fresh from the golden files, same procedure as before. "Cold" is the first load after a backend restart, "warm" two more loads without one. The earlier columns are the tables above, taken on other days; a drift of 5 to 15% between days was seen on 11-21, so only large differences count. A dash means not measured that day.

**First start after the update** (the golden copies are one migration behind, so this is what a user gets):

| | Simon seed (4.6 GB) | Elliot seed (3.4 GB) |
|---|---|---|
| Pre-upgrade backup | 6.0 s | 4.4 s |
| Migration and `ANALYZE` | 13.4 s (`ANALYZE` 12.8 s) | 2.2 s (`ANALYZE` 2.1 s) |
| Until `/health` answers | 19.6 s | 6.7 s |
| In the background after it | daily backup 13.4 s, card pictures 13.3 s | card pictures under 1 s |
| Every later start | 0.1 s | 0.1 s |

**Simon seed** ("Large" 900K photos, "Small" 27K, species filter L. villosus):

| What | 11-20 | 11-21 | 11-22 | Now, cold | Now, warm |
|---|---|---|---|---|---|
| Project list | 7.3 s | 4.5 s | 5.4 s | 4.6 s | 7.5 / 6.4 s |
| "Large", Verification card | 204 s | 125 s | 86 s | 17.0 s | 16.4 / 14.6 s |
| "Large", `labels/progress` | about 23 s | 23 s | 32 s | 20.8 s | 22.8 / 21.4 s |
| "Large", animal photos / progress per label / overview | about 23 s each | 16 / - / 11 s | 25 / 25 / 24 s | 13.2 / 12.3 / 12.8 s | 13 to 16 / 14 to 16 / 11 s |
| "Small", slowest dashboard request | 2.8 s | 2.3 s | 2.2 s | 1.0 s | 0.8 / 1.0 s |
| Labels "Small", slowest request on open | 93 s | 34 s | 26 s | 1.9 s | 2.4 / 2.1 s |
| Labels "Small", sort start to first crop | about 95 s after open | 33 s after open | 26 s after open | 6 s, starts at once | 6 / 5 s |
| Same with the species filter | sort about 7 s | | | 2.7 s slowest, sort 6 s | 2.8 to 3.5 s, sort 7 s |
| Deployment info "Small" / "Large" | 0.6 / 26.8 s | 0.3 / 23.5 s | 0.4 / 22 s | 0.5 / 20.4 s | 0.4 to 0.6 / 23.5 s |
| Site info "Small" / "Large" | 0.1 / 9.6 s | 0.05 / 10.0 s | 0.1 / 9 s | 0.07 / 6.9 s | 0.1 / 7.2 to 7.7 s |
| Verify in the grid, "Small" | | | `bulk-verify` 3.7 s, then 1.8 s and 0.7 s | `bulk-verify` 0.03 s, then `label-counts` 0.9 s and `labels/progress` 1.0 s | |
| Verify in the detail view, "Small" | | | 5.8 s, a re-sort, `label-tree` 22 s | 0.03 s, then the same pair, 0.5 s | |
| Filter change / relabel, "Small" | | | | `label-counts` 0.8 s and a 6 s re-sort / `bulk-relabel` 0.07 s, three refetches under 1 s | |
| Longest `/health` wait during a reprocess of "Small" | | 45 s, then 0.76 s after phase A | | 1.7 s (job 60 s, `ANALYZE` 32 s of it) | |

**Elliot seed** (one project, 1.27M photos, 2.57M boxes, species filter G. guatimalensis):

| What | 11-22 | 11-24, label lists | Now, cold | Now, warm |
|---|---|---|---|---|
| Project list | 7.1 s | | 4.2 s | 5.5 / 6.9 s |
| Dashboard, Verification card | 61 s | | 5.2 s | 8.7 / 12.2 s |
| Dashboard, `labels/progress` | 22.5 s | | 14.1 s | 18.1 / 25.9 s |
| Dashboard, animal photos / progress per label / overview | 16 / 15 / 15 s | | 7.9 / 7.6 / 12.0 s | 12 to 17 / 13 to 16 / 13 to 14 s |
| Labels, open | `labels/stats` 38 s, `filter-options` 37 s, `label-tree` 28 s, sort starts at 38 s | `filter-options` 15.4 s, `label-counts` 10.5 s | `labels/progress` 20.3 s, `labels/stats` 19.8 s, `filter-options` 10.1 s, `label-counts` 9.4 s; sort starts at once | the same four, 13 to 37 s each |
| Labels, open with the species filter | | | the same four, 10 to 30 s | the same four, 19 to 53 s |
| Verify in the grid | `bulk-verify` 2.3 s, then `labels/progress` 17 s and `filter-options` 30 s | | `bulk-verify` 0.06 s, then `label-counts` 10.3 s and `labels/progress` 21.1 s | |
| Verify in the detail view | 5.6 s, a re-sort, `label-tree` 31 s, `labels/progress` twice, `filter-options` 47 s, `label-colors` 10 s | | 0.01 s, then `label-counts` 8.4 s and `labels/progress` 19.3 s | |
| Filter change / relabel | | | `label-counts` 10.2 s and a re-sort / `bulk-relabel` 0.07 s, then three refetches of 12 to 22 s | |
| Sort worker alone, similarity / by event | 24 / 23 s | | 48 to 135 s / 29 to 90 s, see below | |
| Files tab, per page alone: path / event sort / species filter | 0.5 to 0.6 s / 2.2 to 2.9 s / about 7 s (11-23) | | 0.5 to 3.2 s / 4.8 to 9.5 s / 21 to 45 s | |
| Biggest deployment info / site info | 1.0 / 0.2 s | | 1.1 / 0.1 s | 0.7 to 0.9 / 0.2 s |

Each count of the Elliot Labels page alone, through the tracer: `labels/progress` 20 s, `labels/stats` 18 s, `filter-options` 12.5 s, `label-counts` 9 s. In the page the four run at the same time and compete for the disk, and a request the previous page left behind keeps running, so the warm column is mostly that contention.

What got faster, for the release notes:

- **Starting the app** no longer waits for statistics or the daily backup: 41 s to 0.1 s. Only the first start after an update with a database change takes longer, once.
- **Verifying and relabelling** answer at once (2 to 6 s each on a big project, now under 0.1 s). The counts update once the clicking stops, and a verify no longer re-sorts the page.
- **The dashboard Verification card** of a big project: 204 s to 15 s on the Simon seed, 61 s to 5 to 12 s on the Elliot seed. The other cards no longer wait behind each other.
- **A small project in a big database** is as fast as a small project: its Labels page from about 95 s to 6 s until the photos show.
- **A big project's Labels page** starts sorting at once instead of after 38 s; on the Elliot seed the photos showed after about 30 s while the counts ran, and the counts fill in after 10 to 20 s.
- **A job never freezes the app** (a reprocess left `/health` waiting 45 s; now at most 1.7 s).

Still slow, with what the tracer shows (each request alone):

- **Every pass over the boxes of a big project, 9 to 20 s.** `labels/progress` (two passes, 20 s), `labels/stats`, `filter-options`, `label-counts`, animal photos, progress per label, the Verification card (an `EXISTS` per event, 10 to 11 s). Every plan starts from the project; the cost is reading 2.5M rows. Making a pass itself cheaper needs a project column on `detections` (rejected below).
- **Deployment info of a 900K deployment, 20 s.** 13.8 s of it are the two confidence averages, each a pass over the deployment's boxes; they differ only in `label_confidence IS NOT NULL` and fit in one pass. Site info does not compute them (7 s).
- **Project list, 4 to 7 s.** About half is Python: bucketing every file path for trap nights (2.2 s on the Simon seed, 2.9 s on the Elliot seed). The SQL is a file count and an observation sum over all projects (1 to 2 s), plus loading every file's path and capture time (1 s). The overview card runs the same trap-night load.
- **Files tab with a species filter on Elliot, 21 to 45 s.** The count and the page each run a correlated `EXISTS` over the boxes of every file of the project.
- **The sort worker on the Elliot seed.** Similarity 48 to 135 s and event sort 29 to 90 s over six runs each, timed from a script to the stream's `result` line, with nothing else running; the worker run by hand took 77 and 82 s, and 38 and 51 s. On 11-22 a single run each gave 24 and 23 s, timed in the browser to the first crop. The similarity load spends 47 s in one statement and the event sort 26 s in the "has an embedding" pass. See below.

Found on the way:

- **The similarity load changed plan with migration `b9c3d5e7f9a1`.** The worker runs on the model env's SQLite 3.54.1. On the golden copy it walks `idx_files_captured_at_local` and stops at the cap; after the migration it reads every row of `detection_embeddings`, vectors included, and sorts them in a temporary B-tree. Same statistics, one index fewer. But putting `idx_detection_embeddings_detection` back on a copy restored the old plan and not the time (51 s), so the plan flip does not explain the slower sort. Unexplained so far; the spread between runs suggests the OS file cache (16 GB RAM, a 3.4 GB database and Chrome).
- **Putting all three dropped indexes back made the backend slower on 3.45**, not faster: `labels/stats` 56 to 108 s against 27 to 30 s, `labels/progress` 64 to 76 s against 22 s. One run each, right after building the indexes, so indicative only; nothing points at the drop as a cost.
- **A verify in the detail view on "Small" refetched the counts pair twice within a second.** On the Elliot seed once. Cause not looked at.
- **The Files tab path sort is 3 s per page cold, 0.5 s warm** on the Elliot seed. The page sorts every file of the project in a temporary B-tree; on a warm cache that is cheap.

## Simon's database (2026-11-24)

From his `addaxai-db-stats.txt` (SQLite 3.54 on Windows, schema `f769a2f4166a`, so a version from about 10 November, not 7.2.1). He did not say which pages hurt most.

| | Simon, real | "Simon" seed |
|---|---|---|
| File size | 3.6 GB | 4.6 GB |
| Photos | 1.40M | 0.99M |
| Boxes | 1.44M, about 1 per photo | 2.8M, about 3 per photo |
| Boxes at 0.2 or more | 36% | 21% |
| Embeddings | 563K (ViT-S 507K, ViT-B 52K, ViT-L 4K) | 1.48M, ViT-S |
| Events | 28K, about 50 photos each | 329K, about 3 photos each |
| Projects | 36, of which 16 folder runs; 33 deployments, 3 sites | 5 projects, 10 deployments |
| Largest deployments | 291K, 200K, 190K, 144K, 139K, 102K photos, each its own project | one of 900K |

What it shows:

- **The 900K deployment is gone.** He has split it, as advised in October, into projects of one deployment each. "A small project next to a big one" still applies, less extremely: the biggest is about a fifth of the database. The seed stays the harder test, so it was not changed.
- **His machine reads about 6 to 12 times slower per row than Elliot's.** The script's per-deployment counts took 97 s and 202 s on his database against 17 s and 29 s on Elliot's, which has more boxes. A pass over one of his big projects can take a minute or more. Fewer passes per page and per click, the point of phase B, matter more for him than the seed's timings suggest.
- **About a tenth of the database is the same folders analysed again.** Four deployments are identical (24,419 photos, 30,113 boxes each), three more are (9,999 photos), and three pairs more. 13% of the file (121K pages, about 500 MB) is free space left by deletes, which only a compact gives back. Both are what CLEAN UP DATABASE in `TODO.md` is for; neither is a speed fix.
- **Events of about 50 photos** confirm the time-lapse cameras of his October report. That is `future-plans/sequences-and-independence.md`, not this plan.
- **His `idx_detections_track` statistic is a real one** (`1441894 1139`): he has videos and 1,266 tracks. Dropping it after `ANALYZE` (`_UNUSABLE_STAT_INDEXES`) leaves the planner assuming a selective index, which is what that index is on his data.

## Open questions

- How long `ANALYZE`, the backup and the first dashboard load take. Answered 2026-11-20: 34 s, 7 s and 23 s (Verification card 204 s). See the baseline under "Measured facts".
- Whether Simon's `sqlite_stat1` gives the plans we saw on the backup copy. Answered 2026-11-24: his statistics are milder than the seed's (one deployment per project, a label matches 2.8K boxes on average), so the wrong-start plans of the seed are less likely on his database.
- Whether Simon's AddaxAI folder is on an internal disk. His per-row speed suggests it is not. Ask him after the release.
- Whether the partial-index idea for `idx_files_source_video` holds on 3.45, including for the cascade lookup.
- Whether the three duplicate indexes are really unused by cascades.
- What module-level engine needs from the test setup and from the Electron e2e sandbox. Answered 2026-11-21: the engine is rebuilt when the database URL changes, which is what `tests/db/conftest.py` needs. The e2e suite starts its own backend process, so it is unaffected.

## Sources

- FastAPI, `def` versus `async def` with a blocking database library: https://fastapi.tiangolo.com/async/
- Thread limits and context propagation: https://anyio.readthedocs.io/en/stable/threads.html, https://docs.python.org/3/library/asyncio-task.html#asyncio.to_thread
- ANALYZE, `analysis_limit`, fixed statistics: https://www.sqlite.org/lang_analyze.html
- `PRAGMA optimize` and what changed in 3.46: https://www.sqlite.org/pragma.html#pragma_optimize, https://www.sqlite.org/releaselog/3_46_0.html
- Join order hints and the planner checklist: https://www.sqlite.org/queryplanner-ng.html
- What `sqlite_stat1` stores: https://www.sqlite.org/fileformat2.html#stat1tab
- Partial indexes: https://www.sqlite.org/partialindex.html
- WAL and long readers: https://www.sqlite.org/wal.html
- Online backup: https://www.sqlite.org/backup.html
- Keyset paging with row values: https://www.sqlite.org/rowvalue.html#scrolling_window_queries
- ATTACH limits: https://www.sqlite.org/lang_attach.html, https://www.sqlite.org/limits.html
- B-tree depth: https://use-the-index-luke.com/sql/anatomy/the-tree
- Maintenance when idle, never at start: Firefox https://searchfox.org/mozilla-central/source/storage/VacuumManager.cpp, Jellyfin https://jellyfin.org/docs/general/server/tasks
- Separate files for rebuildable data and a cleaning tool: digiKam https://docs.digikam.org/en/maintenance_tools/maintenance_database.html

## Key files

- `backend/app/db/base.py`: engine, session, pragmas, `refresh_query_statistics`, `init_db`
- `backend/app/main.py`: lifespan, backup call, the five background tasks after start
- `backend/app/db/backup.py`: `snapshot_db`, `ring_buffer_backup`, retention
- `backend/app/api/routers/`: `labels.py`, `events.py`, `files.py`, `deployments.py`, `sites.py` for the async endpoints
- `backend/app/utils/datetime_serialization.py`: the timezone ContextVar
- `backend/app/ml/inference/similarity_script.py`, `backend/app/services/label_service.py`: sort worker and how it is started
- `backend/app/api/crud/file.py`, `crud/event.py`, `crud/statistics.py`: the heavy aggregates
- `backend/app/api/routers/folder_runs.py`, `crud/project.py`: folder runs and their delete
- `backend/app/api/crud/deployment.py`: `purge_deployment_data`
- `electron/src/main.ts`: `/health` polling, the app menu
- `DEVELOPERS.md`: "Deleting analysis data", "Never run a bare ANALYZE", "Database backups", "Datetime conventions"

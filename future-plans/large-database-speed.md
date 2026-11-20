# Fast with a large database

Overarching plan, written 2026-11-20 after an investigation of the code, the query plans and how other tools handle the same problem. Nothing here is built. Each phase gets its own detailed plan in its own session; this file is the map they start from, so nobody has to redo the investigation.

The short version: the app is slow on a large database mainly because of its own habits, not because of the size. It opens a new database connection for every request, runs heavy queries where they block every other request, and does full-database work at every launch. Those are cheap to fix with standard practice (phase A). A second group is query plans that start from the whole database where they should start from the project (phase B). Deleting data does not make pages faster, it saves disk space and backup time, so a clean-up dialog and an archive option are features for later, not speed fixes.

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
- **Simon's real numbers.** A short mail with two read-only commands: his `sqlite_stat1` rows and the size per table (`dbstat`). A few KB of text, no photos or labels. Needed before phase B, useful for the seed script. Draft pending.

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

## Phase B: queries that start from the project (release 2)

**Decided scope: project-first plans, worker cache, duplicate indexes.** Keyset paging and counter tables wait until a measurement asks for them.

1. **Sort worker, project first.** There is no project column on `detections`, `files` or `detection_embeddings`; the project filter arrives only through `deployments` (`ml/inference/similarity_script.py:136-144`). `sqlite_stat1` holds one average per index for the whole database, so one huge deployment shapes the plan for every small project. Confirmed with `EXPLAIN QUERY PLAN` on a backup copy: with a label filter the plan starts from `idx_detections_label_taxonomy` (every detection with that label in the database), without one it scans all of `detection_embeddings` including the vector blobs. Fix: fixed join order with `CROSS JOIN` (deployments, files, detections), which SQLite documents for exactly this and advises to keep to the few queries that need it.
2. **Count folded into the load.** `_load_metadata` and `_load_embeddings` run `SELECT COUNT(*) FROM (sql)` and then the same query again (`similarity_script.py:369-379`, `:410-422`). One pass with `count(*) OVER ()`.
3. **Worker connections.** Each helper opens its own read-only connection with SQLite's 2 MB default cache, up to four per run (`:366`, `:403`, `:457`, `:476`). One connection per run, with `cache_size` and `temp_store` set.
4. **Duplicate indexes.** `idx_files_deployment` (prefix of `idx_files_deployment_path`), `idx_detection_embeddings_detection` (prefix of the unique index) and `idx_event_files_event` (prefix of the primary key). Each costs space and insert time and buys nothing. Check each against the foreign-key cascade rule in `DEVELOPERS.md` before dropping.
5. **Statistics, if plans still flip.** Idea to test, not verified: make `idx_files_source_video` a partial index (`WHERE source_video_id IS NOT NULL`). SQLite can use it for `= ?` lookups, and it may remove the reason the statistic has to be deleted after every `ANALYZE`. The documented fallback is shipping fixed `sqlite_stat1` rows.
6. **Other project-scoped queries.** The thumbnail pick (`ORDER BY confidence DESC LIMIT 10`) may walk the confidence index across the whole database; after A5 it runs rarely. Check the dashboard and Labels queries on the synthetic database and fix only what shows up in the slow-request log.

## Later: features

Neither is a speed fix once A and B are in. Both wait until users have tried the faster version.

### Folder runs (small, can go with either release)

A folder run is a hidden project with all its data, kept forever, no cap (`routers/folder_runs.py:524-560`). When several runs point at the same folder only the newest is listed, so the older ones are invisible and cannot be deleted (`:603-609`, `:386-387`). `delete_folder_run` also leaves the job rows.

- **Decided:** a new run for the same folder replaces the hidden duplicate.
- **Decided:** existing hidden duplicates are not deleted automatically. The clean-up dialog lists them.
- **Rejected:** keeping only the last ten runs. It deletes data without the user noticing.

### Clean-up dialog

"Clean up database…" in the File menu under Back up and Restore (`electron/src/main.ts:1191-1216`, add to `SETUP_GATED_MENU_IDS`). Pattern from `docker system df`, macOS Storage and Thunderbird: show what can be reclaimed, confirm, back up, act. Sold as a way to save space and backup time. No "X% faster" claim, because we cannot honestly make one.

| Candidate | Verdict |
|---|---|
| Boxes below the project threshold | **Never.** The threshold is applied at read time (`threshold_or_verified`), users can lower it later, and exports and the reprocess matcher need the rows. Deleting them is the design that caused the phantom reprocess errors. |
| Rejected and false-detection rows | **Never.** They record a human judgement; undo and `addaxai-detections.csv` depend on them. |
| Legacy boxes between 0.005 and 0.01 | Yes. No screen reaches them; 19% of rows in one measured database. |
| Embeddings, per project | Yes. They can be rebuilt, and they are the largest per-row cost (about 3.9 KB each on disk with ViT-L, measured). |
| Folder runs, including the hidden duplicates | Yes, listed with size and date, chosen by the user. |
| Projects that have been inactive for a while | Listed with size and last use, so the user can delete them here (or archive them, once that exists). Never suggested for deletion on age alone. |
| Compact (VACUUM) | Required, or nothing above shrinks the file. The live database is never vacuumed today and has `auto_vacuum=NONE`, so every delete leaves free pages in the file and in each backup. Minutes on a 4 GB file, needs up to twice the size in free disk space, runs with the app locked and progress shown. |

Sizes come from `dbstat` and `PRAGMA freelist_count`; check that the packaged SQLite has `dbstat` compiled in. Everything goes through the normal delete code (`purge_deployment_data`), never an ad-hoc script.

### Archive a project

Built on project bundles (`services/project_package.py`), which already carry detections, verifications, embeddings, events, observations and tracks by id, and already migrate an older bundle forward on a temporary copy at import. Archive = write a bundle without media to an archive folder, delete the project, compact, show a greyed-out card with a restore option. Migration happens at restore, the way Lightroom and Camelot do it; migrating every archive at update time breaks when one sits on an unplugged drive.

Gaps to close first: the audit log, the deployment queue, job details and a custom project picture are not carried; label rows are matched to local ones on import; `shortcut_labels` after a label remap is unchecked.

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

## Open questions

- How long the full `ANALYZE`, the backup and the first dashboard load take on the synthetic database. This decides A3.
- Whether Simon's `sqlite_stat1` gives the plans we saw on the backup copy.
- Whether the partial-index idea for `idx_files_source_video` holds on 3.45, including for the cascade lookup.
- Whether the three duplicate indexes are really unused by cascades.
- What module-level engine needs from the test setup and from the Electron e2e sandbox.

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
- VACUUM and auto_vacuum: https://www.sqlite.org/lang_vacuum.html
- `dbstat`: https://www.sqlite.org/dbstat.html
- Keyset paging with row values: https://www.sqlite.org/rowvalue.html#scrolling_window_queries
- ATTACH limits: https://www.sqlite.org/lang_attach.html, https://www.sqlite.org/limits.html
- B-tree depth: https://use-the-index-luke.com/sql/anatomy/the-tree
- Maintenance when idle, never at start: Firefox https://searchfox.org/mozilla-central/source/storage/VacuumManager.cpp, Jellyfin https://jellyfin.org/docs/general/server/tasks
- Separate files for rebuildable data and a cleaning tool: digiKam https://docs.digikam.org/en/maintenance_tools/maintenance_database.html
- Archive by export, migrate on open: Lightroom https://helpx.adobe.com/lightroom-classic/desktop/manage-catalogs-and-files/create-catalogs.html, Camelot https://camelot-project.readthedocs.io/en/latest/datasets.html
- Show reclaimable space, then act: https://docs.docker.com/reference/cli/docker/system/df/

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
- `backend/app/services/project_package.py`: bundles, the base for archive
- `electron/src/main.ts`: `/health` polling, the app menu
- `DEVELOPERS.md`: "Deleting analysis data", "Never run a bare ANALYZE", "Database backups", "Datetime conventions", "Project packages"

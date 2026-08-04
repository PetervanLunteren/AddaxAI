# Developer Documentation

## Runbooks (skills)

Recurring, gotcha-heavy jobs have step-by-step runbooks under `skills/`. Read the
relevant one before starting, and follow its gotcha list and validation checklist
rather than re-deriving the process:

- `skills/add-classification-model/SKILL.md` : add a new classifier to the zoo end to end.
- `skills/build-taxonomy-csv/SKILL.md` : produce a model's taxonomy.csv via GBIF.
- `skills/run-model-test-harness/SKILL.md` : run and read `test_models.py`.

The sections below are the reference material those runbooks point at.

## After cloning

Activate the commit-msg hook that strips auto-generated co-author lines:

```bash
git config core.hooksPath .githooks
```

This only needs to be run once per clone.

## Logging & Debugging

**Log files:** All logs (backend + frontend) are written to `~/AddaxAI/logs/backend.log`

**Watch logs in real-time:**
```bash
tail -f ~/AddaxAI/logs/backend.log
```

**Add logging in code:**
```python
# Backend (Python)
from app.core.logging_config import get_logger
logger = get_logger(__name__)
logger.info("Operation completed")
logger.error("Something failed", exc_info=True)  # Include stack trace
```

```typescript
// Frontend (TypeScript)
import { logger } from "@/lib/logger";
logger.info("User clicked button", { buttonId: "create-project" });
logger.error("API call failed", { endpoint: "/api/projects", error: err.message });
```

**Log retention:** Automatic rotation at 33MB per file, keeps 3 backups (100MB total, ~7 days).

## Database migrations

Schema changes ship as Alembic migrations under `backend/alembic/versions/`. The app runs `alembic upgrade head` automatically on every startup (`init_db()` in `backend/app/db/base.py`), so users never run migrations by hand.

**Adding a migration:**

```bash
cd backend
source venv/bin/activate
PYTHONPATH=. alembic revision --autogenerate -m "short description"
```

Review the generated file before committing. Autogenerate is helpful but imperfect: it misses index renames, check constraints, and server-default changes. Edit the upgrade/downgrade bodies if needed.

**Shipped migrations are immutable.** Once a migration has shipped, never edit it or change the schema it created. Make a new migration instead. This is the one rule Alembic depends on: `upgrade head` is only reliable when every DB started at a known revision and ran the official chain forward, in order. Editing a shipped migration (or editing a model without a matching migration) makes the live schema disagree with what the recorded revision claims, and that drift is what caused the repeated startup crashes on beta-tester DBs. `test_upgrade_from_base_matches_models` in `tests/db/test_migrations.py` is the CI guard: it runs the whole chain from base and asserts `schema_problems()` is empty, so a migration that drifts from the models fails the build.

**A migration that touches rows needs a test in `tests/db/test_migration_data.py`.** The CI guard above runs the chain against an *empty* database, so every `UPDATE` and `DELETE` in it matches zero rows and its correctness is never exercised. That is the one remaining way to lose data silently: a wrong data migration leaves the schema matching the models perfectly, so the startup check waves it through, and the only symptom is a user whose verification work is quietly wrong. The recipe is four steps and there are six worked examples in that file:

```python
def test_<revision>_<what it should do>(engine):
    upgrade_to("<the revision before yours>")   # the input schema
    ...insert_row(conn, ...)...                 # the data shape it handles
    upgrade_to("<your revision>")               # one step
    ...assert what the rows became...
```

Use raw SQL via `insert_row` from `tests/db/conftest.py`, never the ORM factories in `tests/conftest.py`: those describe the schema at head, and these tests write into the schema as it was. `insert_row` fills in every required column you do not name, so a test mentions only the columns it cares about.

Nothing detects for you whether your migration needs one of these. That is deliberate: telling "does this touch rows" from the source means a regex over migration text, and a guard that silently false-negatives is worse than none because it converts "I should think about this" into "CI would have caught it".

**Prove the test can fail.** Break the migration on purpose and watch the test go red before you commit it. A data test that passes against a broken migration is the same trap as the idempotency test described below.

**`alembic_version` is ground truth.** The app does not try to work out what revision a schema is "really" at. It runs `alembic upgrade head` and then checks the result against `Base.metadata`. An earlier design did guess, by introspecting marker columns listed in a hand-maintained fingerprint table, and when the guess disagreed with the stamp it re-stamped the DB backwards and replayed the chain. Guessing cannot work in general (drops, renames and data backfills leave no trace), it needed a new bookkeeping row per migration forever, and the replay re-ran one-time data migrations over data that had already moved on. That destroyed user verification work on 2026-05-27. **Never reintroduce automatic replay.** A DB we cannot trust gets handed back to its owner with an error they can act on, not silently rewritten.

**Four rules in `init_db()`:**

1. Empty DB: `alembic upgrade head` builds the whole schema from base.
2. User tables but no `alembic_version` row: refused. Alembic has run on every launch since 2026-05-08 (`78dc9d9c`, which replaced `Base.metadata.create_all` plus a hand-rolled column patcher), so such a DB is from an early beta. This is also what catches the issue #11 shape (Arky's Linux install), which used to be stamped at the initial revision and then died mid-chain with `KeyError: 'captured_at_local'`.
3. A stamped revision that is not on disk, or more than one version row: refused. Alembic raises `CommandError` while resolving the chain, before running anything, and `ensure_upgradable` rejects an ambiguous version table up front rather than letting `get_current_revision` read whichever row SQLite returned first.
4. After upgrading, `schema_problems()` must be empty. Anything missing means the stamp lied, so we stop.

Every refusal raises `SchemaError`, whose message is written for the end user. The lifespan writes it to `~/AddaxAI/.startup-error.txt` and the Electron error page shows it verbatim with **Restore from backup** and **Delete database and start fresh** buttons, because the backend exits before the API or the frontend exist and the in-app dialogs are unreachable at that moment. Those buttons write the same `.restore-on-next-launch` / `.wipe-db-on-next-launch` markers the in-app flows use, so there is one recovery mechanism, not two. Electron deletes the error file just before every spawn, so a message there always belongs to the current launch.

**A slow migration is not a failure.** `waitForBackend` waits on a live backend indefinitely; the only failure is the process dying. After `BACKEND_SLOW_NOTICE_MS` (60s, `ADDAXAI_SLOW_NOTICE_MS` to override) the splash is replaced by a "Still working" page with Open logs and Quit, and **no Retry**. That omission is the point. A backend part-way through a migration has not finished its lifespan, so it does not answer `/health` and looks identical to a wedged one. The old three-minute deadline sent the user to the error page, and its Retry re-entered `ensureBackend`, which probed `/health`, saw nothing, concluded the port was free, and spawned a *second* backend running `alembic upgrade head` against the same SQLite file, orphaning the first. The bigger the database, the likelier that was. The trade made here is that a genuinely wedged backend now waits forever rather than erroring, which is the right side to be wrong on when the two cannot be told apart and Quit is one click away.

**The chain is not replay-safe, and does not need to be.** The forward path runs each migration exactly once against the input it expects. A DB whose stamp is legitimately behind (a restored older backup) replays forward, which is that same once-each path, not a re-run. Nothing replays a migration over its own output: stamp an at-head DB back to base and upgrade and it dies immediately on `table audit_log already exists`. Making that work would mean guarding all 24 shipped migrations including the initial `CREATE TABLE`s, for no benefit. **Do not write a test asserting the chain is idempotent.** There was one; it re-ran zero migrations, because alembic no-ops at head, so it was green and asserted nothing.

**Guard DDL anyway in new migrations.** It costs nothing at authoring time and it pays off on a drifted DB, where it turns a mid-chain crash into the clean `SchemaError` refusal with the recovery buttons. `d4e5f6a7b8c9` is the worked example. Bare `op.batch_alter_table(...).alter_column(...)` throws `KeyError` on SQLite when the reflected table lacks the column. The older migrations do not all do this and are not being retrofitted.

**What `schema_problems()` compares.** Alembic's own `compare_metadata`, filtered to the additive operations, plus a hand-rolled walk over foreign key `ON DELETE` actions. One rule: anything the models declare that the DB lacks. That covers missing tables, columns, indexes and named unique constraints. It deliberately ignores the `remove_*` ops (a half-applied `DROP COLUMN` is harmless) and the `modify_*` ops (nullability, column types, server defaults), where SQLite's loose typing makes a false alarm likelier than the skipped migration it would catch, and a false alarm here refuses a healthy user's launch. The foreign key walk exists because `compare_metadata` reports no FK diffs at all, and a lost `ON DELETE CASCADE` is the one way this design can orphan or lose rows (see "Deleting analysis data").

**Helpers** (in `backend/app/db/migrations.py`): `SchemaError`, `ensure_upgradable(engine)`, `schema_problems(engine)`, `get_current_revision(engine)`, `get_head_revision()`, `needs_upgrade(engine)`, `upgrade_to_head()`. All alembic imports are local to the function bodies so the module is cheap to import.

**Dropping a column on SQLite: use raw DDL, not batch mode.** Alembic's `op.batch_alter_table(...) as batch: batch.drop_column(...)` is the textbook pattern but it builds its starting-point table from `target_metadata` (the SQLAlchemy models). When you remove the column from the model in the same commit as the migration (the natural workflow), the metadata no longer has the column and the batch flush dies with `KeyError`. `copy_from=sa.Table(name, sa.MetaData(), autoload_with=op.get_bind())` is the documented escape hatch but does not reliably pick up the column on every live DB. Since SQLite 3.35+ (Python 3.13 ships with 3.51) supports native `ALTER TABLE DROP COLUMN`, just write the DDL directly.

**Always guard column drops with a presence check.** A blind `DROP COLUMN` on a DB that is stamped past the add-column migration but never actually got the column will crash startup, and a DB whose stamp is behind replays the chain forward as a matter of routine. Skip the DDL when the column isn't there, since the end state matches what the migration was trying to achieve:

```python
import sqlalchemy as sa
from alembic import op

def _projects_columns(bind) -> set[str]:
    return {c["name"] for c in sa.inspect(bind).get_columns("projects")}

def upgrade() -> None:
    bind = op.get_bind()
    if "observations_max_detections" in _projects_columns(bind):
        op.execute(
            "ALTER TABLE projects DROP COLUMN observations_max_detections"
        )

def downgrade() -> None:
    bind = op.get_bind()
    if "observations_max_detections" not in _projects_columns(bind):
        op.execute(
            "ALTER TABLE projects "
            "ADD COLUMN observations_max_detections INTEGER NOT NULL DEFAULT 20000"
        )
```

One line of DDL, no reflection gymnastics, no batch-mode failure mode, idempotent against drifted DBs. Add-column never needed batch (`op.add_column` works directly). Batch mode is still the right call for type changes and nullability tweaks, which do not have a native single-statement DDL on SQLite.

## Deleting analysis data

Deleting a project, deleting a folder run, re-running a folder run and deleting a deployment all end in the same place: one `db.delete(<parent>)` that has to remove every file, detection, embedding, event and observation underneath it. On a large run that is millions of rows, so how it is done matters.

**The database owns the cascade, not the ORM.** Every parent/child foreign key declares `ON DELETE CASCADE`, `PRAGMA foreign_keys=ON` is set on every connection (`db/base.py`), and every `cascade="all, delete-orphan"` relationship sets `passive_deletes=True`. SQLAlchemy therefore emits one `DELETE` for the parent and lets SQLite do the rest in C.

**The rule, with no exceptions:**

> every relationship with `delete-orphan` whose child foreign key declares `ON DELETE CASCADE` sets `passive_deletes=True`

`tests/models/test_cascade_config.py` enforces it. Use `True`, never `"all"` (that one is rejected in combination with `delete-orphan`).

Without it, SQLAlchemy loads every child row into memory as a Python object before deleting it, one lazy SELECT at a time. Deleting a 50,000-file run emitted 277,014 statements and peaked at 1.5 GB of RAM.

**Index the child column of every foreign key.** When a parent row is deleted, SQLite has to find the rows referencing it. With no index on the child column that is a full table scan **per deleted parent row**. `event_observations.max_n_file_id` (`ON DELETE SET NULL` to `files`) was unindexed, so deleting a 50,000-file run scanned the whole observations table 50,000 times: 8 minutes on a fast Mac, hours on a beta tester's laptop. The same shape applies to `NO ACTION` foreign keys, where SQLite still has to prove no child references the row.

These indexes exist for constraint enforcement, not for queries, so they look unused when you grep for them. Do not remove them. `test_upgrade_from_base_matches_models` in `tests/db/test_migrations.py` is the guard: it runs the chain from base and asserts `schema_problems()` is empty, which covers missing tables, columns, indexes, named unique constraints and foreign key `ON DELETE` actions in one assertion.

**That guard is one-directional, on purpose.** `schema_problems()` reports only what the models declare and the database lacks (`_ADDITIVE_OPS`). So it catches "added an index to the model, forgot the migration", which is the mistake people actually make, and it stays quiet about an index the migrations create that no model declares. Verified by removing each half in turn: the forgotten migration fails with `missing index <name> on <table>`, the extra live index passes. If you need a query index gone, remove it from both sides or it silently survives.

Also index the columns a hot query filters on, not only foreign keys. `files.file_path` was unindexed while ingest looked every file up by `(file_path, deployment_id)` before inserting it, so the planner fell back to `idx_files_deployment`, which has no selectivity when a whole library sits in one deployment. File N scanned N rows and a 1M-image ingest spent ~11 hours on lookups alone. `idx_files_deployment_path` (migration `7a8b9c0d1e2f`) makes it a single seek.

**Consequences to know about:**

- After a passive delete, child objects already in the session are stale until the session is expired. Every call site commits immediately (`expire_on_commit` is on), so app code never sees this, but a test that does `db.delete(parent); db.flush()` and then `db.get(Child, id)` will get the cached object back. Query the database instead.
- Already-loaded collections still cascade in Python. `passive_deletes` suppresses *loading*, never cascading of what is loaded.
- A missing `ON DELETE CASCADE` fails loudly with `FOREIGN KEY constraint failed`, not silently with orphan rows, because the child foreign keys are `NOT NULL`.

**Timing.** `POST /api/folder-runs/{id}/rerun` and `_delete_deployment_artifacts` both log elapsed seconds. The on-disk `.addaxai` cleanup stays inside the request and is unbounded on a slow external drive, so when a delete is reported as slow, those two lines are what tell you whether it was the database or the disk.

## Database backups

The DB at `~/AddaxAI/addaxai.db` holds irreversible work (human verifications). It is the only piece of user state that cannot be rebuilt by re-running analysis, so it gets a backup story. Backups live under `~/AddaxAI/backups/` and use SQLite's online backup API (`sqlite3.Connection.backup`) so they are WAL-safe and produce a single consolidated `.db` file with no `-wal` / `-shm` siblings.

Four kinds of snapshot:

| Kind | When | Filename pattern | Retention |
|---|---|---|---|
| Daily rolling | App startup, throttled to one per UTC date | `addaxai-<utc-iso>.db` | Keep 5 newest |
| Pre-upgrade | Startup, only when alembic detects a pending upgrade | `addaxai-pre-upgrade-<rev>-<utc-iso>.db` | Keep 5 newest, one per revision |
| Pre-restore | Right before a restore swaps a backup in | `addaxai-pre-restore-<utc-iso>.db` | Keep 5 newest |
| Manual | User clicks "Back up database", or the app is about to wipe the DB | `addaxai-manual-<utc-iso>.db` | Never auto-pruned |
| Manual to chosen folder | User clicks "Back up database" → "Save to chosen folder…" | `addaxai-<utc-iso>.db` in user-picked dir | Untouched by the app |

**One pre-upgrade snapshot per revision.** A second copy of the same unmigrated DB is worth nothing and costs a full DB copy, so `pre_upgrade_backup` returns `None` when it already has one for that revision. Without that, repeated launches at the same revision push the real pre-upgrade snapshot out of the ring buffer with identical duplicates, which is exactly the snapshot that matters on the day a migration eats data. Two real ways that happens: a `uvicorn --reload` storm in dev (one observed run left 805 MB of five identical copies), and a user clicking Retry on the startup error page.

**Deleting the DB takes a manual snapshot first.** The `.wipe-db-on-next-launch` marker used to be reachable only by typing `RESET` in the Settings dialog. The startup error page can now write it behind a native confirm, which is a lighter gate on an irreversible action, so the lifespan snapshots the DB before it unlinks it. Manual snapshots are never auto-pruned and show up in the restore picker, so the wipe stays undoable.

**Restore flow.** The frontend posts `/api/backup/restore` with a source path. The backend validates it and writes `~/AddaxAI/.restore-on-next-launch` containing the absolute path. The renderer then asks Electron to quit; the next launch's lifespan calls `consume_restore_marker(settings)` before `init_db()`, which force-snapshots the current live DB to the ring buffer first, then swaps the source file in. The marker is consumed unconditionally even on failure, so a corrupt request can't loop the user through restore-fail-restore-fail forever; the live DB is left untouched on validation failure. The Electron startup error page writes the same marker directly, since a refused DB means the API never comes up.

**Validation is `PRAGMA integrity_check` plus an `alembic_version` row.** The version row is not pedantry: a DB without one predates 2026-05-08 and `init_db` refuses it, so accepting it here would restore a file that fails on the very next launch, with nothing telling the user that the file they picked was the problem.

**Key files:**

| File | Purpose |
|---|---|
| `backend/app/db/backup.py` | Snapshot / validate / ring-buffer logic, restore-marker helpers |
| `backend/app/api/routers/backup.py` | `/api/backup/{dir,list,snapshot,restore}` endpoints |
| `backend/app/main.py` lifespan | Consumes the restore marker, takes daily + pre-upgrade snapshots before `init_db()` |
| `backend/app/core/startup_error.py` | Writes the refusal the Electron error page reads |
| `frontend/src/components/diagnostics/BackupNowDialog.tsx` | Manual backup UI (ring buffer or chosen folder) |
| `frontend/src/components/diagnostics/RestoreBackupDialog.tsx` | Restore UI with type-`RESTORE`-to-confirm gate |
| `frontend/src/components/layout/AppHamburger.tsx` | Back up / Restore / Open backups folder menu items |
| `electron/src/main.ts` (`db:restore`, `db:reset`) | The same two actions when the backend will not start |

Pre-init backups in lifespan are best-effort. If `~/AddaxAI/` is read-only or the disk is full, the failed snapshot is logged at error level and startup continues, so the user can at least open the app to see the diagnostic banner and react.

## Removing the legacy AddaxAI install

AddaxAI 7 installs to different locations than AddaxAI 6, so upgrading leaves two full apps on the machine, the old one holding 10 to 30 GB of conda envs and model weights. The app finds the old install and offers to delete it.

**Where legacy lives** (verified against the legacy repo's own install scripts):

| OS | Install root | Also |
|---|---|---|
| Windows | `%USERPROFILE%\AddaxAI_files\` | junction `%USERPROFILE%\EcoAssist_files` pointing at the root, plus a manual-install variant at `%ProgramFiles%\AddaxAI_files` |
| macOS | `/Applications/AddaxAI_files/` | desktop symlink `~/Desktop/AddaxAI.app` |
| Linux | `~/.AddaxAI_files/` | `~/Desktop/Linux_open_AddaxAI_shortcut.desktop`, `~/.icons/logo_small_bg.png` |

Legacy writes nothing outside that tree. Its analysis outputs land in the user's own image folders and a destination folder they picked, so no user data is at risk.

**Why this is not in the installers.** macOS ships a dmg and drag-to-Applications runs no code. The Linux deb `postinst` runs as root, so `$HOME` is root's home and `$SUDO_USER` is unset under App Center / PackageKit. Only Windows NSIS could do it, which would mean one platform covered and three implementations. One Python implementation running as the logged-in user covers all three, with the right permissions on each.

**Detection is one rule on every platform:** `<root>/AddaxAI/AddaxAI_GUI.py` exists. Folder name alone would be wrong on Windows, where our own installer creates `AddaxAI_files` just to hold the Timelapse shim.

**The Windows shim exception.** `electron/build/installer.nsh` writes a Timelapse launcher to `%USERPROFILE%\AddaxAI_files\AddaxAI\open.bat`, inside the legacy install root. Timelapse still looks for that path, so the purge deletes everything under `AddaxAI_files` **except** that one file. Do not "simplify" this into deleting the whole root.

**Junctions.** NSIS `RMDir /r` follows a junction and deletes through it. Python's `shutil.rmtree` has not followed junctions since 3.8 but raises on a top-level one, so `legacy_install._remove` detects a junction and calls `os.rmdir()`, which drops the reparse point and leaves its target alone. `os.path.isjunction()` is not usable: the frozen build runs Python 3.11 and that helper landed in 3.12.

**Desktop entries are only touched during removal, never during the scan.** On macOS, reading `~/Desktop` triggers a permission prompt for a non-sandboxed app. The scan runs on every launch, so scanning the Desktop would prompt every user including those who never had legacy installed.

**Failure handling.** There is no pre-flight "is legacy running" check. The purge runs, then `remove()` re-checks the marker and returns any surviving paths, and the UI says to close the old app and retry. One rule that covers a running legacy app, antivirus locks and open file managers, on every platform, with no extra dependency.

**The disk-space dead end.** Setup refuses with a 507 below 7 GB free, and the removal prompt only appears once setup has finished. A user with a 15 GB legacy install and a nearly-full disk could therefore never reach the thing that would free the space. `_legacy_disk_hint()` in `routers/setup.py` appends the legacy path to that error so they can delete it by hand. Keep the hint best-effort: it must never turn a clear 507 into a 500.

**Not covered:** installs moved by hand to a custom path (legacy's docs sanction moving to Program Files; EcoAssist 4.x let users type any path). The Program Files copy is reported so the user can delete it, everything else is out of scope. No disk scan.

| File | Purpose |
|---|---|
| `backend/app/services/legacy_install.py` | Path table, detection, purge |
| `backend/app/utils/fs_remove.py` | `safe_rmtree`, shared with the reset flow |
| `backend/app/api/routers/setup.py` | `/api/setup/legacy-install` and `.../remove` |
| `frontend/src/components/diagnostics/RemoveLegacyDialog.tsx` | The dialog |
| `frontend/src/components/layout/MenuCommands.tsx` | Auto-prompt, menu command, dismissal flag |

The junction branch cannot be tested on the Linux and macOS CI runners, so it is verified by hand on Windows.

## Linting (CI enforcement)

GitHub Actions runs **ruff** on every push and PR (`ruff check app tests`). The build fails if there are any errors, so check locally before pushing:

```bash
cd backend
ruff check app tests          # check only
ruff check app tests --fix    # auto-fix import sorting (I001) and unused imports (F401)
```

**Common pitfalls that CI catches:**

| Rule | What it means | How to fix |
|------|---------------|------------|
| **E501** | Line exceeds 100 characters | Break the line: wrap args, use intermediate variables, etc. |
| **I001** | Imports not sorted | Run `ruff check --fix` (auto-fixable) |
| **F401** | Unused import | Remove it, or run `ruff check --fix` |
| **F841** | Variable assigned but never used | Remove the assignment |
| **B904** | `raise` inside `except` without `from` | Use `raise ... from err` or `raise ... from None` |

The max line length is **100 characters** (configured in `pyproject.toml`). This is the #1 source of CI failures, so keep lines short.

## Testing

Backend tests use **pytest** with an in-memory SQLite database. Each test gets a fresh DB session that rolls back after the test, so tests are fully isolated.

```bash
cd backend
pytest                        # run all tests
pytest tests/api/             # run only API tests
pytest tests/ml/              # run only ML/taxonomy tests
pytest tests/integration/     # run integration tests
pytest -x                     # stop on first failure
pytest -k "test_label_tree"   # run tests matching a name pattern
```

Coverage is collected automatically (`--cov=app` in `pyproject.toml`).

**Test structure:**

| Directory | What it tests |
|-----------|---------------|
| `tests/api/` | API endpoints via FastAPI `TestClient` |
| `tests/ml/` | ML utilities (taxonomy parsing, rollup, postprocessing) |
| `tests/integration/` | Multi-step pipelines (event generation, detection pipeline) |
| `tests/models/` | SQLAlchemy model constraints and relationships |
| `tests/` (root) | Standalone unit tests (scoring, websocket, etc.) |

**Writing tests:** Use the factory helpers in `tests/conftest.py` (`make_project`, `make_site`, `make_deployment`, `make_file`, `make_detection`, `make_event_with_files`) to build test data. Use the `client` fixture for API tests and the `db` fixture for direct DB tests.

### Electron end-to-end tests

`electron/tests/` holds Playwright tests that launch the real app, which spawns the real backend against a real database.

```bash
cd electron
nvm use 20
npm run test:e2e
```

They exist for the startup error page, which has no unit-testable seam: the backend refuses a database and exits *before* the API or the frontend exist, so the whole recovery path lives in the main process (a file the backend writes, a page rendered from it, two IPC handlers writing marker files the backend consumes next launch). Only running the app proves those line up.

Two settings exist so they can run in isolation, and both are worth knowing about outside tests too:

- `USER_DATA_DIR` is honoured by the Electron side as well as the backend, so the whole app can point at a throwaway directory. Every path in `main.ts` derives from it. It has to: the two processes talk to each other through files in there, so a value they disagree on means markers land where nothing reads them.
- `ADDAXAI_BACKEND_PORT` moves the backend off 8000. The app kills any AddaxAI backend already holding its port, so without this a test run would kill your dev server.

Native dialogs cannot be driven from Playwright, so `dialog.showOpenDialog` / `showMessageBox` are stubbed via `electronApp.evaluate()`, along with `app.relaunch` (which would otherwise leave a second app running). What is under test is the wiring from button to marker file, not Electron's own APIs.

Note that `npm run build` only typechecks `src/`; Playwright transpiles the specs without typechecking them.

## Detection threshold and verified override

Three confidence values exist and must not be confused:

1. **MD output** — MegaDetector always runs untresholded (`MD_OUTPUT_CONFIDENCE_THRESHOLD = 0.005`, MD's own internal default). Everything above it is stored: raw results.json, database, and the folder-run data exports (which bypass all thresholds by design).

All confidence defaults live in `backend/app/core/confidence.py`, mirrored by `frontend/src/lib/confidence.ts` — change them there, never as literals at call sites.
2. **`Project.classification_gate`** (default 0.1) — detection confidence above which animal crops are classified and embedded. Inference-time; changing it applies to new analyses. Gating both per-crop model passes is what keeps the untresholded MD output from multiplying compute.
3. **`Project.counting_threshold`** (default 0.2) — the counting/visualization filter described below. Folder runs pin it to the classification gate.

Detections below `counting_threshold` are hidden from the UI. However, verified detections always pass, regardless of confidence. A human verification is a stronger signal than a model score.

**The rule:** anywhere you query detections and the result is user-facing, apply:

```python
or_(Detection.confidence >= threshold, Detection.verified == True)
```

This must be applied consistently across every module that counts, lists, filters, or displays detections. The places where this is currently enforced:

| Module | What it covers |
|--------|---------------|
| `crud/statistics.py` | Dashboard stats (overview, species, activity, trend, categories) |
| `crud/label_tree.py` | Label filter tree counts (detection and event modes) |
| `crud/event.py` | Event label filter, standalone confidence filter, verification stats, filter options |
| `crud/project.py` | Project card detection counts (single and bulk) |
| `routers/projects.py` | Detection count, label stats, category stats, independent event stats |
| `ml/inference/similarity_script.py` | Similarity sort/search (raw SQL) |

**When adding a new query that touches detections**, check whether the result is user-facing. If yes, apply the threshold with the verified override. If you skip this, detection counts and filter options will be inconsistent with what the user sees in the verification grid.

**Two exceptions where `OR verified` does not apply:**
1. **User-driven confidence range filters** (e.g. a max_confidence ceiling). When a user explicitly sets a confidence range, respect it literally. The override only applies to the project's threshold floor, not to user-specified ceilings.
2. **Per-file detection lists** (`crud/detection.py`). These serve the file detail view where the caller controls what to show. Not tied to the project threshold.

**Common mistake:** writing `Detection.confidence >= threshold` without `OR Detection.verified == True`. This silently drops verified low-confidence detections from counts, filters, and charts. The result is that users see different numbers on different pages.

## Non-label detection skip

MegaDetector sometimes produces false positive bounding boxes. When a classification model (SpeciesNet or custom) classifies a detection as one of the non-label classes, the detection is not loaded to the database at all. This keeps false positives out of counts, filters, and the verification UI.

**Non-label classes** (defined in `backend/app/ml/label_exclusion.py`): `bait`, `blank`, `empty`, `false detection`, `none`, `vide` (French for empty). These are always stripped, regardless of project settings.

**The rule:** a detection is skipped when its **raw top-1** classification is a non-label class. That is `should_skip_detection`, and it is what the DB load calls (`json_pipeline.py`, gated on `category == "animal"`). Detections with no classifier output (unclassified animals) are still loaded with `label=NULL`. Person and vehicle detections are never classified and are always loaded.

Do not confuse it with `is_non_label_detection` in the same module, which skips only when *every* remaining classification has been filtered out. That one is legacy: nothing in `app/` calls it, only the unit tests do. The distinction matters because the JSON keeps the top 5 classifications per detection, so "all filtered out" is a far rarer condition than "top-1 is blank", and reading the wrong function gives you the wrong mental model of what reaches the database.

User species exclusion is a separate path: `apply_label_exclusion_to_results` in postprocessing, which builds its excluded set from the non-label classes plus the project's `excluded_classes`.

**Observation type:** files where all detections were skipped get `observation_type="blank"`. They will not appear in the verification grid and will be counted as blank images on the dashboard.

**Raw JSON preservation:** the JSON on disk (`results.json`) is never modified. It contains all original detections including those classified as blank. The skip only applies during the in-memory DB load step.

**Key files:**

| File | What it does |
|------|-------------|
| `backend/app/ml/label_exclusion.py` | `NON_LABEL_CLASSES` set, `is_non_label_detection()` helper |
| `backend/app/ml/json_pipeline.py` | Skip logic in `load_json_to_database()` and `_load_to_database()` |

## What a file is about

One rule, everywhere:

> A file is its **single strongest passing detection**. Strongest is verified first, then detector confidence. `File.observation_type` is that detection's raw category; the file's folder is that detection's species if it has one, else its category. Nothing passes, the file is `blank`.

`backend/app/ml/observation_type.py` is the only implementation, and it is two functions: `strongest_passing_detection` picks the box, `derive_observation_type` reads that box's category. Anything needing another attribute of the deciding box calls the first one rather than re-deriving the ordering. The Files export does exactly that, carrying `detection_confidence` / `classification_label` / `classification_confidence` / the five taxon ranks / `scientific_name` / `common_name` off the same box `observation_type` came from. Note the consequence for `detection_confidence`: because the ordering puts verified first, it is the deciding box's score and not the file's highest, so it can sit below the project threshold and filtering a CSV on it drops verified files. Documented in `docs/docs/reference/exports.md` rather than solved with another column. The rule knows no category vocabulary, needs no classifier, and works for any detector.

**For a video, "its detections" means the best frame's.** The module itself is frame-blind: every caller passes it the file's *visible surface* first, via `on_visible_frame()` / `on_visible_frame_of()` in a query or `visible_detections(file, dets)` on a list already in memory. So one sentence covers a video everywhere: AddaxAI saves one frame per video, and every still surface and every summary of that video comes from that frame. (`VideoPlayer` still draws every frame's boxes on the real video, deliberately, which is why the sentence says "still surface" and not "everything".) Consequences worth knowing: a video whose best frame holds nothing passing reads `blank` even when a confident box sits on another frame, and a video with no `best_frame_number` at all has no visible surface, so only a verified box can speak for it. Both are the honest answer, because those boxes have no card in the Labels grid, no MaxN count and no crop. They are still in `detections.csv` and the recognition JSON. Migration `6f7a8b9c0d1e` backfilled existing videos; it is scoped to `file_type = 'video'` so an image, whose `frame_number` and `best_frame_number` are both NULL, can never be touched by the comparison.

**An event uses a different rule, on purpose. Do not unify them.** `build_event_primary_labels` in `postprocessing_outputs/separate_folders.py` picks a burst's species by the *most common* verified species, not by the strongest single box. That is not drift, and "fixing" it in either direction breaks something.

A file is one photograph: one look at the animal, nothing to average, so the strongest box is the whole of the evidence. An event is dozens of looks at the same animal walking through, and per-frame classification is noisy: one raccoon crossing a clip read as northern raccoon, american badger, american badger, blank, virginia opossum, northern raccoon, blank on seven consecutive frames (see "The best frame is the only frame a video detection can be shown on"). Taking the mode across those frames cancels that out. Taking the strongest box would let one lucky misread frame name the whole visit, on the surface the user actually sees, the folder on disk.

The reverse is worse. A file usually has one to three boxes, so a mode over them is close to a coin flip, and it would destroy the guarantee that `observation_type`, the label, the five ranks and the two names all come from one box, because you would have to pick a representative box for the taxonomy anyway.

The two never meet today: the event label is used only by `separate_folders` and its preview, only when `group_events` is on, and reaches no export. **If an event-level label is ever added to an export, revisit this**, because then two rules for "what is this about" sit side by side in one file and the difference has to be explained to users rather than only to maintainers.

**Why not a category priority.** Until 2026-07-31 this ranked categories instead (animal > human > vehicle), so one animal box at 0.21 beat thirty person boxes at 0.95. A test clip of a person in camouflage inspecting a camera produced 31 person boxes at 0.65 to 0.95 and one false-positive animal box that SpeciesNet called "chimpanzee" at 29%. Priority made the file an animal, that lone box was then the only labelled detection so it named the folder, and the run wrote `addaxai-media/chimpanzee/IMG_0001_still.jpg` containing a picture correctly labelled `Person 73%`. Ranking categories cannot be right when the thing being ranked is the detector's own guess about the category.

**The category is the detector's, and is never translated.** `Detection.category` and `File.observation_type` carry whatever the run's own `detection_categories` map said: `animal` / `person` / `vehicle` from MegaDetector, `shark` / `fish` / `turtle` from a detector that emits those. `json_pipeline` reads that map rather than assuming, and **refuses an id the run never declared** instead of defaulting it to `animal`, which is what silently turned every class of a non-MegaDetector model into wildlife.

**The one translation is Camtrap DP.** `observationType` there has a fixed controlled vocabulary (`animal`, `human`, `vehicle`, `blank`, `unknown`, `unclassified`) defined by the standard, not by us. `_obs_type_from_category` in `crud/export.py` is the only place a category is converted: `person` becomes `human`, and anything that is not a person, vehicle or blank becomes `animal`, which is where Camtrap DP puts all wildlife with the species in `scientificName`. Emitting a raw `shark` there would fail validation in the `camtrapdp` R package and in GBIF ingestion.

**Do not add a fifth category map.** There were four, all subtly different, and one Pydantic `Literal` that rejected any other detector's output at the schema boundary before the code that would have handled it ran. They are now one function plus the Camtrap boundary. If you find yourself writing `if category == "animal"`, ask whether you mean "is this the strongest detection" or "is this wildlife", and reuse the existing helper for the first.

`observation_type` is denormalised, so it is recomputed at ingest, after postprocessing, on any detection edit, and on a project threshold change. A rule change therefore needs a data migration; `5e6f7a8b9c0d` is the worked example, with its data test in `tests/db/test_migration_data.py`.

## Datetime conventions

There are two kinds of datetimes in this codebase and they must never be mixed in arithmetic or comparison:

1. **Observational datetimes** are naive wall-clock time at the camera, in the project's local camera timezone (`Project.timezone`). They come from EXIF `DateTimeOriginal` on images or exiftool `QuickTime:CreateDate` and friends on videos, and are stored verbatim: no conversion, no offset attached at rest. Columns: `File.captured_at_local`, `Event.event_start_local`, `Event.event_end_local`, `Deployment.start_date_local` (Date), `Deployment.end_date_local` (Date).

2. **Audit datetimes** are tz-aware UTC, written via `datetime.now(UTC)` (never the deprecated `datetime.utcnow()`). They record when the server did something: row creation, job scheduling, human verification, folder re-link. Columns all end in `_utc` and are typed `DateTime(timezone=True)`, e.g. `Project.created_at_utc`, `Project.updated_at_utc`, `File.verified_at_utc`, `Detection.verified_at_utc`, `Job.started_at_utc`.

| Type | Naming | SQL type | Write site | Tz |
|------|--------|----------|------------|-----|
| Observational | `*_local` (or `*_date_local`) | `DateTime` / `Date` | EXIF / exiftool / event clustering | camera local, naive |
| Audit | `*_utc` | `DateTime(timezone=True)` | `datetime.now(UTC)` | UTC, tz-aware |

**Wire format.** Observational datetimes are serialized to the frontend as ISO 8601 with the UTC offset that applies to the project's timezone on *that file's local date* (so DST is resolved per-file). The field serializer lives in `backend/app/api/schemas/file.py` and `backend/app/api/schemas/event.py`; it reads the active project timezone from a `ContextVar` set at the top of each endpoint. The helper `app.utils.datetime_serialization.to_local_iso_with_offset` does the actual formatting. Audit datetimes serialize naturally because they're already tz-aware.

**Endpoints that return observational datetimes must be `async def`.** The `ContextVar` is set inside the endpoint body; for sync endpoints FastAPI runs the body in a threadpool, and the ContextVar set there is invisible to the response serialization stage that runs in the event loop task. `async def` keeps the body and serialization in the same task context. Pinned by tests in `backend/tests/api/test_datetime_wire_format.py`.

**Frontend rendering.** The UI must always show the camera's wall-clock time, not the viewer's browser-local time. `new Date(iso).toLocaleString(...)` parses the ISO string to an absolute UTC moment and then converts to the *viewer's* timezone, which silently shows wrong hours for any user not in the project's tz. Use `formatCameraDate` / `formatCameraTime` / `formatCameraDateTime` from `frontend/src/lib/datetime.ts` instead. They strip the offset and render the local components verbatim (locale-aware via `Intl.DateTimeFormat` with `timeZone: "UTC"` after the strip).

**Missing capture time is tolerated, not fatal.** `backend/app/ml/json_pipeline.py:load_json_to_database` reads each image's `DateTimeOriginal` (images) or exiftool `QuickTime:CreateDate` (videos). A file with no extractable timestamp is ingested with `captured_at_local=NULL` and recorded in `PipelineResult.skipped_missing_timestamp`; the worker logs the count and the job still succeeds. These files drop out of time-based features (events, trap nights, activity, trend charts) but are still detected and classified. This holds in both folder-run and project mode; neither rejects a deployment for missing dates. We never substitute `datetime.now(UTC)`, and we never reach for `fromtimestamp(mtime)` on our own: both silently lie about when the animal was actually there.

**The one mtime exception is explicit, per folder, and never silent.** Some cameras write no capture date anywhere. A Browning MJPG AVI is the worked example: the RIFF container has no `LIST INFO` chunk, so no `IDIT` or `ICRD`, nothing is embedded in the frames, and the date exists only burned into the pixels and in the filesystem timestamp. On the SD card that timestamp is right. Copied, it is not: the same test file read `09:55` on the card and `15:55:26 CEST` after a transfer, and a plain `cp` reset it to the day of the copy. Only the person holding the card can tell which, so it is their decision and not ours.

`DeploymentQueue.use_file_mtime_fallback` records that decision. The rules, all of which matter:

- **Offered only when there is no alternative.** The folder scan surfaces the checkbox only when `missing_datetime` is true, i.e. it found no capture date at all.
- **The user sees the result first.** `scan_folder` computes `mtime_start_date` / `mtime_end_date` over *every* media file (a `stat()` is not an EXIF decode) and only when the metadata pass came back empty, so those two fields are non-null exactly when the opt-in is shown. That displayed range is the whole safeguard: there is no heuristic behind it, and a folder copied last week reads as this week.
- **It fills gaps, never overrides.** `_resolve_capture_timestamp` puts mtime dead last, after the `addaxai-` filename marker. That ordering is load-bearing, not stylistic: `file_mtime_datetime` succeeds for every readable file, so anywhere earlier it would shadow every source below it.
- **One helper, three call sites.** `app/utils/media_dates.file_mtime_datetime` is used by the scanner, by `GET /api/deployments/file-datetime`, and by the ingest. The probe endpoint matters more than it looks: without it the Adjust-dates modal shows "unknown" for every file in such a folder and the offset can never be worked out.
- **The clock is the user's computer, not `Project.timezone`.** `fromtimestamp` gives the naive local time the OS file browser shows, which is exactly what the preview showed before they ticked the box. The preview endpoint takes a path and knows nothing about a project, so it could not do anything else even if we wanted to. A user whose cameras ran on another clock corrects the whole-hour shift with `datetime_offset_seconds`, which applies downstream of resolution and so lands on these values for free.
- **Nothing records that a date came this way.** Deliberate, and the main cost of the design: once ingested these are indistinguishable from camera dates. The only trace is one `logger.info` counting them per run, which is why that line stays even though it looks redundant.

This does not reinstate what commit `b9f23739` removed. That was an automatic mtime path plus a three-hour minimum-span heuristic guessing whether to trust it. The rule here is "never mtime unless the user looked at the result and said yes".

Fix the source data instead where you can. camtrapR's `fixDateTimeOriginal` writes a real `DateTimeOriginal` with exiftool and is the better answer, except exiftool refuses to write AVI at all, which is what forced this.

**Project timezone.** `Project.timezone` is a required IANA string (`"Europe/Amsterdam"`, `"UTC"`, `"Etc/GMT-3"`, etc.) describing what clock the cameras were configured to. The `TimezoneSelect` combobox in `frontend/src/components/ui/timezone-select.tsx` exposes both DST-aware regional zones and fixed-offset `Etc/GMT±N` zones for cameras set to "local winter time" (no DST). Used by the activity-pattern sun overlay (astral) and any future camtrap-dp export. Never used to convert stored datetimes.

## Insights (in-depth analytical views)

The Insights section in the sidebar hosts page-wide, scientifically-grounded visualisations that go deeper than the Dashboard's glanceable summary. Each view is its own route under `/projects/:id/insights/...` with its own filter bar and URL state persistence (via `frontend/src/lib/filter-url.ts`). The parent `insights` path redirects to the first child (`insights/map`) so clicking the parent does something useful. Future views slot in next to the existing ones.

### Map

`/projects/:id/insights/map`: per-deployment observation rate per 100 trap nights plotted on a base map. Three spatial views: trap-night-normalised rate, absolute observation count, and a heat-style density layer.

### Activity overlap

`/projects/:id/insights/activity-overlap`: 1- or 2-species temporal activity comparison modelled on the R `overlap` and `activity` packages. Two `SpeciesPicker` dropdowns drive the chart; with two species selected, the page also renders the Ridout & Linkie 2009 overlap coefficient Δ with a 1000-rep percentile bootstrap CI.

The math lives server-side in `backend/app/ml/activity_analysis.py` so it can be unit-tested in isolation:
- `fit_circular_kde()`: von Mises kernel density on a 240-point grid over [0, 24) hours. Post-normalized numerically rather than computing the closed-form `1/(2π·I₀(κ))`, which avoids a scipy dependency.
- `overlap_coefficient()`: Δ = ∫ min(f_a, f_b) dt over the grid.
- `bootstrap_overlap_ci()`: 1000-rep percentile bootstrap, fixed seed for deterministic results.
- `classify_diel()`: Bennie et al. 2014 ≥ 0.70 density-in-phase rule (diurnal / nocturnal / crepuscular / cathemeral). Falls back to a fixed 06:00-18:00 day window when sun bands are unavailable (polar latitudes, missing tz).

Sun-time mode (Vazquez et al. 2019 double-anchored transform) lives in `backend/app/ml/sun_time.py`:
- `per_date_sun_phases()`: looks up `(dawn, sunrise, sunset, dusk)` per unique observation date via astral, using the project's IANA timezone. Works for DST zones (`Europe/Amsterdam`) and fixed-offset zones (`UTC`, `Etc/GMT-3`) alike. Returns `None` for polar dates that astral refuses.
- `compute_anchors()` / `compute_anchor_bands()`: mean sunrise / sunset (and dawn / dusk) across every non-polar observation date. Both species share the same anchors so the two KDE curves sit in the same frame.
- `transform_to_sun_time()`: piecewise-linear double-anchor mapping: per-day sunrise / sunset stretch or compress to the anchor sunrise / sunset, with daylight and nighttime portions handled separately. Observations on polar dates are dropped and counted separately so the UI can surface the loss.

The CRUD function `get_activity_overlap()` in `backend/app/api/crud/statistics.py` reuses `_avg_site_location()` and `_compute_sun_bands()` from the existing activity-pattern endpoint, applies the project's `independence_interval` to event grouping (no per-plot override: the interval is shown as read-only metadata next to the chart), and returns one round-trip with both species' KDE densities, the rug-tick samples, the Δ block, the diel classifications, the clock-mode sun bands, the sun-mode anchor bands, and the effective `time_axis` (which silently downgrades to `clock` if the project has no site coordinates or every observation date is polar). Wire format and tz handling follow the conventions in the "Datetime conventions" section below.

References that motivate the design choices: Ridout & Linkie 2009 (J Agric Biol Environ Stat), Meredith & Ridout's `overlap` R package vignette 2014, Rowcliffe et al. 2014 (MEE), Vazquez et al. 2019 (MEE), Lashley et al. 2018 (Sci Rep), Bennie et al. 2014.

## Best frame selection (videos)

After video detection (phase 1) and frame extraction, a single representative frame number is selected per video. The algorithm:

1. Score each frame by summing **every** detection's confidence (>= 0.3), whatever its category
2. Among confidence ties, prefer the largest union bbox area (within 90% of the best)
3. Blank videos, and videos whose detections are all below 0.3: the middle frame

**The decision reads the JSON only, and happens before anything is decoded.** `scoring.choose_frame_number` takes the detection list plus the frame count and returns a number. Ordering it this way is what keeps memory flat, and it is why there is no sharpness tier.

**How many frames you want decides how you fetch them.** Measured on real camera-trap clips: a seek costs ~85 ms, a sequentially walked frame ~1.6 ms, so **one seek is worth about 55 walked frames**. That single ratio is the whole rule, and it splits the two callers:

| Caller | Frames wanted | How |
|---|---|---|
| `best_frame.py`, and the classifier when a clip has nothing to crop | 1 | `read_frame_by_seek` |
| the classifier with crops (11 to 51 frames), the filmstrip (9) | many | `iter_wanted_frames`, walking |

Until 2026-08-03 everything walked, and `best_frame.py` carried a comment claiming it decoded "only the frame we are going to keep". That was false: it decoded every frame *up to* the one it kept. Since a video with no confident detection is sent to `total_frames // 2`, and blank is the majority of clips, it decoded **half of every empty video and discarded it**. On a beta tester's 1595-video run that was 85 minutes, 71% of the MegaDetector video detection time itself. This was invisible for two months because the May refactor (bulk frames to disk removed) and the July one (sharpness tier dropped, `wanted` shrank to two frames) both really did fix something, disk and memory respectively, and the comments they left described *retention* in language that reads like *work*.

**Seeking is verified, never trusted.** `read_frame_by_seek` refuses unless it can prove where it landed, and the caller then walks instead. Two gates, and the order matters: the range check (`0 <= n < frame_count`) decides from the container's own count before any backend is involved, so an out-of-range frame is refused without depending on how honestly a backend clamps; the `CAP_PROP_POS_FRAMES` check afterwards is the weaker one, because FFmpeg's seek sets that counter to whatever you asked for, so it mostly only catches a backend that does not implement the property at all. The consequence to hold onto: **a codec that seeks badly loses the speed-up, it never gets the wrong frame.** What neither gate catches is variable frame rate, where a seek targets a timestamp while MegaDetector numbers frames by counting decoded ones. Camera traps are effectively all constant frame rate; `backend/scripts/check_seek_accuracy.py` compares seeked pixels against walked pixels over a folder of real footage and is how a new camera make gets checked before it is trusted.

**This step must never run on the event loop.** It is minutes of work on a large folder and it used to be a plain synchronous call inside `_process_batch_job`, so the backend answered no HTTP request for the whole 85 minutes and the progress websocket could not report the very step blocking it. That is why it read as a hang rather than as a slow step. It is now `asyncio.to_thread` with a `video_frame_selection` progress phase and a per-video cancel check, and the `except Exception` around it re-raises `JobCancelledError` so a cancel is not swallowed as a non-fatal failure.

**The phase reports no compute device, on purpose.** Every other phase announces one because a model is loading somewhere; this one is CPU-bound video decoding sitting between two GPU phases, and a lone "CPU" row reads as a fault rather than as the ordinary fact that decoding is not GPU work. It is also why `send_progress` only carries `compute_device` forward *within* a phase and `useTaskProgress` clears it when a message omits one: both used to latch, so this phase inherited the detector's "GPU" and claimed hardware it never touched.

There was one, Laplacian variance, sitting below the area tier and also deciding blank videos outright. It required the pixels of every candidate frame, so the caller had to decode and hold each one before it could know which single frame it wanted: tens of full-size images in RAM for a 30-second clip. Measured across real deployments the tier never once broke a tie, because summed detection confidence decided every video that had detections at all. Paying that to settle a tiebreak that does not fire is a bad trade. For blank videos the middle frame is as defensible as the sharpest of three arbitrary samples, and beats the first frame, which is often the empty scene that triggered the camera.

**`best_frame_number` and the JPEG on disk must always describe the same frame.** Deciding before decoding means the chosen frame might never arrive, because containers over-report their frame count. When that happens both callers fall back to frame 0 *and move the stamped number with it*. This is also why the seek is verified rather than assumed, and why the tests assert the written JPEG's pixel content and not just its filename: a filename-only assertion passes happily when the frame came from the wrong place. Stamping the frame you wanted while writing the frame you got is the same class of bug as the crop-service one above: the Labels grid would draw one moment's boxes over another moment's picture.

See `backend/app/ml/best_frame.py` and `scoring.choose_frame_number`.

**Score on detection confidence, never on category and never on classification confidence.** Detection confidence is the only signal present in every detector and classifier combination, so one rule covers all of them, including detectors whose categories are not animal/person/vehicle (`fish`, `shark`, `turtle` need no code change). Two dead ends:

- *Scoring only animals* is what the classifier-fused path did until 2026-07-31. It took its candidates from the classification input, which `extract_animal_detections` restricts to category `"1"` above the classification gate. A clip containing only people therefore scored nothing, fell through to the blank-video fallback, and picked a frame with no idea where the person was. Since the Labels grid only shows best-frame detections, such a video could show no cards at all. It also made the chosen frame depend on whether a classifier was configured, because `best_frame.py` always scored every category.
- *Averaging detection and classification confidence* looks like the general rule and is not. It is really two rules, since a classifier-off run has no second number to average, so the same footage would get a different frame depending on a setting unrelated to how the frame looks. Classification confidence also has no fixed scale: SpeciesNet spreads mass over ~2500 classes and returns top-1 around 0.3 to 0.7 where a 20-class model returns 0.9+. Person and vehicle are never classified at all, so they would compete on a different scale within one frame. And a low classification score usually means two similar species, not a bad picture, which is the opposite of what best-frame is asking.

Both code paths (`best_frame.py` for classifier-off runs, `classification_worker._process_video_group` for classifier-on) score the same population through the same `choose_frame_number`, so they pick the same frame for the same video. The worker receives the population as `scoring_detections`, separate from the `items` it classifies. `tests/ml/test_best_frame_scoring.py` pins all of this, including that the two paths agree.

**A consequence worth knowing:** in a clip holding both, a confident person or vehicle can now win the frame over a marginal animal. Animals used to win by construction, because nothing else was scored.

That reads like a cost and the first real example was the opposite. A test clip of a person in camouflage inspecting a camera produced 31 person detections at 0.65 to 0.95 and exactly one animal box at 0.677 on a single frame, which the classifier then labelled "chimpanzee" at 0.29. Scoring animals only made that lone false positive the whole video: best frame 150, observation "chimpanzee". Scoring every category picked frame 420 and reported "person", which is what the video shows.

Weighting one category above the rest does not encode "animals matter more", it encodes "trust one detector output over its others". When the detector is wrong about the category, that is precisely backwards, and it is why re-introducing a category preference is not the safe-looking option it appears to be.

**Changing this needs a re-analysis to take effect.** `best_frame_number` is written once at load time, so existing deployments keep whatever frame the old rule chose. (Ingest does refresh it when a JSON is re-loaded onto an existing `File` row, so the stored frame never disagrees with the detections sitting beside it.)

**Changing the scoring here now changes `File.observation_type` too.** Since `6f7a8b9c0d1e` a video is summarised by its best frame, so the frame this function picks decides what the file is, not just which JPEG is shown. `CONFIDENCE_THRESHOLD = 0.3` in `scoring.py` is the reason a video only reads `blank` on a project threshold above roughly 0.6: the chosen frame carries the confidence mass. Move that floor and observation types move with it, silently, for new analyses only, with no migration to catch it.

**Storage:** No separate frame JPEG is saved; `best_frame_path` points to the frame inside `video_frames/`: `{deployment_folder}/.addaxai/video_frames/{video_name}/frame{N:06d}.jpg`. The `files` table stores `best_frame_number` (0-based index) and `best_frame_path` (absolute path to the JPEG). Both are `NULL` for images.

**Usage:** The best frame is the canonical image representation of a video. Use it anywhere you'd use a photo for an image file:
- Thumbnails in the UI
- Human verification workflows
- Depth estimation
- Any future per-file visual feature

If you're building a feature that works on images, check `file.best_frame_path` for videos instead of extracting frames yourself.

### The best frame is the only frame a video detection can be shown on

MegaDetector runs over every sampled frame, so a 30-second clip stores dozens of detections spread across dozens of frames. Only the best frame is written to disk as a JPEG. That makes one rule non-negotiable:

> a video detection is displayable only when `Detection.frame_number == File.best_frame_number`

Break it and nothing errors. `crop_service` happily crops the best frame at a bbox belonging to a different moment and returns a perfectly valid JPEG of the wrong place. On a clip of a person walking, 31 of 32 grid tiles were pictures of the background they had already left. The bug survived for months because a slow subject hides it completely: successive bboxes overlap, so the crops still contain the animal, just shifted, and they look right. Only a moving subject exposes it. If you are ever unsure whether a surface honours this, test it with a clip of something walking across the frame, never with a browsing deer.

Enforced in:

| Module | How |
|--------|-----|
| `services/crop_service.py` | `_resolve_image_path` returns `None` off the best frame |
| `ml/inference/similarity_script.py` | `_COMMON_JOINS` gates all three grid queries |
| `ml/embedding_utils.py` | `build_embedding_input` skips them (no pixels to embed) |
| `api/routers/labels.py` | `on_embeddable_surface` in the stats and unprocessed counts |
| `ml/postprocessing_outputs/annotated_copies.py` | one annotated still per video |
| `api/crud/event_observation.py` | off-best-frame species don't spawn observation rows |
| `frontend/src/lib/detection-utils.ts` | `shouldDrawBbox` for every canvas and overlay |
| `api/crud/file.py`, `ml/json_pipeline.py`, `ml/postprocessing.py` | the three writers of `File.observation_type` |
| `api/crud/export.py`, `ml/postprocessing_outputs/{separate_folders,output_preview}.py` | the Files export species block, the folder a video is copied into, and its preview |

**Verified detections are the one exception.** They pass on any frame, in the grid query and in `rebuild_event_observations`. A human decision must never end up out of reach: the counts already honour a species verified on some frame, so the grid has to be able to show the card that count came from. Its thumbnail will be missing, which is the honest answer, and `CropCard`'s `onError` degrades to a plain tile.

**The off-best-frame detections are not junk, and are not deleted.** Their per-frame classifications are exactly what smoothing and taxonomic rollup consume, and they disagree a lot: one raccoon walking through a clip was read as northern raccoon, american badger, american badger, blank, virginia opossum, northern raccoon, blank on seven consecutive frames. Rollup turning that into one label is the system working. They also draw in `VideoPlayer`, which has the real frames. They are hidden from still surfaces only.

## Keeping installed models up to date

A model that is already downloaded is never re-fetched by the normal flows: `check_weights_ready` only tests that the weights file exists, so once the `.pt` is on disk, preparing the model downloads nothing. That is fine for weights and wrong for everything else in the repo, because a fixed `inference.py` or a corrected `taxonomy.csv` would never reach anyone who already had the model.

On startup, `ModelCatalogUpdater.sync()` therefore asks HuggingFace for one file listing per **installed** model (a catalog stub with no weights next to it costs no request) and compares it to what is on disk. `find_stale_files` in `backend/app/ml/model_storage.py` is the whole rule:

- **LFS files are skipped.** Those are the weights. HuggingFace's `blob_id` for an LFS file is the hash of the pointer stub rather than of the content, so comparing it would report every install as stale forever, and hashing the real file means reading gigabytes on every launch. Weights are versioned by `model_id` instead: re-uploading weights in place under the same id will not be noticed, so bump the id (`EUR-DF-v1-1` through `v1-4`, `AFR-DFV-v1` to `v2`).
- **Documentation and OS litter are skipped** by basename: `README.md`, `LICENSE`, `LICENSE.md`, `.gitattributes`, `.DS_Store`, `manifest.json`.
- **Everything else is compared by git blob SHA-1**, which is exactly what `blob_id` holds for a non-LFS file, so nothing is downloaded to reach a verdict. That covers `inference.py`, `taxonomy.csv`, `taxon-mapping.csv`, class lists, geofence JSON, `hubconf.py` and the vendored `dinov2/` and `dinov3/` trees with no allowlist to maintain. Covering all of them rather than just the obvious two matters: commits routinely touch several files at once, and shipping half of one leaves new code reading an old data file.

`POST /api/ml/models/{id}/update` recomputes the stale set server-side (a client can never name a path) and downloads only those files, via the `include` and `overwrite` options on the HF downloader. `overwrite` is not decoration: the downloader otherwise skips any file whose byte size already matches, so an upstream edit of the same length would be skipped by the very call that came to replace it.

**Nothing about upstream is recorded on disk.** An earlier design stored an `hf_revision_sha` in the local `manifest.json`, which `write_manifest` then overwrote from the catalog on the next launch, so the check silently never fired and every model was logged as refreshed forever. Comparing the files themselves has no state to fall out of date, works on an install that predates the feature, and self-heals a partial download. Keep it that way: if you add a local-only key to `manifest.json`, you reintroduce that bug.

**`~/AddaxAI/models` is a managed cache, not a source tree.** Editing a model's `inference.py` in place now shows up as an available update on every launch, and applying it overwrites your edit with no backup.

## Creating a custom classification model

To add a new classification model to AddaxAI, create an `inference.py` file in your model's directory that implements the `ModelInference` class.

**Template:** See `/backend/templates/inference_template.py` for a complete template with examples.

**Required interface:**
```python
class ModelInference:
    def __init__(self, model_dir: Path, model_path: Path):
        # Store paths and initialize
        pass

    def check_gpu(self) -> bool:
        # Return True if GPU available
        pass

    def load_model(self) -> None:
        # Load model once at startup
        pass

    def get_crop(self, image: Image.Image, bbox: tuple[float, float, float, float]) -> Image.Image:
        # Crop and preprocess image for your model
        pass

    def get_classification(self, crop: Image.Image) -> list[tuple[str, float]]:
        # Return [(class_name, confidence), ...] for ALL classes
        pass

    def get_class_names(self) -> dict[str, str]:
        # Return {"1": "label1", "2": "label2", ...} (1-indexed)
        pass
```

**Benefits of class-based approach:**
- No global variables or `global` keyword needed
- Clear ownership (`self.model`)
- Framework-agnostic (works with PyTorch, Keras, JAX, TensorFlow, etc.)
- IDE autocomplete and type checking work properly

**Examples:**
- NAM-ADS-v1: YOLOv8 (PyTorch) - `/Users/peter/AddaxAI/models/cls/NAM-ADS-v1/inference.py`
- TAS-BB-v1: MEWC-Keras (Keras/JAX) - `/Users/peter/AddaxAI/models/cls/TAS-BB-v1/inference.py`

## Label taxonomy and the hierarchical filter tree

The label filter in the UI can render as either a flat multiselect or a hierarchical tree (class > order > family > genus > species). The tree is built from the `label_taxonomy` table. If no taxonomy rows exist for a project's classification model, the frontend falls back to the flat list.

### Database table: `label_taxonomy`

See `backend/app/models/label_taxonomy.py`.

| Column | Purpose |
|--------|---------|
| `classification_model_id` | Links to the classification model |
| `name` | Display label, **must match `Detection.label`** (this is the join key) |
| `taxon_class` .. `taxon_species` | Formal taxonomy ranks (nullable) |
| `level` | Most specific non-empty rank: `"class"`, `"order"`, `"family"`, `"genus"`, or `"species"` |
| `is_custom` | `True` for user-created entries, `False` for model-sourced entries |

Unique constraint: `(classification_model_id, name)`. All taxonomy functions are idempotent: calling them twice inserts 0 the second time.

`Detection.label` is a plain text field, **not** a foreign key. The tree builder matches it against `label_taxonomy.name` by string equality. This means the `name` value must exactly match whatever string ends up in `Detection.label`.

### How taxonomy gets populated

Taxonomy is populated automatically during two worker phases. All population functions live in `backend/app/ml/taxonomy_db.py`.

#### 1. Custom models with `taxonomy.csv`

Custom classification models (e.g. EUR-DF, NAM-ADS) ship a `taxonomy.csv` alongside their weights:

```csv
model_class,class,order,family,genus,species
leopard,mammalia,carnivora,felidae,panthera,pardus
bird,aves,,,,
```

`populate_taxonomy_from_csv(model_id, csv_path, db)` reads this file and inserts one row per line. The `model_class` column becomes `label_taxonomy.name`. Entries with only partial taxonomy (e.g. "bird" with just `class=aves`) get `level="class"`.

#### 2. Taxonomic rollup entries

When taxonomic rollup is enabled and a detection's top-1 confidence is below threshold, confidences are summed up the taxonomy tree. If a higher-level taxon (e.g. "felidae" at family level) crosses the threshold, `Detection.label` is set to that taxon name.

`add_rollup_taxonomy_entry(model_id, name, level, taxonomy_lookup, db)` inserts a new `label_taxonomy` row for the rolled-up label so it appears in the tree under the correct branch. Called from `backend/app/ml/postprocessing.py` for each new rolled-up label.

### Where population is triggered

Both workers call `populate_taxonomy_from_csv` when a `taxonomy.csv` exists in the model directory:

| Worker | When | Code location |
|--------|------|---------------|
| `detection_worker.py` | After loading results to DB (phase 6) | ~line 520 |
| `postprocessing_worker.py` | After reprocessing all deployments | ~line 174 |

```python
if taxonomy_csv.exists():
    populate_taxonomy_from_csv(model_id, taxonomy_csv, db)
```

The detection worker runs this once per deployment. The postprocessing worker runs it when reprocessing (e.g. after changing model or settings). Since all functions are idempotent, running them multiple times is safe.

### How the filter tree is built

`build_label_filter_tree()` in `backend/app/api/crud/label_tree.py`:

1. Queries which labels actually have detections in the project
2. Joins against `label_taxonomy` to get taxonomy columns
3. Builds the hierarchy: class > order > family > genus > species
4. Annotates each leaf with detection or event counts
5. Labels with no taxonomy match go under an `"__other__"` node
6. Returns `null` if no taxonomy rows exist (frontend shows flat list)

Exposed via `GET /api/events/label-tree?project_id=<id>&count_by=<event|detection>`.

### The `is_custom` flag

All model-sourced entries (CSV, JSON, rollup) set `is_custom=False`. The flag exists for UI-driven taxonomy creation where users can add custom labels with taxonomy info. Custom entries work identically in the tree builder: it queries all `label_taxonomy` rows for the model regardless of `is_custom`.

### Key files

| File | Purpose |
|------|---------|
| `backend/app/models/label_taxonomy.py` | SQLAlchemy model |
| `backend/app/ml/taxonomy_db.py` | Population functions (CSV, JSON, rollup) |
| `backend/app/ml/taxonomic_rollup.py` | Rollup algorithm (sums confidences up tree) |
| `backend/app/ml/postprocessing.py` | Orchestrates rollup + calls `add_rollup_taxonomy_entry` |
| `backend/app/api/crud/label_tree.py` | Builds the filter tree from `label_taxonomy` |
| `backend/app/ml/taxonomy_parser.py` | Parses CSV into a tree structure (used for validation, not DB) |
| `backend/tests/ml/test_taxonomy_db.py` | Tests for all population functions |
| `backend/tests/api/test_label_tree.py` | Tests for tree building + API endpoint |

### Rules

**No ad-hoc database fixes.** Do not run one-time scripts to patch database state. If data is stale or incorrect, fix the code that produces it. The data will be corrected when the user re-runs the relevant operation (analysis, reprocessing, taxonomy population). The app must handle its own data integrity.

**Never overwrite verified detections.** When a user manually verifies or relabels a detection (`Detection.verified == True`), that human judgment takes priority over any machine output. Postprocessing, reprocessing, taxonomic rollup, smoothing, and any other automatic pipeline must skip verified detections. If you are writing code that updates `Detection.label`, `Detection.label_confidence`, or `Detection.category`, always check `verified` first and leave verified records untouched.

This applies to the *reprocessing* path only. A full re-analysis (a folder-run rerun via `POST /api/folder-runs/{id}/rerun`, or re-analysing a deployment) deletes every detection under the parent and rebuilds from the JSON, so verified corrections do not survive it, by design: it is a "start over" operation, not an in-place update. The user-facing docs must not claim corrections survive a re-run of the AI; they survive threshold changes and reprocessing, not re-analysis. See "Deleting analysis data".
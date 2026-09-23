# Multi-user verification in AddaxAI: investigation and future plan

Status: investigation only, no code written.
Date: 2026-09-22
Branch the investigation ran on: `claude/repo-overview-kagsfq`
Repo state at time of audit: `5c4401d` on `main`, VERSION `0.0.0-dev`

This document is the raw material for a proper implementation plan. It holds the original
brief, the questions to answer, what the audit of the repo found, the three approaches that
were considered, why the simplest one won, the design that was agreed, the feasibility
assessment against the real code, the security and concurrency analysis, what happens when
somebody runs a second AddaxAI of their own alongside the shared one, what to deliberately
leave out, the risks, and the sources. It is written to be picked up cold months later
without rerunning the investigation.

It deliberately stops short of UI design and of writing any code. What matters here is which
constraints are real, which of them the codebase already satisfies, and which of them would
have to be worked around.

One framing point runs through the whole document and is worth stating up front. The request
sounds like it needs multi-user software. It does not. It needs one database that several
people can reach, which is a much smaller thing, and the app turns out to be about eighty
percent of the way there already without anyone having planned for it.

---

## 1. The original brief

A lab user wrote in, quoted verbatim as forwarded:

> I wanted to ask if you have any suggestions for implementing a multi-user workflow, and I
> realized I had already mentioned this in a previous email (below). You proposed a potential
> solution involving backing up the database to transfer between computers, while keeping the
> images and JSON files stored on an external drive.
>
> Do you think these steps are still the best approach with v7, or do you have any other
> suggestions? My overall goal is to have several lab members working on verification
> simultaneously (ideally on different camera folders, so we could keep track of progress
> easily). The processing step could either be done on one primary computer or on each
> computer separately; I imagine there are tradeoffs between each computer's GPU capabilities
> and the ease of merging results or databases afterwards. Mostly, I just want to make sure we
> will be able to combine our results cleanly before getting started.

The instruction given for the investigation, verbatim:

> What do you think? Is there a clean kiss dry yagni way to achieve this? Please investigate
> and propose three approaches that range from simple to complex. Do not code edit, only read.
> Do webqueries if needed. Be honest no sugar coating. A simple method that caters for all is
> the best

The scope then narrowed across several follow-ups, each quoted where it decided something:

> Lets focus on option 1.

> Can we somehow kiss create passwords and users ? Default to none, but possible if needed?

> And would this be something that users need to toggle when installing? During the install
> wizard? Or during runtime?

> Ok, if we implement it as we just discussed. Is it safe to?

> So you'd recommend against user specific passwords, right? And for verification just a user
> name drop down that users can select. Not a username login credentials combination, correct?

> Can multiple users verify at the same time?

> So basically a blocker on inference, right? So of inference is being run, show the pbars and
> block other functions (like it was designed for already on a single user)

> What would happen if a user would run inference on their local device while on the AddaxAI
> served from the main device?

That last question produced section 13, and two requirements that were missing from the
original ten:

> Yes, add it and include the instance indicator as R11. Inference can only run on the main
> device that is hosting the AddaxAi.

## 2. Questions to answer

1. Can several people verify at the same time, on different camera folders?
2. Can their results be combined cleanly, and does anything have to be merged at all?
3. Should analysis run on one machine or on each machine?
4. Is the database-shuttle approach from the earlier email still right for v7?
5. Can a password be added without building a user system?
6. Should there be per-user accounts, or something smaller?
7. Where does such a setting live: installer, wizard, or runtime?
8. Is any of this safe to expose on a network?
9. Does the app actually hold up with several simultaneous clients?
10. Should inference block everything while it runs?
11. What happens if somebody runs inference in their own local install at the same time?

## 3. Goals

What this is trying to achieve, in priority order.

**Several people verifying at the same time.** Simultaneously, not in turns. This is the one
the lab actually asked for and it is the one that eliminates most candidate designs.

**No merging.** Combining results is the stated worry. The best answer to a merge problem is
to not have one, so a design where results never diverge beats a design with good merge
tooling.

**Progress visible per folder.** So the lab can see who has done what without a spreadsheet
on the side.

**Nothing changes for the ordinary user.** The overwhelming majority of AddaxAI users run it
alone on a laptop. Any feature here must be invisible to them and off by default.

**Small enough to maintain.** One person maintains this repo. A login system with accounts,
roles, sessions and password resets is not a feature, it is a permanent obligation.

## 4. Repo audit: the lay of the land

### 4.1 The app is already a web app

This is the finding that makes the whole plan cheap, and nobody designed it in on purpose.

AddaxAI is a local web server plus a web page. The Electron window is a browser in a costume:
`electron/src/main.ts:139` calls `loadURL('http://localhost:8000/...')`. The backend serves
the built SPA from a catch-all route at `backend/app/main.py:628`, mounting hashed Vite assets
at `/assets` and returning `index.html` with `Cache-Control: no-store` for everything else.

The frontend resolves its own API base as `import.meta.env.VITE_API_URL || window.location.origin`
(`frontend/src/lib/api-client.ts:26`). The comment there explains why: a hardcoded
`127.0.0.1:8000` was baked in at build time and broke when the backend moved off port 8000.
The consequence, unintended but decisive, is that a browser pointed at
`http://some-machine:8000` automatically sends its API calls to that same host. No build
flag, no configuration, no code change.

Media travels over HTTP rather than `file://`. Images come from `/api/files/{id}/image`
(`backend/app/api/routers/files.py:277`), video from `/{file_id}/video`, filmstrips from
`/{file_id}/filmstrip`, and detection crops from `/api/detections/{id}/crop`. All of them
carry `Cache-Control: public, max-age=86400, immutable`
(`backend/app/api/routers/files.py:256`).

Electron-only features already guard themselves. `frontend/src/lib/platform.ts` exposes
`isElectron()` and `getPlatform(): 'electron' | 'browser'`, and call sites degrade quietly,
for example `useRevealInFolder` in `frontend/src/lib/file-reveal.ts` which returns early when
`window.electronAPI` is absent. There are 38 references to the Electron bridge across the
frontend, and they cluster in folder pickers, reveal-in-finder, backup and restore, and menu
commands. None of them are in the verification path in a way that breaks a browser.

So a plain browser on another machine is already a nearly complete AddaxAI client. What stops
it is one line.

### 4.2 The one line that stops it

`backend/run_server.py:43` passes `host="127.0.0.1"` as a literal. Meanwhile
`backend/app/core/config.py:76` declares:

```python
api_host: str = "127.0.0.1"
```

and a grep across the whole backend finds exactly one hit for `api_host`, which is that
declaration. The setting exists, is documented by its own prefix convention
(`ADDAXAI_API_HOST`), and is wired to nothing. The port next to it is correctly threaded
through: `run_server.py` reads `settings.api_port`, Electron passes `ADDAXAI_API_PORT` at
`electron/src/main.ts:489`, and there is a comment explaining that the port must come from
settings rather than a literal because a hardcoded 8000 broke a user who needed another port.
The host never received the same treatment.

### 4.3 Identifiers: everything is a UUID

Every primary key in the schema is a `String(36)` defaulting to `uuid.uuid4()`. All fourteen
tables: `projects`, `sites`, `deployments`, `deployment_queue`, `files`, `detections`,
`detection_embeddings`, `events`, `event_observations`, `jobs`, `label_taxonomy`, `audit_log`,
and the `event_files` association table (which uses a composite of two UUIDs). There is not
one autoincrement integer anywhere.

This matters more than it looks. Two databases filled independently on two machines can never
collide on an identifier. Merging them would be an insert in foreign key order with no ID
rewriting at all, which is unusual. With integer keys this investigation would have had a much
worse set of options.

The design that was chosen does not need merging, so this is insurance rather than a
foundation. It is recorded because it is the thing that would make a future change of mind
cheap.

### 4.4 The database layer

SQLite, and not incidentally. `backend/app/db/base.py:66` sets `PRAGMA journal_mode=WAL` on
every connection, along with `foreign_keys=ON`, `synchronous=NORMAL` and `cache_size=-64000`
(a 64MB page cache ceiling, per connection). `backend/app/db/backup.py:373` refuses any URL
that is not `sqlite:///`, with the comment that nothing else in the app would work anyway. The
backup system uses `sqlite3.Connection.backup` directly.

So SQLite is not a configuration choice that could be swapped for Postgres. It is assumed by
the backup story, the pragma setup, the migration checks and the health checks.

Write concurrency is handled by waiting rather than failing. `connect_args={"timeout": 30}` at
`backend/app/db/base.py:89` carries a comment explaining that Python's 5 second default turned
any slow write into "database is locked" errors on every concurrent request.

One structural problem, covered properly in section 10.1: `get_db` builds a brand new engine
for every single request.

### 4.5 Where media lives, and why paths are fragile across machines

`Deployment.folder_path` is stored exactly as the user picked it and is never resolved.
DEVELOPERS.md has a whole section on this ("Paths to user media are never resolved") and the
rule is absolute, because `Path.resolve()` on Windows expands a mapped network drive to its
UNC target and every `relative_to` comparison downstream then fails. There is a test guard
that fails when `.resolve()` reappears in the four files that produce or compare the stored
form.

The consequence for this investigation: a database written on a Mac holds
`/Volumes/CamTrap/...` and the same database opened on Windows holds nothing matching
`E:\CamTrap\...`. There is a bulk relink endpoint at
`backend/app/api/routers/deployments.py:332` that rewrites every `File.file_path` for a
deployment, plus a relink target suggester, so this is recoverable. But it is a chore, and it
recurs on every handoff between machines with different mount points.

This is the strongest technical argument against the database-shuttle approach and it is the
one the lab user would have discovered the hard way.

### 4.6 Jobs and inference

Inference does not run inside the API process. `backend/app/workers/detection_worker.py`
drives subprocesses and deliberately keeps them off the event loop: `run_in_executor` at lines
416 and 664, `asyncio.to_thread` at lines 502 and 738, the last with the comment "to_thread
keeps it responsive". Progress is marshalled back to the loop with
`asyncio.run_coroutine_threadsafe`.

The ML environments themselves are separate conda environments under `backend/app/ml/envs`
(addaxai-base, pytorch, pywildlife, tensorflow-v1, tensorflow-v2), managed by
`environment_manager.py`.

`Job.status` is a plain database column with an index on it (`idx_jobs_status` at
`backend/app/models/job.py:68`). So "is anything running right now" is one cheap indexed query
that any client can ask, which becomes relevant in section 8.

Progress reaches the UI over a WebSocket scoped to a single job id
(`backend/app/api/routers/websocket.py:25`), using a ready-handshake protocol in
`backend/app/core/websocket_manager.py` so work starts only once the frontend says it is
listening.

### 4.7 What already exists that this plan can reuse

**Per-resource conflict guards.** The codebase already refuses conflicting operations with
409. Deployment split returns 409 when an active job or queue entry blocks it
(`backend/app/api/routers/deployments.py:894`), model download returns 409 if that model is
already downloading (`backend/app/api/routers/ml_models.py:326`), folder runs look up an
active job before acting (`folder_runs.py:532`). This is an established pattern to extend
rather than a mechanism to invent.

**Verification progress per deployment.** `DeploymentVerification` at
`backend/app/api/schemas/deployment.py:175` carries `verified` and `total`, surfaced in the
Deployments Info sheet. The lab's "keep track of progress" requirement is already built.

**Free-form tags on deployments.** `Deployment.tags` is a JSON column
(`backend/app/models/deployment.py:91`) and the deployments CSV import reads `tag:<name>`
columns straight into it (`backend/app/services/csv_import_deployments.py:383`). So assigning
folders to people needs no schema change at all: a `tag:assigned_to` column in the import
spreadsheet does it today.

**A middleware pattern.** `backend/app/main.py:535` has an `@app.middleware("http")` for
unhandled exceptions, with a long comment about ordering: `add_middleware` inserts at the
front of `user_middleware` and the stack is built by wrapping in reverse, so the
first-registered middleware ends up innermost. Registering last therefore means outermost,
which is where authentication has to sit.

**A shared crop cache.** `backend/app/services/crop_service.py:23` is a process-global
`OrderedDict` LRU capped at 2000 entries (line 22). Under several users this is shared, so one
person browsing warms the cache for everyone.

**A scripted setup path for IT.** `backend/app/setup_cli.py` provides `backend --setup` and
`backend --list-models` for deployment scripts. The audience of technically-capable deployers
is already recognised in the codebase.

### 4.8 What does not exist

**Any concept of a user.** `audit_log.user_id` is nullable and carries the comment "Future:
for multi-user support" (`backend/app/models/audit_log.py:38`), and `AuditLog` is written in
exactly one place in the whole backend (`crud/deployment_split.py:721`). `Detection.verified`
is a bool with `verified_at_utc` beside it and no record of who.

**Any authentication.** No middleware, no dependency, no token, nothing. CORS allows only
localhost origins (`backend/app/main.py:559`), which is not authentication and does not try to
be.

**Any import of verification results.** CSV import covers sites and deployments only. Export
is rich (CSV, TSV, xlsx, GeoJSON, shapefile, gpkg, Camtrap DP) but strictly one-way.

**Any concurrency test.** 2065 tests in the suite, none exercising several simultaneous HTTP
clients.

**Any global job-running state in the frontend.** Greps for `jobRunning`, `isAnalysing`,
`analysisRunning` and `queueRunning` return nothing. The app blocks conflicting operations per
resource, it does not lock globally.

## 5. The three approaches considered

### 5.1 Option 1: one machine serves, everyone else uses a browser

One lab computer holds the drive, the models and the database, and runs the backend bound to
the local network. Everyone else opens it in a browser.

One database, one copy of truth, nothing to merge, ever. Lab members need neither the external
drive mounted nor a GPU nor any model installed. Only JPEGs cross the network. Work divides by
deployment, progress comes from the existing `verified / total`.

This is the KISS answer because it deletes the problem rather than solving it.

### 5.2 Option 2: partition so merging is never needed

Each person gets their own install, their own database and their own camera folders. Nobody
merges databases. Everyone exports at the end and the exports are combined in R or Python,
joined on site and deployment names rather than UUIDs.

Distributing one `sites.csv` and one `deployments.csv` through the existing CSV import keeps
the metadata vocabulary consistent. The export side is rich enough to support this properly.

The honest limitation is that it is one-way: there is no path to pull a colleague's
verifications back into a master database, so the combined export has to be the analysis
artifact. For most labs that is acceptable because the analysis happens in R anyway.

On the processing question this splits further. Either each machine analyses its own folders,
duplicating GPU work, or one machine with the good GPU analyses everything first and then
hands out copies of the database, which costs one relink per machine but only one analysis
pass.

### 5.3 Option 3: a real merge, or a real server

Two sub-variants, both rejected.

A merge script is feasible because of the UUID keys, and the insert itself would be
straightforward. What makes it real work is everything around it: `projects.name` is globally
unique and `sites` is unique on `(project_id, name)`, so those collide by design;
`label_taxonomy` is populated per database and needs reconciling; both databases must sit on
the identical alembic revision; and there must be a stated policy for when two people verified
the same detection differently. It also lives outside the app, so it rots the moment the
schema moves.

Moving to Postgres with real accounts and roles means rewriting the backup system (which is
`sqlite3`-specific and explicitly crashes on anything else), dropping the WAL pragmas, and
building authentication from nothing. That is a large change to a desktop app in order to
arrive where TRAPPER already is: an open source, database-driven, multi-user, role-based
camera trap web application built for exactly this.

### 5.4 Why option 1 won

It is the only one of the three that delivers simultaneous verification, which is what was
actually asked for. It is the least code. And it makes the merge question disappear rather
than answering it, which is the better move when the merge question is the user's main worry.

## 6. Why the obvious alternatives fail

Recorded because each of these is what someone reaches for first, and two of them are actively
dangerous.

**Put `addaxai.db` on a NAS or shared drive and have everyone open it.** This corrupts the
database. SQLite's locking relies on POSIX `fcntl()` advisory locking, which is unreliable over
NFS and SMB, and WAL specifically requires all processes to be on the same host and is
unsuitable for a database on a network filesystem. AddaxAI forces WAL on every connection
(`backend/app/db/base.py:66`), so this is not even a tradeoff to weigh. The same applies to
Dropbox, OneDrive and any other sync folder, where a sync client rewriting pages underneath an
open database is a corruption machine.

**Shuttle the database between computers.** This works and is what the earlier email proposed,
but it gives turn-taking, not simultaneity. Only one person holds the live database at a time.
It also copies a file that carries the DINOv2 embeddings as `LargeBinary` blobs
(`backend/app/models/detection_embedding.py:37`), which at 384 float16 dimensions is roughly
768 bytes per detection before overhead, so the file reaches hundreds of megabytes to gigabytes
on a real project. And every handoff between different operating systems or mount points needs
a full relink, per section 4.5.

**Run two AddaxAI instances against one database on one machine.** Blocked anyway:
`electron/src/main.ts:128` takes a single instance lock.

## 7. The agreed design

Ten requirements, as decided across the conversation.

**R1. Server mode.** One lab machine runs the backend bound to the LAN and holds the drive,
models and database. Everyone else opens it in a browser. Loopback remains the default, so
nothing changes for existing users. Implemented by pointing `run_server.py` at the already
declared `settings.api_host`.

**R2. One shared password, optional.** A single password via `ADDAXAI_AUTH_PASSWORD`, not
per-user accounts, no login screen. HTTP Basic, because the browser attaches it automatically
to `<img>`, `<video>` and WebSocket requests. Enforced in one ASGI middleware covering both
HTTP and WebSocket scopes. Localhost exempt, decided from the real socket peer and never from
a header. A `Host` header allowlist alongside it.

**R3. Startup interlock.** The backend refuses to boot when the bind host is non-loopback and
the password is empty. This makes the unsafe configuration unreachable rather than merely
discouraged, and it is the single highest-value piece of the whole plan.

**R4. Configured by environment variables only.** No install wizard step, no Settings toggle. A
documented deployment feature for the one person setting up the shared machine.

**R5. No user accounts.** No per-user passwords, no roles, no sessions, no password resets, no
admin screen.

**R6. Attribution, separate from access.** A nullable `verified_by` on both `Detection` and
`File`. A self-populating dropdown sourced from `SELECT DISTINCT verified_by`, sticky per
browser in `localStorage`, always visible in the verify UI. Null everywhere means today's
behaviour exactly.

**R7. No global lock during inference.** Keep the existing per-resource 409 guards. Analysis
runs for hours and locking everyone out would defeat the purpose of the shared setup.

**R8. Global job visibility.** A banner every browser can see, derived from `Job.status`, so
other users understand why things are slower instead of just experiencing it.

**R9. Gate only the heavy writes.** Bulk verify, delete, split and export, using the 409
pattern that already exists. Single-detection verification never blocks.

**R10. Work splits by deployment**, with the existing per-deployment `verified / total` as the
progress readout and `tag:assigned_to` as the assignment mechanism.

**R11. The UI always says which instance it is.** A persistent indicator naming the backend a
client is talking to: the server's hostname for a remote session, "this computer" for a local
one. Nothing in the frontend currently reveals this, and a lab member can have a local install
and a browser tab open side by side that look identical. Section 13 is why this is a
requirement rather than a nicety.

**R12. Inference runs only on the host machine.** Analysis is started on the machine that
serves AddaxAI, and lab members do not run their own local installs against the shared camera
folders. Partly enforced already (a browser has no native folder picker), partly enforceable
(the backend knows whether a client is loopback), and partly documentation only (nothing can
stop somebody's own installed copy). Section 13.6 separates the three.

## 8. The core architectural decision

Three separations carry the whole design. Each one is the difference between a small feature
and a large one.

### 8.1 Authentication is not attribution

The request "can we have users" bundles two unrelated things. Keeping strangers out needs a
password. Knowing who verified a box needs a name. A name needs no password, and a password
needs no name.

Fusing them produces accounts, hashing, sessions, logout and an admin screen. Separating them
produces one environment variable and one nullable column. The functional difference for a lab
of four is nothing.

This is why R2 and R6 are independent, and why R6 should be decided on whether attribution
matters scientifically rather than as part of the security work.

### 8.2 Per-user passwords buy nothing without an authorization model

There is no per-project permission model in AddaxAI and no plan to build one. Every account
would therefore be able to delete every project, wipe the database and browse the host
filesystem. Accounts would deliver the cost of access control and none of the containment that
people assume accounts bring.

A single shared password gives exactly the same real protection as five individual ones, at a
fraction of the cost, and is honest about what it does.

### 8.3 Exposure and password are one decision, not two

A password on a loopback-only server protects nothing, since anyone at that machine can open
`addaxai.db` with any SQLite browser. Server mode without a password is the dangerous state.
Shipping them as independent toggles guarantees that somebody eventually flips one and not the
other, and it will be the wrong one.

R3 makes them a single act at the level that cannot be bypassed: the process refuses to start.

## 9. Feasibility: what makes this cheap

Point by point against the real code.

**R1** is finishing a setting that is already declared. `settings.api_host` exists at
`config.py:76` and is used nowhere; `run_server.py:43` has the literal. The frontend needs no
change at all because of `window.location.origin` (`api-client.ts:26`), and the backend already
serves the SPA (`main.py:628`). CORS is not involved, because a browser at
`http://host:8000` is same-origin with the page it loaded.

**R3** hangs off the `@model_validator(mode="after")` that already exists at `config.py:163`.
That validator already crashes on a relative `ADDAXAI_USER_DATA_DIR`, so refusing a
non-loopback host with no password is the same shape of rule in the same place, and it matches
repo conventions 1 and 2 (crash early and loudly, explicit configuration with no defaults).

**R2**'s middleware copies the pattern at `main.py:535`, whose comment already documents the
ordering rule needed to put authentication outermost. FastAPI ships `HTTPBasic` for the
dependency form, which satisfies convention 13 (use built-in features), though the middleware
form is what keeps it DRY across 164 routes.

**R8** is one indexed query, since `idx_jobs_status` already exists
(`backend/app/models/job.py:68`).

**R9** extends the 409 pattern already present at `deployments.py:894` and
`ml_models.py:326`.

**R10** needs no code whatsoever. `DeploymentVerification` already reports `verified / total`,
and `tag:assigned_to` already works through the deployments CSV import.

**R11** has a source of truth already. `useAppVersion` reads `/health`, which works in a
browser, and the backend knows its own bind address and whether a given client is loopback. The
work is displaying it, not deriving it. Worth folding into the same pass as R8, since both put
persistent status in the chrome.

**R12** is mostly already true. A browser has no native folder picker, so the interactive
"analyse a new folder" path cannot be reached remotely at all
(`frontend/src/components/analyses/FolderSelector.tsx:130, 160`). What remains is replacing a
dead button with an explanation, and deciding the open question in section 13.7.

Two further audit findings help materially:

**Inference does not block the API.** The detection worker already pushes blocking subprocess
work off the event loop (`detection_worker.py:416, 502, 664, 738`). This removes what would
otherwise have been the dominant multi-user risk, and it is why R7 is safe.

**The crop cache is shared.** `crop_service.py:23` is process-global, so under several users
one person's browsing warms the cache for the rest. With 24 hour immutable cache headers on
images and crops, repeat views do not reach the server at all.

## 10. Feasibility: what is genuinely risky

### 10.1 The engine is rebuilt on every request

`get_db` calls `get_session_factory`, which calls `get_engine`, which calls `create_engine`
fresh each time (`backend/app/db/base.py:73-103`). `dispose()` appears nowhere in the backend.

SQLAlchemy 2.0 gives file-based SQLite a `QueuePool` by default, so every request constructs
and discards an entire connection pool, opens a connection, runs four pragmas and registers the
`seeded_hash` UDF. `cache_size=-64000` is a 64MB ceiling per connection, not per application.
Keeping one engine per database rather than one per request is the documented pattern, and
creating one per request defeats pooling by definition.

Single user this is invisible waste. With several browsers each opening multiple parallel
connections it becomes real overhead.

The fix is small, roughly an `@lru_cache` on `get_engine`. But check why it is uncached before
changing it: the restore flow swaps the database file at startup and the reset flow wipes user
data while the process is alive, and a cached engine holding a connection to a deleted file is
exactly the kind of thing that would have driven someone to write it this way. If that turns
out to be the reason, the answer is disposing the cached engine on those two paths rather than
rebuilding it on all 164 endpoints.

### 10.2 Blocking database calls inside `async def`

Twelve endpoints in the events router are `async def` and call synchronous SQLAlchemy directly,
for example `event_crud.get_event_with_files(db, event_id)` at
`backend/app/api/routers/events.py:327`. FastAPI runs `async def` bodies on the event loop, so
a blocking call there stalls the whole process: every other user's requests, the static files
and the WebSockets.

One user never notices, because they are waiting on their own request anyway. Five users means
one heavy query pauses everyone.

This is the most awkward item, because the obvious fix is wrong. These endpoints cannot simply
become `def`: DEVELOPERS.md explains that observational datetimes are serialized using a
`ContextVar` set in the endpoint body, and for a sync endpoint FastAPI runs that body in a
threadpool where the `ContextVar` is invisible to the serialization stage running in the event
loop task. Converting them to sync would silently break timezone correctness, which is pinned
by tests in `backend/tests/api/test_datetime_wire_format.py`.

The fix that satisfies both constraints is to keep `async def` and wrap only the blocking query:

```python
event = await asyncio.to_thread(event_crud.get_event_with_files, db, event_id)
```

The `ContextVar` is then still set in the async body, in the same task as serialization, so
the timezone behaviour is unchanged while the loop stays free. The repo already uses
`asyncio.to_thread` in the detection worker, so this follows existing practice rather than
introducing a pattern. It is mechanical but touches a dozen endpoints and wants tests.

One caveat to verify during implementation: a `Session` is not thread-safe for concurrent use,
though handing it to one worker thread at a time as above is the normal pattern. Confirm that
no endpoint interleaves loop-side and thread-side use of the same session.

### 10.3 The WebSocket bypasses HTTP middleware

`@app.middleware("http")` is Starlette's `BaseHTTPMiddleware` and only sees
`scope["type"] == "http"`. WebSocket connections pass straight through it. Using the decorator
form for R2 would leave `/ws/jobs/{job_id}` unauthenticated.

Impact is modest on its own, since an attacker would have to guess a job UUID to learn progress
percentages, but it is a hole in something advertised as protected. Writing R2 as a plain ASGI
middleware that handles both scopes is about the same amount of code and closes it.

### 10.4 Bandwidth

Full-size originals are served raw as `FileResponse`. Camera trap JPEGs at 5 to 10MB, times
several people flipping through images, is comfortable on gigabit ethernet and noticeably less
so on shared wifi. The 24 hour cache headers help on revisits, not on first pass. The crop and
thumbnail paths are much lighter and are what the verify UI uses most, so this mainly affects
full-image inspection.

Worth measuring during the load test rather than guessing.

### 10.5 Concurrent edits to the same file

Last write wins, silently. There is no row versioning, no conflict detection, and the
WebSocket carries job progress only rather than a change feed, so nobody's screen refreshes
when someone else edits. Two people verifying the same event will overwrite each other without
either noticing.

R10 avoids this by convention. Nothing enforces it, and enforcing it would mean a locking or
assignment model that is out of scope.

### 10.6 Nothing has ever been tested under concurrency

2065 tests, none exercising several simultaneous HTTP clients. Every claim in sections 9 and 10
about behaviour under load is reasoning from the code, not measurement.

## 11. Security assessment

### 11.1 What the password does and does not do

It controls who gets in. It does not constrain what they can do once in, because there is no
authorization model to constrain them with.

Post-authentication, every user is a full administrator. Specifically, and each verified in the
source:

- `GET /api/deployments/preview-folder?path=/` scans any absolute path handed to it
  (`deployments.py:120`), so the entire host filesystem is enumerable.
- `GET /api/deployments/preview-image` blocks `..` traversal but only within its `folder`
  argument, and `folder` is caller-supplied (`deployments.py:211`). Any image or video anywhere
  on the host is therefore readable. Non-image content is not readable, because Pillow or
  ffmpeg has to decode it, but the error path returns `str(e)` in a 500.
- `POST /api/setup/reset` wipes user data and can schedule a database wipe
  (`setup.py:549`). Its only guard is the literal string "RESET", which the client supplies and
  which is not a secret.
- `POST /api/backup/restore` takes an arbitrary source path (`backup.py:106`).
- `POST /api/backup/snapshot` can write a database copy to a chosen directory.
- `GET /api/logs/diagnostic-zip` bundles system info, paths, database info and recent jobs
  (`logs.py:452`).

None of these are bugs and none are introduced by this change. They are the normal surface of a
single-user desktop application that has always assumed the only caller is the person at the
keyboard. Binding to a network is what changes their meaning.

The accurate description for documentation is a door lock on a house with no interior walls.

### 11.2 Transport

HTTP Basic sends base64, which is encoding and not encryption. Anyone able to capture traffic
on the network reads the password. On WPA2 or WPA3 wifi and switched ethernet this is not
trivial for another client, but it is not a guarantee either.

The practical consequence is one rule for the docs: the AddaxAI password must not be a password
used anywhere else.

HTTPS would fix this and is not worth it. On a LAN it means self-signed certificates and
browser warnings on every machine, or a real certificate requiring a domain name. That rabbit
hole is deeper than the feature.

### 11.3 Cross-origin and rebinding

A lab member's browser caches Basic credentials for the origin. If that person later visits a
malicious page, that page can cause requests to the AddaxAI server with credentials attached.

CORS stops it reading responses, and JSON bodies trigger a preflight that fails because
`main.py:559` allows only localhost origins. But `multipart/form-data` is a simple request type
that does not preflight, and the CSV import endpoints take `UploadFile`. The same shape of
problem applies to DNS rebinding.

The cheap standard mitigation is a `Host` header allowlist in the same middleware, rejecting
requests whose `Host` is not the expected server name or address. This is part of R2 rather than
an optional extra.

### 11.4 The localhost exemption

Exempting loopback is sound, because anyone sitting at the host machine can already open
`addaxai.db` directly, so requiring a password from `127.0.0.1` protects nothing. It also keeps
the Electron desktop app working unchanged with authentication switched on.

The implementation trap is that the exemption must read the real socket peer from
`scope["client"]`, never a `Host` or `X-Forwarded-For` header. Headers are attacker-controlled
and would hand anyone a bypass by typing one. Nothing in the backend currently reads forwarded
headers, and a grep confirms no `TrustedHostMiddleware` or `X-Forwarded` handling exists. Keep
it that way.

### 11.5 Verdict

Safe enough for a lab LAN with a handful of known people, where the realistic threat is a
curious colleague or a wrong click.

Not safe as a general multi-user mode, and it must never be port-forwarded or placed on a
public address. Not appropriate on an open or campus-wide network where "everyone on the LAN"
means thousands of strangers.

### 11.6 The honest argument against shipping R1 at all

Today a user cannot easily expose AddaxAI to a network. Shipping R1 creates a category of user
who puts a filesystem-browsing, data-deleting API on a network, and some of them will ignore
the documentation.

R3 is what makes this acceptable, which is why the interlock should be treated as load-bearing
rather than as a nicety. Run `/security-review` on the branch before merging.

## 12. Concurrency assessment

### 12.1 The write lock is not the main problem

This is what people expect to be the issue and mostly is not. SQLite allows one writer at a
time, and the 30 second busy timeout makes competing writers wait rather than fail. WAL means
readers never block on the writer, so verification reads during an analysis are unaffected.

Verification writes are small and human-paced, so several people clicking labels interleave
fine.

### 12.2 What does hold the lock long enough to hurt

Bulk verify takes up to 500 files in one transaction (`files.py:104`), and `set_file_verified`
(`crud/file.py:621`) is not a trivial write: it rejects every sub-threshold box on the visible
frame, verifies the visible ones, re-derives `observation_type` and recomputes the event MaxN.

Deletes are the other case. `purge_deployment_data` is heavily optimised (leaves first, bulk
statements, 39 seconds for a 400,000 file project versus 124 seconds for a naive cascade), but
it is still one long transaction.

Exports and the analysis ingest are the same shape.

This is what R9 targets, and why single-detection verification is deliberately excluded from
it.

### 12.3 Why R7 rejects a global lock

Inference runs for hours on a large folder. Blocking all verification for all lab members for
that long is the opposite of what the shared server exists to provide, and it would remove a
capability that works today for one user, who can start an analysis and go verify a different
deployment while it runs.

It is also heavier than the problem needs, since WAL means reads are unaffected and small
writes queue for milliseconds.

### 12.4 What R8 fixes instead

The real multi-user gap is not blocking, it is visibility. The WebSocket is scoped per job id
and the browser that started the job is the one that subscribes. Every other lab member's
browser has no idea an analysis is running. They experience unexplained slowness with nothing
on screen to explain it.

Server-derived state from `Job.status` reaches every connected browser for free, with no new
plumbing. A banner naming what is running, and its progress, converts mysterious sluggishness
into an understood condition.

### 12.5 Expected behaviour, stated as a prediction

Three to five people verifying separate deployments should be fine. The app will feel slower
than single-user because of 10.1 and 10.2, and there will be visible pauses during bulk
operations and analysis ingest.

This is a prediction from reading the code, not a measurement. See section 17.

## 13. A second AddaxAI on a lab member's own machine

The question that produced this section: what happens if a lab member runs inference in their
own locally installed AddaxAI while also using the served one from the main machine?

The short answer is that nothing breaks, and that is exactly what makes it dangerous. This is
not a concurrency problem. It is a data fragmentation problem, and it is invisible.

### 13.1 The two instances never meet

Their laptop runs its own backend process, its own database at `~/AddaxAI/addaxai.db` and its
own models. The browser tab talks to the main machine. There is no port conflict, because the
local backend binds `127.0.0.1:8000` while the served one is on the host's address, and those
are different hosts. The Electron single instance lock (`electron/src/main.ts:128`) is per
machine and does not apply.

Local inference therefore never touches the shared database, never competes for its write lock
and never stalls its event loop. The lab server is completely unaffected. Somebody could run a
twelve hour analysis on their laptop and no colleague would notice anything.

### 13.2 The results go nowhere

Everything that analysis produces lands in the laptop's own database. The shared database gets
no deployment, no files, no detections and no verifications.

So the person has done real work that nobody else can see and that cannot be combined without
the merge tooling section 5.3 explicitly rejected. This is the problem the whole design exists
to eliminate, reintroduced by accident, by a user who believed they were helping.

### 13.3 Artifacts on a shared drive coexist rather than collide

Worth checking rather than assuming, and the answer is better than expected.
`backend/app/workers/detection_worker.py:313` scopes analysis artifacts to
`.addaxai/projects/<project_id>/`, and `project_id` is a UUID that necessarily differs between
the two databases. The laptop therefore writes an entirely separate artifact tree beside the
server's, with its own `results.json`, best frames and video frames. Nothing overwrites
anything.

There is a non-scoped fallback at `backend/app/ml/inference/megadetector.py:356` writing
`.addaxai/detection_results.json` directly, but the project analysis path always passes an
explicit `output_path`, so that branch is not reached here. Worth remembering if a future code
path stops passing one.

The cost is disk rather than corruption, and video frame extraction is the bulk of it. A second
full set of artifacts on a shared drive is not small.

This scenario also requires the media to be reachable from the laptop at all. A USB drive
attached to the host machine makes it impossible for the shared folders. A NAS makes it easy,
and unlike SQLite, JSON and JPEG artifacts over SMB or NFS are perfectly safe.

### 13.4 The real hazard is that the two look identical

This is the failure to plan around, and it is a user interface gap rather than a technical one.

Both windows are AddaxAI, with the same interface and the same branding, and nothing anywhere
on screen says which database is being looked at. A grep of the frontend finds no hostname, no
server identity, nothing. The About page is worse than neutral: it reads the version over
Electron IPC and falls back to "(dev)" in a browser
(`frontend/src/pages/AboutPage.tsx`), so the served instance displays less identifying
information than the local one. The health-backed `useAppVersion` hook does work in a browser,
so the fix has a source of truth already.

The realistic incident is not a deliberate rogue analysis. It is somebody with both open,
verifying for two hours in the wrong tab, with nobody noticing until the counts fail to add up.

R11 exists because of this paragraph.

### 13.5 The legitimate version of this, and why it is still not supported

A lab member with a much better GPU wanting to run the heavy analysis is a reasonable thing to
want, and the original email raised exactly this trade-off between per-machine GPU capability
and ease of combining results.

There is no path for those results to return to the shared database, so the only supported
answer is that analysis happens on the server machine. A lab that genuinely needs a second GPU
is choosing option 2 from section 5.2, which means accepting separate databases and combining
exports at the end.

Say this plainly in the documentation. The alternative is people discovering the capability
themselves and reasonably assuming it works.

### 13.6 What R12 can and cannot enforce

Three layers, and it matters which is which.

**Already enforced by the absence of a capability.** A browser has no native folder picker.
`frontend/src/components/analyses/FolderSelector.tsx:130` computes `isElectron()` and line 160
returns early when `window.electronAPI` is missing, because the dialog lives entirely in the
Electron main process (`electron/src/main.ts:1297`, `dialog:selectFolder`). A remote user
cannot interactively choose a new folder to analyse. Today that shows up as a dead button
rather than an explanation, which is the part worth fixing.

**Enforceable cheaply.** The backend already has to know whether a client is loopback for R2's
localhost exemption, reading `scope["client"]`. The same signal can drive the UI, so a remote
client is told plainly that analysis runs on the host machine instead of being shown controls
that cannot work.

Note that the picker's absence does not close every path. `POST /api/deployment-queue/import`
accepts a CSV carrying folder paths, and `POST /api/folder-runs/{id}/rerun` re-runs an existing
run without any folder selection. Both are reachable from a browser.

**Documentation only.** Nothing can stop a lab member running their own installed copy against
their own folders. That is their machine and their software. The only lever is telling labs
clearly, in the setup documentation, that analysis belongs on the server and that local installs
pointed at shared folders produce work nobody else will ever see.

### 13.7 One open question

R12 says inference runs on the host. That is automatic for anything triggered through the
served app, since the server executes the job regardless of which browser asked.

What is genuinely undecided is whether a remote user should be able to *trigger* a run at all,
through the queue or a re-run. Allowing it is convenient and lands the results in the right
database. Refusing it means the person who monopolises the shared machine for the next six
hours has to be sitting at it.

The recommendation is to allow triggering but not to hide it: pair it with R8's banner so
everyone can see who started what. Blocking it would be simple to implement via the loopback
signal if a lab asks for that instead.

## 14. What to deliberately not build

**Per-user accounts, password hashing, sessions, logout, password reset, admin screens.**
Section 8.2.

**HTTPS on the LAN.** Section 11.2.

**A merge tool for two databases.** Section 5.3. The UUID keys mean this stays cheap to
reconsider later if the decision changes.

**Postgres support.** Section 5.3.

**A global lock during inference.** Section 12.3.

**Row-level locking or assignment enforcement to stop two people editing the same file.**
Section 10.5. Convention plus visible progress is the proportionate answer.

**A live change feed so browsers see each other's edits.** Out of scope, and unnecessary when
people work on separate folders. Would mean broadening the WebSocket from per-job to
per-project, which is a real design change.

**An install wizard step or a Settings toggle for server mode.** Section 7, R4. The wizard is
an installer, not a preferences screen, and a bind-host change requires a restart anyway.

**A managed list of people for the attribution dropdown.** Section 7, R6. The list builds
itself from `DISTINCT verified_by`.

**Any technical attempt to stop a lab member running their own local install.** Section 13.6.
It is their machine and their software, the app cannot know the shared server exists, and
trying would be both futile and rude. Documentation is the whole of the answer.

**A way to import a colleague's locally analysed results.** That is the merge tool again, and
building it would legitimise the workflow R12 exists to discourage.

## 15. Risks and honest limits

**Typos fragment the attribution list.** "Peter", "peter" and "Pete" become three people with
no merge path. Mitigate by trimming and collapsing whitespace on write, deduping
case-insensitively for display, and putting existing names in front of people so picking is
easier than typing. Accept the residue.

**A sticky name plus a shared computer misattributes work.** Bob sits at Alice's machine and
four hundred images are recorded as Alice. This makes attribution data worse than none, because
it is confidently wrong. Mitigate by keeping the current name visible in the verify UI at all
times, so the wrong state is self-correcting.

**The name is not an identity claim.** Anyone can select anyone. It is a lab notebook
signature. The UI wording must not read like a login, or users will assume it protects
something.

**Client isolation on campus wifi.** Many university and guest networks block device-to-device
traffic even on the same SSID, and eduroam usually does. Large buildings also split floors onto
separate VLANs. No configuration on our side fixes this. The test is a ping from a lab member's
laptop to the host machine, and it should be step one in the documentation.

**The host machine must stay awake.** A laptop that suspends takes everyone down. Sleep
settings matter, and the drive must stay mounted.

**Firewall prompts.** The OS firewall will prompt or silently block inbound connections on the
port. This trips people up more often than anything else here.

**Address stability.** DHCP can hand the host a different address after a reboot and break
everyone's bookmark. Reserve the address on the router, or use the machine name. On macOS the
Bonjour `.local` name works out of the box and is usually simpler.

**The two robustness items may turn out to matter more than expected.** Sections 10.1 and 10.2
are reasoning, not measurement. If the load test shows them dominating, the effort estimate in
section 16 roughly doubles.

## 16. Effort estimate

Rough, and estimates rather than measurements.

| Item | Estimate | Notes |
|---|---|---|
| Load test (step zero) | 0.5 day | Five browsers, five deployments, a bulk verify mid-run |
| R1 + R3 together | 1 day | Includes the docs page; neither is safe alone |
| R2 | 1 to 2 days | Most of it is the ASGI middleware, WebSocket scope and Host check |
| R8 | 0.5 day | One indexed query plus a banner |
| R9 | 0.5 day | The 409 pattern already exists |
| R6 | 1 to 2 days | One migration, two nullable columns, modest UI |
| R11 | 0.5 day | Same pass as R8; the data already exists |
| R12 | 0.5 day | Mostly explaining a restriction that already exists, plus docs |
| Engine caching (10.1) | 0.5 day | Plus real thought about restore and reset |
| `to_thread` conversion (10.2) | 2 days | Mechanical, a dozen endpoints, wants tests |

Suggested order: the load test first, because it tells you whether the last two rows are
required or merely nice. Then R1 and R3 together. Then R2. Then R8 and R11 in one pass, which
is what makes the multi-user experience legible. Then R12, since it depends on the loopback
signal R2 introduces. R6 and R9 last, as both are independent of everything else.

R10 needs no work at all.

## 17. Step zero, before any of that

Three things, none of which is code.

**Run the load test.** Five browsers on five deployments, someone running a bulk verify in the
middle, someone else starting an analysis. Measure request latency and note where it stalls.
This converts every "should be fine" in this document into a number, and it decides whether
sections 10.1 and 10.2 are urgent or merely untidy.

**Confirm the lab's network allows it.** Ask them to ping the host machine from a lab member's
laptop before anything is planned around this. Client isolation is invisible until you hit it
and no amount of work on our side fixes it.

**Confirm whether their machines share an operating system.** Mixed Mac and Windows is what
turns any copy-the-database fallback into constant relinking, and it is worth knowing before
recommending a path.

## 18. Reproducibility notes

Everything in this document was established by reading the repository at `5c4401d`. No code was
written, no tests were run, and the backend was not executed (its dependencies are not
installed in the investigation environment), so nothing here is empirically confirmed.

Key files, with the line numbers current at that commit:

| Finding | Location |
|---|---|
| Frontend uses `window.location.origin` | `frontend/src/lib/api-client.ts:26` |
| Backend serves the SPA | `backend/app/main.py:628` |
| Electron is a browser on localhost | `electron/src/main.ts:139` |
| Bind host hardcoded | `backend/run_server.py:43` |
| `api_host` declared, never used | `backend/app/core/config.py:76` |
| Validator to hang the interlock on | `backend/app/core/config.py:163` |
| Middleware pattern and ordering rule | `backend/app/main.py:535` |
| CORS origins, localhost only | `backend/app/main.py:559` |
| WAL forced on every connection | `backend/app/db/base.py:66` |
| 30 second busy timeout | `backend/app/db/base.py:89` |
| Engine built per request | `backend/app/db/base.py:73-103` |
| Backup refuses non-sqlite URLs | `backend/app/db/backup.py:373` |
| UUID primary keys | every file in `backend/app/models/` |
| `audit_log.user_id`, unused | `backend/app/models/audit_log.py:38` |
| `Job.status` index | `backend/app/models/job.py:68` |
| `Deployment.tags` JSON column | `backend/app/models/deployment.py:91` |
| Embeddings as blobs | `backend/app/models/detection_embedding.py:37` |
| `File.verified` is a rollup | `backend/app/models/file.py:110` |
| `set_file_verified` cascades to boxes | `backend/app/api/crud/file.py:621` |
| Verification progress schema | `backend/app/api/schemas/deployment.py:175` |
| Image serving and cache headers | `backend/app/api/routers/files.py:256, 277` |
| Bulk verify, 500 files | `backend/app/api/routers/files.py:104` |
| Arbitrary folder scan | `backend/app/api/routers/deployments.py:120` |
| Arbitrary image read | `backend/app/api/routers/deployments.py:211` |
| Bulk relink | `backend/app/api/routers/deployments.py:332` |
| 409 conflict guard pattern | `backend/app/api/routers/deployments.py:894` |
| 409 on model download | `backend/app/api/routers/ml_models.py:326` |
| Reset endpoint | `backend/app/api/routers/setup.py:549` |
| Restore endpoint | `backend/app/api/routers/backup.py:106` |
| WebSocket scoped per job | `backend/app/api/routers/websocket.py:25` |
| Async endpoint with blocking DB call | `backend/app/api/routers/events.py:327` |
| Crop LRU, 2000 entries, shared | `backend/app/services/crop_service.py:22-23` |
| Inference kept off the event loop | `backend/app/workers/detection_worker.py:416, 502, 664, 738` |
| Artifacts scoped by project UUID | `backend/app/workers/detection_worker.py:313` |
| Non-scoped artifact fallback | `backend/app/ml/inference/megadetector.py:356` |
| Folder picker is Electron-only | `frontend/src/components/analyses/FolderSelector.tsx:130, 160` |
| Native folder dialog | `electron/src/main.ts:1297` (`dialog:selectFolder`) |
| Electron single instance lock | `electron/src/main.ts:128` |
| Version over IPC, "(dev)" in a browser | `frontend/src/pages/AboutPage.tsx` |
| Version over `/health`, works anywhere | `frontend/src/hooks/useAppVersion.ts` |
| Deployment tags via CSV import | `backend/app/services/csv_import_deployments.py:383` |
| Media paths never resolved | DEVELOPERS.md, "Paths to user media are never resolved" |
| ContextVar forces `async def` | DEVELOPERS.md, "Datetime conventions" |

Pinned dependency versions at time of audit: `fastapi==0.115.6`, `uvicorn[standard]==0.34.0`,
`sqlalchemy==2.0.36`, Python `>=3.11`.

Counts at time of audit: 164 route decorators across 18 routers, 2065 tests, 170 backend Python
files, 322 frontend TypeScript files.

Sources consulted:

- SQLite over a network (https://www.sqlite.org/useovernet.html)
- SQLAlchemy SQLite dialect, pool defaults (https://docs.sqlalchemy.org/en/20/dialects/sqlite.html)
- SQLAlchemy connection pooling, one engine per database (https://docs.sqlalchemy.org/en/20/core/pooling.html)
- FastAPI HTTP Basic auth (https://fastapi.tiangolo.com/advanced/security/http-basic-auth/)
- Basic authentication over HTTP (https://www.acunetix.com/vulnerabilities/web/basic-authentication-over-http/)
- Camera trap software review, including TRAPPER and Camelot (https://pmc.ncbi.nlm.nih.gov/articles/PMC6202726/)

## 19. Plain English summary

A lab asked how several people can verify camera trap images at the same time and combine their
results cleanly. The answer that came out of this investigation is that they should not combine
anything. One lab computer keeps the drive, the models and the database and runs AddaxAI, and
everyone else opens it in a web browser over the local network. There is then only one database
and nothing to merge, nobody else needs the drive or a GPU, and the app already reports how much
of each camera folder has been verified so progress tracking comes for free.

This turns out to be cheap because AddaxAI is already a web application without anyone having
intended it that way. The desktop app is a browser window pointing at a local web server, the
frontend already works out its own address from whatever host it was opened on, and images and
video already travel over HTTP. One line currently stops the server answering anything other
than itself, and the setting needed to change it is already written into the config file and
connected to nothing.

The old advice of copying the database between computers still works but only lets one person
work at a time, so it does not do what was asked. Putting the database on a shared network drive
would corrupt it and must never be suggested.

On passwords, the recommendation is one shared password, off by default, set through an
environment variable by whoever sets up the shared machine, with the app refusing to start if it
is exposed to a network without one. Individual user accounts are not worth building, because
the app has no way to limit what any user can do, so five separate passwords would protect
exactly as much as one. Knowing who verified what is a separate and much smaller thing: let
people pick their name from a list that fills itself in, and store that name next to the
verification.

One thing to warn labs about explicitly. If somebody also runs their own copy of AddaxAI on
their laptop and analyses camera folders there, nothing crashes and nothing is corrupted, but
the results land in their own private database and no colleague will ever see them. Because both
windows look exactly alike, with nothing on screen saying which computer is being used, it is
easy to spend an afternoon verifying in the wrong place. So the app should always show which
machine it is connected to, and analysis should happen on the computer doing the serving.

The honest limits are that a password here protects the door and not the rooms, since anyone who
gets in can browse the host computer's files and delete all the data; that the password travels
in a form anyone sniffing the network could read, so it must not be reused from elsewhere; and
that this belongs on a trusted lab network and must never be exposed to the internet. Two
technical weaknesses would also make the app feel slow with several people, one where it rebuilds
its database connection on every request and one where some requests make everyone else wait.
Both have known fixes. Nothing here has been tested with real simultaneous users yet, so the
first thing to do is not write code but run five browsers at once and measure what actually
happens.

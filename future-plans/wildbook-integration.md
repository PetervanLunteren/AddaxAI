# Wildbook integration in AddaxAI: investigation and future plan

Status: investigation only, no code written.
Date: 2026-09-17
Branch the investigation ran on: `claude/access-capabilities-question-icvtpc`
Repo state at time of audit: `abe69cd`, then rebased onto `c6f537c` on `main`, VERSION `0.0.0-dev`
Wildbook source read at: `WildMeOrg/Wildbook` `main`, shallow clone taken 2026-09-17

This document is the raw material for a proper implementation plan. It holds the original
brief, the questions to answer, what the audit of both codebases found, how Wildbook's API
and identification pipeline actually work, the recommended flow, what to deliberately leave
out, the risks, what to ask the Wild Me team, and the sources. It is written to be picked up
cold months later without rerunning the investigation.

It deliberately stops short of UI design and field-level schema detail. The mechanics are
what matter here: what talks to what, in which direction, with which credential, and which
identifier survives the round trip.

---

## 1. The original brief

Verbatim, as given:

> I would like you to do an full audit and investigation of the two repos and see how and if
> there could be an integration possible with WildBook. I know the maintainers of WildBook so
> we can ask them to change or add some small features or API calls. I would be interested in
> whether or not we can have the images and or video's defined to the individual ID using the
> WildBook system. How would the connection be (API key / login credentials) and can it be
> done automatically (full automatic for addaxAI Connect real time, or only as a Im depth
> thing for extra info with manual steps). And what we get from it. Please take your time.
> I've got plenty of tokens. Be thorough. Uh, be honest. No sugar coating. And plan according
> to the principles of kiss dry yagni. A simple model that caters for all is a good model. I
> don't want you to edit any code. This is just an investigation. first do a thorough uh, lay
> of the land audit investigation and read all the docs from Wildbook and then suggest
> something to me and always end with a plain English summary of a few sentences and then
> we'll continue discussing from there.

Scope was narrowed partway through, verbatim:

> Ok let's focus on AddaxAI desktop for now. Forget all the connect stuff.

The AddaxAI Connect findings are therefore not carried into this document beyond one note in
section 11. Connect was audited and the original recommendation was Connect-first; that
recommendation was wrong and was withdrawn. Section 11.2 records why, because the reasoning
matters if Connect is revisited.

## 2. Questions to answer

1. Can camera trap images and videos be assigned to an individual animal ID using Wildbook?
2. How does the connection authenticate: API key, bearer token, or login credentials?
3. Can it be fully automatic, or does it need manual steps?
4. What do we actually get out of it, scientifically?
5. Which species does this work for, and which does it not?
6. What is the unit we send: a file, a detection, an event, an observation?
7. Which identifier lets us match Wildbook's answer back to our own rows?
8. Does Wildbook accept video?
9. What are the scale limits?
10. What would we need the Wild Me team to change, and how small can that ask be?
11. What is best in terms of UX, KISS, DRY and YAGNI?

## 3. Goals

- Let an ecologist send a scoped set of camera trap detections to a Wildbook instance and get
  individual animal IDs back onto their own records.
- Fill the `individualID` column that already exists in the Camtrap DP export and is written
  empty on every row.
- Turn detection rates into capture-recapture input, which is a different class of result.
- Keep AddaxAI's offline-first posture: no mandatory account, no mandatory network.
- Store no long-lived secret.
- Depend on as little unreleased Wildbook behaviour as possible, so a Wildbook upgrade does
  not break shipped desktop builds.

Non-goals for a first version: running re-identification locally, building an individual ID
review UI inside AddaxAI, automating the upload, syncing continuously, supporting Connect.

---

## 4. Repo audit: AddaxAI desktop

### 4.1 Shape

Electron shell, FastAPI backend, React frontend, SQLite. Local-first and folder-based: the
user points the app at a folder and nothing leaves the machine unless they ask.

Data model: `Project → Site → Deployment → File → Detection`, with `Event` and
`EventObservation` layered on top as the independence-filtered observation view.

### 4.2 The camtrap export is the template to copy

A Wildbook export is the Camtrap DP export with different column names and a different media
step. Everything it needs already exists. Files to read before writing anything:

| Concern | File |
|---|---|
| Async export job, progress, zip, temp file | `backend/app/workers/camtrap_export_worker.py` (172 lines) |
| Prepare and download endpoints | `backend/app/api/routers/export.py:299` and `:354` |
| Table builders, headers, scoping | `backend/app/api/crud/export.py` |
| Serialisers and zip builder | `backend/app/api/crud/export_formats.py` |
| Job type literal, declared twice | `backend/app/models/job.py:18` and `backend/app/api/schemas/job.py:16` |
| Progress protocol | `backend/app/core/websocket_manager.py` (`ws_manager.send_progress`) |

`camtrap_export_worker.py` is the exact shape needed: an async job, throttled WebSocket
progress at a 100 ms floor, Pillow run on a thread pool so the event loop keeps flushing,
a zip written to a temp file, and the finished path stashed back on `Job.payload` for the
download endpoint to find. A `wildbook_export_worker.py` is that file with the table builder
and the media step swapped.

### 4.3 The observation model maps onto Wildbook's almost one to one

`EventObservation` is documented in its own module docstring as "one cohort of one species
within an event, with a count and optional sex / life stage / behaviour". A Wildbook
Encounter is one animal at one place and time carrying sex, lifeStage, behavior, genus and
specificEpithet. These are the same idea.

| AddaxAI | Wildbook | Note |
|---|---|---|
| `EventObservation` | one Encounter row | the unit to send |
| `Event` | `Encounter.sightingID` | also the durable join key, see section 7 |
| `File`s in the event | `Encounter.mediaAsset0..N` | `File.best_frame_path` for video |
| `LabelTaxonomy.taxon_genus` | `Encounter.genus` | separate columns already, no string splitting |
| `LabelTaxonomy.taxon_species` | `Encounter.specificEpithet` | same |
| `LabelTaxonomy.level` | filter, not a field | drops genus and family rollups Wildbook cannot take |
| `Site.latitude` / `.longitude` | `Encounter.decimalLatitude` / `.decimalLongitude` | |
| `Site.name` | `Encounter.verbatimLocality` | free text, no vocabulary |
| `File.captured_at_local` | `Encounter.year/month/day/hour/minutes` | or `dateInMilliseconds` |
| `sex`, `life_stage`, `behavior` | same names | conditionally, see section 7.3 |
| `Event.notes` | `Encounter.sightingRemarks` | |
| `Deployment` | nothing | lost, Wildbook has no equivalent |

`LabelTaxonomy` holding `taxon_genus` and `taxon_species` as separate columns is a piece of
luck worth noting: Wildbook wants them separate and case-sensitive.

### 4.4 Video

`File.file_type` distinguishes image from video. `File.best_frame_path` is a real JPEG on
disk, chosen by `backend/app/ml/best_frame.py` and documented in DEVELOPERS.md under "Best
frame selection (videos)". Wildbook does not accept video on bulk import, so the best frame
is the only route video has into Wildbook, and AddaxAI is the only place it exists.

### 4.5 What already exists that is adjacent but should not be used yet

- `DetectionEmbedding` stores float16 DINOv2 vectors with a precomputed `l2_norm`.
- `backend/app/services/label_service.py` runs FAISS k-NN and a greedy nearest-neighbour
  chain sort in a subprocess, backing "find similar" and the suggestion cohorts.
- `models.json` has an `emb` category with three DINOv2 entries; an entry is a small JSON
  blob naming an env, a checkpoint, an input size and a dimension.
- `backend/app/services/crop_service.py` produces square crops with context padding and a
  blurred edge extension when the box runs off the image.

Together these are most of the machinery a local re-identification feature would need. That
is a different project. See section 10.

### 4.6 Where configuration lives

There is no settings table and no per-user preference store in the database. `Settings` in
`backend/app/core/config.py` is pydantic-settings over environment variables, and UI
preferences live in `localStorage` (`frontend/src/lib/species-name-mode.ts`,
`frontend/src/lib/folderRunSettings.ts`). This is a design input, not a gap: it argues for
holding the Wildbook token in memory for the duration of one pull and never persisting it.

### 4.7 The empty column

`backend/app/api/crud/export.py:223` declares `individualID` in `_CAMTRAP_OBS_HEADERS`.
Lines 1698, 1800 and 1876 write an empty string into it on every row. The slot has been cut
since the export was written and has never been filled. Its neighbours at `:224` and `:225`,
`individualPositionRadius` and `individualPositionAngle`, are the ones the depth estimation
plan targets. Same row, three empty columns, two unrelated future features.

---

## 5. Wildbook audit

### 5.1 What Wildbook is

Java on Tomcat, DataNucleus JDO over PostgreSQL, OpenSearch for search, with model serving
split out into a separate FastAPI service. Maintained by Wild Me, which merged into
Conservation X Labs.

"Wildbook" is not one system. It is a family of separate instances, each with its own
database, its own user accounts, its own catalogue, and its own taxonomy-to-model
configuration: Whiskerbook, African Carnivore Wildbook, Zebra Codex, Wildbook for Lynx,
Wild North, DeerSpotter, Flukebook, Seal Codex, Amphibian and Reptile Wildbook and others.
Any integration is configured per instance, and a user must already have an account on the
instance they are sending to.

### 5.2 Data model

`MediaAsset → Annotation → Encounter → MarkedIndividual`, plus `Occurrence` (called Sighting
in the UI), `Project` and `Survey`.

- MediaAsset is a photo or video file.
- Annotation is a bounding box on a MediaAsset carrying an `iaClass` (for example
  `giraffe_whole`, `cheetah_body`), a viewpoint, and MiewID embeddings.
- Encounter is one animal at one place and time. It carries the metadata.
- MarkedIndividual is the named animal that Encounters are assigned to.
- Occurrence groups Encounters that happened together.

There is no camera trap concept anywhere. `grep -ril 'camera trap'` over
`src/main/java` returns nothing, and the documentation site has no camera trap page. No
deployments, no trap nights, no effort, no independence interval. Camera trap data is
something users push into Wildbook sideways, not something it models.

### 5.3 The identification pipeline

Detection finds animals and labels each Annotation with an `iaClass` and a viewpoint. The
`iaClass` is what routes an Annotation to an identification algorithm. Configuration is per
instance, in `IA.json`, keyed `genus → species → ia_class → _id_conf`.

Identification algorithms, in current order of importance:

- **MiewID (MIEW-ID, µID)**, the current default. Contrastive embeddings, matched by cosine
  similarity. `MatchResult.DEFAULT_PAIRX_MODEL_ID` is `miewid-msv4.1`.
- **PIE v2**, pose invariant embeddings, per species. Does not handle 8-bit greyscale.
- **HotSpotter**, texture hot spots, the original algorithm, still used for zebra-like coats.

The pipeline produces a `MatchResult` holding ranked "prospects" with scores. It does not
assign anything. From `docs/data/matching-process.md`:

> Wildbook assists you in photo ID but never makes a decision for you.

This is deliberate, not an oversight. `grep -rn 'autoMatch\|autoAssign\|threshold'` over
`src/main/java/org/ecocean/ia/*.java` returns zero hits. There is no confidence threshold
that sets an individual anywhere in the codebase. A human opens the match results, inspects
the candidates, and clicks "Set to Individual" or creates a new ID.

### 5.4 The API

Modern surface at `/api/v3`, with an OpenAPI spec shipped in the repo at
`src/main/resources/openapi.yaml` (1389 lines) and served at `/api/v3/docs/openapi.yaml`,
with Swagger UI at `/api/v3/docs`.

Authentication splits in two, and the split drives the whole design:

| | Reads | Writes |
|---|---|---|
| Endpoints | `/api/v3/search/{encounter,individual,annotation}`, `/api/v3/media/resolve` | `/api/v3/bulk-import`, `/api/v3/encounters`, `/api/v3/annotations` |
| Filter | `tokenAuthSearch` | `authc` |
| Credential | `Authorization: Bearer <jwt>` | session cookie from `POST /api/v3/login` |
| How the user gets it | Wildbook UI, account menu, "API Access" | username and password |
| Lifetime | short, `AuthToken.java:22` defaults to 30 minutes, configurable per instance | session |

The authoritative statement of this is `src/main/webapp/WEB-INF/web.xml` lines 99 to 116, the
Shiro filter chain. `/api/v3/bulk-import = authc` and `/api/v3/bulk-import/** = authc`. The
bearer filter is wired only to search and media resolve.

So: **reading from Wildbook needs only a pasted short-lived token. Writing to Wildbook needs
a stored username and password.** That asymmetry is the single most important mechanical fact
in this document.

Tokens are RS256 JWTs carrying identity only, signed with a private key in
`apiAccessKeys.properties` (`JwtService.java`). `POST /api/v3/auth/token` mints one from HTTP
Basic credentials, pinned to `context0`.

There is also an anonymous `/api/v3/agent-skill` endpoint serving markdown instructions for
AI agents operating the read-only API. `api-reference.md` there is the best written
description of the token API that exists and should be read before implementing the pull.

### 5.5 Bulk import mechanics

Fully headless-capable, three steps:

1. Generate a UUID client-side. This is the `bulkImportId`.
2. Upload each file to `/upload?subdir=<bulkImportId>` (the resumable upload servlet,
   `org.ecocean.resumableupload.UploadServlet`, mapped in `web.xml` at `/upload`).
3. `POST /api/v3/bulk-import` with a JSON body.

The payload keys, read from `BulkImport.java:133` onwards:

```
{
  "bulkImportId":       "<the uuid you generated>",   // required
  "rows":               [ { "Encounter.genus": "Panthera", ... }, ... ],
  "fieldNames":         [ "Encounter.genus", ... ],
  "validateOnly":       false,   // always 200, "success" tells you if the data is valid
  "skipDetection":      false,   // skipping detection forces skipIdentification
  "skipIdentification": false,
  "processInBackground": false,
  "matchingLocations":  [ "<locationID>", ... ],      // scopes the match candidate set
  "tolerance": { "failImportOnError": true, "skipRowOnError": true,
                 "badFieldnamesAreWarnings": true }
}
```

`validateOnly` is worth knowing about: it is a dry run that always returns 200 and reports
validity in the body, so a client can check a batch before committing it.

Accepted field names are enumerated in `BulkValidator.java:24` onwards. Only
`Encounter.genus` and `Encounter.specificEpithet` are hard-required
(`BulkValidator.java:66`), which is a lower bar than the documentation implies. The docs ask
for location, date and `Encounter.mediaAsset0` too, and those are all genuinely needed for
the result to be useful.

Progress is read back with `GET /api/v3/bulk-import/{taskId}`.

### 5.6 Scale and format limits

From `docs/data/bulk-import-beta.md`, the two hard practical constraints:

> Your spreadsheet should have 200 Encounters or fewer. Spreadsheets larger than this will
> slow down the site for all users and will result in longer wait times for detection and
> identification.

> A folder that contains only your photos (no videos)

The 200 figure is not enforced anywhere in code. It is a shared-instance courtesy limit: the
detection and identification job queue is shared by every user of that Wildbook. Treat it as
real.

### 5.7 Other operational frictions

- **No self-registration.** `docs/getting-started-with-wildbook.md`: accounts come from site
  managers. A user cannot sign themselves up.
- **Location IDs are a configuration file.** `Encounter.locationID` must match a value in the
  instance's `locationID.json`, and adding one needs a Wildbook admin or a pull request
  (`docs/data/location-ids.md`). It is not strictly required on import, but it is what scopes
  the match candidate set, so a catalogue of any size wants it set.
- **Taxonomy is per instance and case-sensitive.** `Encounter.genus` must start uppercase and
  `Encounter.specificEpithet` lowercase, and both must match the instance's taxonomy list or
  the record shows "Not Available".

---

## 6. Which species this actually works for

This is the part that decides whether the feature is worth building at all, and it is the
part most likely to be glossed over.

Re-identification works where an animal carries a visually distinctive, stable, repeatable
pattern. MiewID msv3 was trained on a 64-species dataset and generalises to unseen taxa
better than per-species models, but "better" is not "well", and generalisation does not
create signal that is not in the pixels.

| Works | Marginal | Does not work |
|---|---|---|
| leopard, jaguar, ocelot, cheetah, tiger, snow leopard, clouded leopard, hyena, African wild dog, lynx, zebra, giraffe | some deer (DeerSpotter exists), tapir, some bears, individuals with distinctive scars or ear notches | red fox, badger, roe deer, red deer, wild boar, pine marten, hare, wolf, most rodents |

The right-hand column is not a model gap that more training data fixes. A red fox in a
night-time infrared camera trap frame does not carry marks that distinguish it from other red
foxes. For a typical Dutch or central European deployment this feature would be dead weight.
It pays for itself at sites with spotted or striped carnivores.

Practical consequence for the design: the export must be scoped to one species per batch, and
the UI should be honest that most species will never be in the supported list of any Wildbook
instance. Do not present this as a general feature.

---

## 7. The core mechanical problem: which identifier survives

This is the finding that most affects implementation, and it was only found by reading the
Java source. It invalidated the first version of this plan.

### 7.1 Wildbook clones encounters

An Encounter is defined as one animal. When detection finds more than one annotation on a
photo, Wildbook clones the Encounter so each animal gets its own. From
`docs/data/report-encounter.md`:

> If machine learning Detection has been configured for the submitted species, it will process
> the submitted photo and clone new encounters for each animal found.

The clone is `Encounter.cloneWithoutAnnotations` at `Encounter.java:3868`. It copies: day,
month, year, hour, minutes, size guess, verbatimLocality, genus, specificEpithet, decimal
lat/lon, submitterID, submitters, photographers, **sex**, locationID, country, recordedBy,
state, **alternateID**, occurrenceRemarks. It joins the clone to the same Occurrence, the same
Projects, and the same ImportTask.

### 7.2 Neither obvious key is both cloned and searchable

- `Encounter.otherCatalogNumbers` is indexed in OpenSearch as a normalised keyword
  (`Encounter.java:5010`) and serialised into the document (`Encounter.java:4506` onwards).
  It is **not** copied by the clone.
- `Encounter.alternateID` **is** copied by the clone. It does not appear in
  `opensearchMapping()` or in the document serialiser, so it cannot be queried.

Stamping only `otherCatalogNumbers`, which was the first plan, silently loses every cloned
encounter. A frame with two leopards gives one joinable record and one orphan, with no error.

### 7.3 Two keys that do work, both chosen client-side

- **`importTaskId`** is indexed as a plain keyword. `BulkImport.java:778` constructs the task
  as `new ImportTask(user, id)` where `id` is the `bulkImportId` the client generated. So the
  batch identifier is known before the request is even sent, and clones inherit it because
  `cloneWithoutAnnotations` calls `itask.addEncounter(enc)`.
- **`occurrenceId`** is indexed as a plain keyword. `Occurrence.getId()` (`Occurrence.java:444`)
  returns `occurrenceID`, and bulk import sets that from the `Encounter.sightingID` supplied in
  the row. Clones join the same Occurrence. So a per-event key chosen by the client survives
  cloning.

The working strategy is therefore: scope the pull by `importTaskId`, then resolve each hit
back by `otherCatalogNumbers` where it is present and fall back to `occurrenceId` where it is
not. Everything resolves. What is lost on a cloned encounter is per-box precision: the result
says this event contained individuals A and B, not which of our two bounding boxes is which.
That is honest, because we genuinely do not know.

### 7.4 Two rules that fall out of the cloning behaviour

**Bundle when the count is one, split when it is not.** If the observation's effective count
(`human_count` if set, else `max_n`) is 1, put every file in the event on one Encounter as
`mediaAsset0..N`; more views of the same animal is strictly better for matching. If the count
is greater than 1, emit one row per file and let Wildbook clone per detected animal. This is
Wildbook's own recommendation for social species, from `docs/introduction/data-entry.md`:

> Data for social species, with one photo per Encounter (one photo per Excel row), allowing ML
> to split each Annotation into new Encounters without assuming relationships among photos

**Only send sex and life stage when the count is one.** The clone copies `sex` to every animal
in the frame. From the same page:

> if you indicate the sex, life stage, or identity of an animal, there is no guarantee that,
> when the Encounter is replicated and Annotations are each associated with the newly created
> Encounters, the sex, life stage, and/or ID will apply to the animal you intended

Two males and a female in one frame would arrive as three males. Send nothing rather than
something wrong.

---

## 8. The core architectural decision

Two things are separable and mixing them is where this gets expensive.

**Can we get individual IDs onto our images?** Yes, via Wildbook, for the species Wildbook
covers on the instance being used.

**Can that be automatic?** The push can be. The identification cannot, ever, by design
(section 5.3). So "real time" buys nothing: nothing happens faster at the far end, because
the far end is a person.

Combine that with the auth asymmetry from section 5.4 and the answer falls out:

> Push manually. Pull automatically.

Uploading is the credential-heavy, rate-limited, version-coupled half, and Wildbook already
has a good UI for it that handles validation, review, location assignment and error reporting.
Reading is the cheap, safe, stable half, and doing it by hand means transcribing individual
names into a spreadsheet.

Automating the upload would cost a stored password and coupling to a platform we do not
control, to save a few minutes of drag and drop. Automating the download saves the user from
manual transcription of every result. Those are not the same trade.

---

## 9. Recommended flow

Three phases. Each is useful on its own and each can be stopped after. Phases 1 and 2 need
nothing from the Wild Me team.

### 9.1 Phase 1: build a Wildbook import package

Mechanics, end to end:

1. The user scopes a batch: one species, a site or deployment, a date range, optionally
   "verified only". One species per batch, because Wildbook's match candidate set and its
   configured ID model are both per taxon.
2. AddaxAI filters to species-rank labels with a binomial (`LabelTaxonomy.level`,
   `taxon_genus`, `taxon_species`), and reports what it dropped and why: not species rank, no
   binomial, no timestamp, no site coordinates.
3. It caps the batch at 200 Encounters and says why, quoting the shared-instance reason.
4. It generates a `bulkImportId` UUID client-side. This is the batch key and it is recorded
   locally before anything is sent.
5. It assigns each AddaxAI `Event` a `sightingID` and each Encounter row an
   `otherCatalogNumbers` value. Both are keys we choose. Section 7.3.
6. A background job builds a zip containing the media files plus `wildbook-import.xlsx` plus
   a small `addaxai-batch.json` recording the `bulkImportId`, the sightingIDs, and the
   per-row keys.
7. The user drags the zip into Wildbook's own bulk import page and runs the import there.

Design decisions inside that:

- **Send full frames, not crops.** Wildbook runs its own detector, and the `iaClass` it
  assigns is what routes the annotation to the right ID model. A pre-cropped animal gives the
  detector nothing to work with, and `crop_service.py` output is a square with blurred edge
  fill, which is a UI artefact rather than a photograph. For video, send
  `File.best_frame_path`.
- **Do not map location IDs, and do not send submitter.** Wildbook's bulk import review screen
  has a "+" next to columns including location ID and submitter ID that sets the value for
  every row in two clicks (`docs/data/bulk-import-beta.md`, step 3). Leaving both out removes
  an entire configuration surface from AddaxAI, and the `sightingID` join key is unaffected.
- **Do not send sex or life stage on multi-animal rows.** Section 7.4.

Cost of this phase: no credentials, no network calls, no coupling to a Wildbook version. If
Wildbook changes its import format, the user sees Wildbook's own validation errors in
Wildbook's own review screen, which is a soft failure.

### 9.2 Phase 2: pull the individual IDs back

1. The user does the identification work in Wildbook: reviews match results, sets individuals.
   This is theirs and it cannot be automated.
2. In Wildbook, account menu, **API Access**, they mint a short-lived bearer token.
3. In AddaxAI they paste the instance base URL and the token into a dialog and press fetch.
4. AddaxAI pages through the batch:

```
POST {base}/api/v3/search/encounter?from=0&size=200
Authorization: Bearer <token>
Content-Type: application/json

{"query": {"term": {"importTaskId": "<the bulkImportId generated in phase 1>"}}}
```

5. It reads `X-Wildbook-Total-Hits` first and pages with `from`/`size`, respecting the
   `from + size <= 10000` ceiling.
6. Each hit carries `individualId`, `otherCatalogNumbers`, `occurrenceId` and `mediaAssets[]`.
   Resolve by `otherCatalogNumbers` when present, else by `occurrenceId`. Section 7.3.
7. Write the individual onto our rows, and discard the token.

Mechanical gotchas for whoever implements this:

- The response envelope is **flat**: `{"hits": [ {fields...}, ... ]}`, not the standard
  OpenSearch `{"hits":{"hits":[{"_source":...}]}}` nesting. This is stated explicitly in the
  `api-reference` agent skill and is easy to get wrong.
- The hit total lives only in the `X-Wildbook-Total-Hits` header, never in the body.
- Under sustained paging the API can intermittently return HTTP 500 with `"query failed"`.
  The agent skill says these are transient and prescribes retry with backoff, roughly 1s, 2s,
  4s, up to four attempts.
- The token lifetime is per instance. The mint response's `expiresInSeconds` is authoritative
  and must not be assumed to be 30 minutes.
- Non-admin tokens see only what that user can see. This is correct behaviour, but it means a
  user who did not submit the batch may get fewer hits than expected.

**Do not persist the token.** Wild Me's own agent skill instructs agents never to log or
persist it, tokens are short-lived anyway, and AddaxAI has no settings table to put it in
(section 4.6). Hold it in memory for the duration of the pull.

### 9.3 Phase 3: automate the upload, only if asked for

Store a Wildbook username and password, `POST /api/v3/login` for a session cookie, upload each
file to `/upload?subdir=<bulkImportId>`, `POST /api/v3/bulk-import`, poll
`GET /api/v3/bulk-import/{taskId}`.

Do not build this until phase 1 has real users asking for it. It saves a few minutes of drag
and drop and costs a stored password plus version coupling in software that ships to laptops
and stays there for months.

### 9.4 Where the result lands

Two nullable text columns, plus one Alembic migration. The migration adds columns only and
touches no rows, so by the rules in DEVELOPERS.md it needs `test_upgrade_from_base_matches_models`
to stay green but does not need a data test in `test_migration_data.py`. Use text types, not
floats, to stay clear of the SQLite 3.45 `ADD COLUMN REAL NOT NULL DEFAULT` integrity-check
trap documented in DEVELOPERS.md.

**Open decision: `event_observations` or `detections`.** `event_observations` matches the unit
that is sent and matches what comes back for a cloned encounter. `detections` is where a person
would expect to see it in the Labels grid. The argument for `event_observations` is that putting
it on `detections` implies a per-box precision Wildbook does not return (section 7.3). This
needs settling before implementation.

**Open decision: where the batch record lives.** There is no natural table. `Job.payload` is
already used as a scratch store by the camtrap worker, which writes `zip_path` and
`total_files` back into it after completion. Reusing that is fine for one or two batches and a
stretch for a list accumulated over months.

Then fill `individualID` in the Camtrap DP export (section 4.7), which is the payoff that
makes the data usable downstream.

---

## 10. What to deliberately not build

- **Local re-identification.** `WildMeOrg/ml-service` is a standalone FastAPI service with a
  `POST /extract/` endpoint returning MiewID embeddings for a bbox, and AddaxAI already has
  crop embeddings, FAISS k-NN, similarity sort and a crop review grid. Adding `miewid-msv4.1`
  as a fourth `models.json` `emb` entry is genuinely close. It is still a different project,
  and it does not give the shared catalogue across organisations, which is the entire reason
  to touch Wildbook. Park it; revisit if Wildbook adoption proves the demand.
- **An individual ID review UI in AddaxAI.** The review happens in Wildbook. Display the
  result.
- **Automated or scheduled sync.** There is nothing to sync to. The far end is a person.
- **Sending our own bounding boxes.** Wildbook's detector assigns the `iaClass` that routes to
  the ID model. MegaDetector's "animal" class carries no such routing information.
- **Supporting every species.** Scope to one species per batch and be honest about coverage.

---

## 11. Risks and honest limits

### 11.1 Limits of the feature itself

- **Most camera trap species are not identifiable as individuals.** Section 6. This is the
  dominant risk: the feature could ship and be useful to a small minority of users.
- **The human step is unavoidable and is the expensive one.** Reviewing match candidates for a
  few hundred encounters is hours of work. AddaxAI can make the data arrive well-formed; it
  cannot make the decision.
- **Onboarding friction is high and outside our control.** A Wildbook account cannot be
  self-registered, a location ID needs an admin or a pull request, and the target species must
  be configured for identification on that instance. Budget more for documentation and support
  than for implementation.
- **Camera trap effort data is lost on the round trip.** Deployment, camera and trap-night
  effort have no Wildbook equivalent. Capture-recapture with effort therefore requires holding
  both halves locally and joining them ourselves, which is fine, but it means Wildbook is not
  the system of record.
- **Desktop version drift.** A shipped build lives in the wild for months. If Wildbook changes
  its import format, old builds keep producing broken packages. Phase 1 fails softly because
  the user sees Wildbook's own validation errors. Phase 2 is a single term query on one indexed
  field, which is about as stable a surface as Wildbook exposes. `catalog_updater.py` is the
  precedent if remotely-updatable config is ever needed.
- **Quantities of evidence.** The species coverage claims in section 6 and the MiewID training
  figures are second-hand, from search summaries, not primary text. See section 14.2.

### 11.2 Why the first recommendation of Connect-first was withdrawn

Recorded because the reasoning matters if Connect is revisited.

The original recommendation was to build this in AddaxAI Connect first, on the grounds that
Connect has a credential store (`ProjectIntegration`), a worker and queue pattern, persistent
crops in MinIO, and a server that stays awake. That was wrong. Phases 1 and 2 need none of
those things, so the argument only ever applied to phase 3, which is not recommended.

Meanwhile AddaxAI desktop already has: the async job-backed zip export
(`camtrap_export_worker.py`), a better crop generator than Connect's flat 10% pad, and mature
outbound HTTPS handling including a CA trust module and a relay for networks that block
HuggingFace. It also has far more users, and two decisive advantages: the archives people
actually want to catalogue in Wildbook live on hard drives, not on FTPS servers, and video
only reaches Wildbook via best-frame extraction, which Connect cannot do because it deletes
`.mp4` at ingestion.

If Connect is revisited later, the payload builder and the join-key strategy from this
document carry over unchanged. Note that the two repos share no code, so that would be a
deliberate duplication decision, not an accident.

---

## 12. What to ask the Wild Me team

Short, because phases 1 and 2 need nothing. In priority order.

1. **Make the external key survive the clone.** Either copy `otherCatalogNumbers` in
   `Encounter.cloneWithoutAnnotations` (`Encounter.java:3868`), or add `alternateID` to
   `opensearchMapping()` and the document serialiser. One line either way. It upgrades the
   join from event-level to per-animal on multi-animal frames, and anyone integrating with
   Wildbook hits this. This is the single highest-value ask.
2. **Is the 200-Encounter guidance a real number?** And is `processInBackground` the supported
   way to go larger? Right now it is a note in a documentation page with no enforcement and no
   description of what happens at 201.
3. **Expose match-result status on the read-only token API.** So a client can show "42
   encounters still awaiting review" and deep-link into Wildbook, rather than the user having
   to remember to go and look. Today `/api/v3/match-inspection` is session-only and the token
   API has no match-results endpoint. Nice to have, not required.
4. **A write-scoped bearer token for bulk import.** Would remove the stored-password
   requirement from phase 3 entirely. Worth asking for even if phase 3 is never built.
5. **Location IDs creatable per organisation or via API.** Needing a pull request against
   `locationID.json` to register a camera trap site is the largest onboarding friction in the
   whole flow.
6. **A question rather than a request: does camera trap data have a future shape in Wildbook?**
   Deployment, camera and effort are dropped on the way in today. If there is a plan, it
   changes whether we design around their absence.

---

## 13. Effort estimate

Deliberately coarse; this document is a starting block, not a plan.

Phase 1, roughly a week: export scope and filters, the Wildbook row builder, the worker
(largely a copy of `camtrap_export_worker.py`), the zip, the batch record, plus tests.

Phase 2, roughly half a week: a token dialog, a paged search client with retry and backoff,
the key resolution logic, the Alembic migration, and the Camtrap DP export change.

Phase 3, two weeks or more, and not recommended. Almost all of it is upload orchestration,
status polling and error handling.

Documentation is a significant share of the real cost, because most of what a user has to do
happens in a system we do not control.

## 14. Step zero, before any of that

Get access to a real Wildbook instance with a real camera trap catalogue, ideally Whiskerbook
or African Carnivore Wildbook, and do one batch by hand. Export 20 leopard or ocelot
encounters from a real AddaxAI project, hand-build the spreadsheet, drag it in, run detection
and identification, do the review, then mint a token and run the `search/encounter` query with
curl to confirm that `importTaskId`, `occurrenceId` and `otherCatalogNumbers` come back the
way section 7 says they do, including on a frame with two animals.

Everything above is conditional on that round trip working. In particular, verify by
experiment that a cloned encounter really does keep `importTaskId` and `occurrenceId` and
really does lose `otherCatalogNumbers`, because that is read from source, not observed.

---

## 15. Reproducibility notes

### 15.1 What was audited, AddaxAI

Read in full or in part: `CONVENTIONS.md`, `README.md`, `DEVELOPERS.md` (table of contents,
plus the migrations, deleting-analysis-data, detection-threshold, best-frame, HuggingFace and
catalog sections), `models.json`,
`backend/app/models/{detection,detection_embedding,file,event,event_observation,project,deployment,site,job,label_taxonomy}.py`,
`backend/app/workers/camtrap_export_worker.py`, `backend/app/api/routers/export.py`,
`backend/app/api/crud/export.py` (headers and function list), `backend/app/api/schemas/job.py`,
`backend/app/services/crop_service.py`, `backend/app/services/label_service.py` (header),
`backend/app/ml/embedding_utils.py`, `backend/app/ml/catalog_updater.py` (header),
`backend/app/core/config.py` (header), `frontend/src/lib/species-name-mode.ts`,
`frontend/src/lib/folderRunSettings.ts`, `metrics/download_counts.csv`, and the directory
listings for `backend/app/api/routers/`, `backend/app/workers/`, `frontend/src/pages/`,
`frontend/src/components/`.

Grepped: `wildbook|wildme|individual_id|individualID|re-id|miew|hotspotter` across the repo
(only hit: the empty `individualID` export column), `localStorage` across `frontend/src/lib`
and `frontend/src/hooks`, `camtrap_export` across the backend.

### 15.2 What was audited, Wildbook

Shallow clones taken 2026-09-17:

- `https://github.com/WildMeOrg/Wildbook` (2417 files)
- `https://github.com/WildMeOrg/ml-service`
- `https://github.com/WildMeOrg/wildbook-docs`
- `https://github.com/WildMeOrg/scout`

Read in full or in part: `src/main/resources/openapi.yaml`,
`src/main/webapp/WEB-INF/web.xml` (Shiro filter chain and servlet mappings),
`src/main/java/org/ecocean/api/{AuthToken,BulkImport,SearchApi,MatchInspection,UploadedFiles,AgentSkill,EncounterExport}.java`,
`src/main/java/org/ecocean/api/auth/JwtService.java`,
`src/main/java/org/ecocean/api/bulk/{BulkValidator,BulkImporter}.java`,
`src/main/java/org/ecocean/ia/{IA,MLService,Task,MatchResult}.java`,
`src/main/java/org/ecocean/{Encounter,Occurrence}.java` (targeted sections),
`src/main/java/org/ecocean/servlet/RestServlet.java` (header),
`src/main/resources/agent-skills/{index,api-reference}.md`,
`devops/deploy/.dockerfiles/tomcat/IA-wbia.json`,
and from the docs repo:
`docs/data/{bulk-import-beta,matching-process,location-ids,data-exports}.md`,
`docs/introduction/{image-analysis-pipeline,data-entry}.md`,
`docs/getting-started-with-wildbook.md`,
plus `ml-service/README.md` and `ml-service/app/model_config.json`.

Grepped: `camera trap|cameratrap` across `src/main/java` and the docs repo (nothing),
`autoMatch|autoAssign|threshold` across `src/main/java/org/ecocean/ia/*.java` (nothing),
`otherCatalogNumbers|alternateID` across `Encounter.java`.

### 15.3 Egress limits during the investigation

The session's proxy blocked `wildbook.docs.wildme.org`, `docs.wildme.org`, `www.wildme.org`,
`wildbook.org`, `community.wildme.org`, `huggingface.co` and `arxiv.org`.
`raw.githubusercontent.com` was reachable and `git clone` over HTTPS to `github.com` worked,
although `github.com` HTML pages returned 403.

Therefore: **everything about Wildbook's data model, API, authentication, bulk import,
encounter cloning and OpenSearch mapping was read directly from the cloned source and the
cloned docs repository, and is first-hand and checkable against the file:line references
above.** Everything about species coverage, MiewID training scale and the Wildbook platform
list came from web-search result summaries, not from primary text.

Specifically second-hand and needing verification before being quoted anywhere:
the "64 species, 225k photos, 37k individuals" MiewID figure, the "12.5% top-1" and "19.2% on
unseen taxa" comparisons against MegaDescriptor, and the per-platform species lists in
section 6.

### 15.4 Sources

Wild Me and Wildbook:

- Wildbook source: https://github.com/WildMeOrg/Wildbook
- Wildbook ML service: https://github.com/WildMeOrg/ml-service
- Wildbook documentation source: https://github.com/WildMeOrg/wildbook-docs
- Wildbook documentation site: https://wildbook.docs.wildme.org/
- MiewID plugin: https://github.com/WildMeOrg/wbia-plugin-miew-id
- Scout (aerial survey annotation, not relevant here): https://github.com/WildMeOrg/scout
- Wild Me: https://www.wildme.org/
- Platform list: https://www.wildme.org/platforms.html
- Conservation X Labs and Wild Me merger:
  https://www.conservationxlabs.com/news/conservation-x-labs-and-wild-me-announce-merger
- Community forum: https://community.wildme.org/
- African Carnivore Wildbook: https://africancarnivore.wildbook.org/
- RWildbook on CRAN: https://cran.r-project.org/web/packages/RWildbook/index.html

Algorithms:

- Multispecies Animal Re-ID Using a Large Community-Curated Dataset (MiewID):
  https://arxiv.org/abs/2412.05602
- MiewID msv3 model card: https://huggingface.co/conservationxlabs/miewid-msv3
- PIE: https://arxiv.org/pdf/1902.10847.pdf
- HotSpotter: http://cs.rpi.edu/hotspotter/crall-hotspotter-wacv-2013.pdf
- WBIA detection pipeline poster (Parham et al., WACV 2018):
  https://cthulhu.dyn.wildme.io/public/posters/parham_wacv_2018.pdf

Camera traps and Wildbook in practice:

- Data Carpentry for Camera Traps, Whiskerbook tutorial:
  https://carpentries-incubator.github.io/camera-traps/04-WhiskerBookTutorial/index.html
- Data Carpentry for Camera Traps, Whiskerbook export:
  https://carpentries-incubator.github.io/camera-traps/05-WhiskerbookExport/index.html
- Identifying individual ocelots by coat patterns using Whiskerbook:
  https://mammalogynotes.org/ojs/index.php/mn/article/download/413/567/

Standards:

- Camtrap DP: https://camtrap-dp.tdwg.org/
- Darwin Core: https://dwc.tdwg.org/terms/

### 15.5 Search queries that produced the useful hits

- `Wildbook Wild Me REST API documentation encounters bulk import`
- `Wild Me Wildbook 2026 status Conservation X Labs maintainers`
- `MiewID msv3 multispecies re-identification model species list Wild Me`
- `Wildbook platforms list Whiskerbook African Carnivore Wildbook camera trap individual identification species`

### 15.6 Open items to settle before building

1. Run the step-zero round trip in section 14 against a real instance and verify the clone
   behaviour by experiment rather than by source reading.
2. Decide: `event_observations` or `detections` for the individual ID column.
3. Decide where the batch record lives: `Job.payload` or a new table.
4. Confirm with Wild Me whether the 200-Encounter guidance is enforced or advisory.
5. Verify the second-hand species and model figures in section 6 against primary text.
6. Establish which Wildbook instances the actual users would be sending to, and whether their
   target species are configured for identification on those instances. If the answer is "none
   of them", stop here.
7. Decide whether video best frames are in scope for a first version.

---

## 16. Plain English summary

Wildbook can put individual animal IDs on camera trap photos, but only for species with
visible individual markings, so leopards, jaguars, ocelots and zebras yes, foxes, badgers and
roe deer no, and that limit is biology rather than a model that needs more training. It will
never be fully automatic either: Wildbook deliberately makes a person confirm every match and
has no auto-assign anywhere in its code. The useful split is that sending data to Wildbook
needs a login and is capped at about 200 records a batch, while reading answers back only
needs a short-lived token the user pastes in. So the recommendation is to push by hand and
pull automatically: AddaxAI builds a zip with the photos and a Wildbook import spreadsheet,
the user drags it into Wildbook and does the identification work there, then pastes a token
into AddaxAI and it fetches the individual IDs back and fills the `individualID` column that
the Camtrap DP export has been writing empty since it was written. The one thing that nearly
broke this is that Wildbook splits a photo of two animals into two records and drops the
external ID when it does, so the join has to use the batch ID and the sighting ID, which are
both values we choose before sending. Almost all of the export machinery already exists as the
Camtrap DP export job. Nothing in the first two phases needs the Wild Me team to change
anything, though there is one genuinely one-line fix worth asking them for.

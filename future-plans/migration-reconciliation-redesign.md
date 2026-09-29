# Replace fingerprint-based migration reconciliation

Plan for a future change. Not scheduled. Captured here so we can pick it up without re-deriving the design.

Owner note: this touches startup code that runs against every user's database on every launch. It deserves its own session, its own plan, and verification against real beta-tester databases before it ships.

## Problem

`backend/app/db/migrations.py` does not trust `alembic_version`. On every startup it tries to work out what revision the schema is *actually* at, by looking for marker columns listed in a hand-maintained table called `SCHEMA_FINGERPRINTS`. If that guess disagrees with `alembic_version`, it re-stamps the database backwards and replays the migration chain forward.

The mechanism was added for a real reason: beta-tester databases got into states where `alembic_version` lied (a historical `stamp_head` bug, half-applied migrations on power loss, hand-restored backups). But the cure has three structural faults, and one of them has already destroyed user data once.

### Fault 1: it asks a question that cannot be answered

"What revision is this schema at?" is not answerable by introspection in the general case. Only `add_column` and `create_table` leave a usable trace. Drops, renames, data backfills, index changes and constraint changes leave nothing, or leave something a later migration erases.

The table shows it. As of 2026-07-27:

```
total fingerprint entries : 24     (one per migration, forever)
  detectable              : 13
  non-detectable          : 11     <- carry no information at all
```

Eleven of twenty-four entries exist purely to satisfy `test_every_alembic_revision_has_a_fingerprint`. They are pure bookkeeping. Every future migration adds another row to this table whether or not it can contribute anything.

### Fault 2: it demands a promise about the future that nobody can keep

A detectable fingerprint says "if column X exists, the schema is at least at revision R". That is only true for as long as no later migration removes column X. You cannot know that when you write it.

**This has already broken.** Revision `f2a3b4c5d6e7` was fingerprinted on `events.verified`. The very next revision, `a3b4c5d6e7f8`, is literally titled "rename events.verified to events.confirmed". So from June 2026 the fingerprint was unsatisfiable on every database at head, `_alembic_version_is_truthful()` returned `False` for every user on every launch, and the app silently took the "the stamp is lying" branch forever. Nobody noticed because the recovery happened to re-stamp at the revision that was already head, so the replay had zero migrations to run.

Fixed 2026-07-27 by repointing that entry at `event_observations.human_count`, plus a new guard `test_every_detectable_fingerprint_is_satisfied_at_head`. That guard converts this class of mistake from a silent runtime failure into a CI failure, which is worth having, but it does not remove the underlying promise. Every future rename or drop can still break an older fingerprint and force someone to go re-point it.

### Fault 3: the recovery path is a loaded gun

This is the serious one. When reconciliation decides the stamp is lying, it re-stamps backwards and replays the chain. The chain contains destructive data migrations. From `f2a3b4c5d6e7`:

```sql
-- overwrites human-entered counts
UPDATE event_observations
SET human_count = max_n
WHERE EXISTS (SELECT 1 FROM detections d ... AND d.verified = 1 ...);

-- unconditional row deletion
DELETE FROM detections WHERE bbox_x IS NULL;
```

Replaying that on a live database overwrites verification work and deletes rows. It cannot be made data-idempotent, because "has the user since edited this count?" is not knowable from the migration.

There is a comment at `migrations.py:537` recording that exactly this replay behaviour destroyed user data on 2026-05-27, when a missing fingerprint sent a healthy database backwards through the chain.

The gun has not fired again only by luck: `detect_schema_revision()` returned the revision that was already head, so the replay window was empty. Adding one migration (the FK index work on 2026-07-27) opened that window by one revision. **The next data migration that lands inside the window would re-run on every affected user, on every launch.**

### Fault 4: the design was built on a misreading

The module docstring justifies itself:

> This mirrors what mature migration tools (Flyway baselining, Django `--fake`, Rails `db:schema:load` + `migrate`) do: trust the introspected schema, not a possibly-wrong stored version.

That is not what those tools do. Verified 2026-07-27:

- **Flyway `baseline`** does not introspect anything. It tags the database with a `baselineVersion` that the operator supplies, once, by hand. See [Flyway baseline version setting](https://documentation.red-gate.com/fd/flyway-baseline-version-setting-277578975.html) and [Baselines](https://documentation.red-gate.com/fd/baselines-273973441.html).
- **Django `migrate --fake`** is a manual operator action that marks migrations applied without running them. (`--fake-initial` does introspect, but only for the *initial* migration, only checks table existence, and is opt-in.)
- **Alembic's own cookbook** has no recipe for reconstructing a revision by introspection. It recommends `command.stamp()` from a *known* state.

All of them baseline **manually and once**. None runs a heuristic re-detection on every startup. There is no upstream precedent for what this module does.

## Proposed design

Stop asking "what revision is this schema at?" (unanswerable, needs bookkeeping, rots). Ask "after upgrading, does the schema match what the code expects?" (exact, needs no bookkeeping, cannot rot).

```
init_db():
  1. Floor check. The ONE introspection rule that survives:
       user tables exist but files.captured_at_local does not
         -> database predates the initial migration (9c173fff3bcd)
         -> refuse, point at Restore / Reset. Do not march the chain forward.

  2. Trust alembic_version. Run upgrade_to_head().        # plain Alembic, no guessing

  3. Verify the result against Base.metadata:
       every table, every column, every index, every FK ondelete action

  4. Match      -> done.
     Mismatch   -> the stamp lied. Back up, log exactly what is missing,
                   and STOP. Surface Restore-from-backup / Reset in the UI.
                   NEVER replay the chain.
```

The load-bearing change is step 4. **Never auto-replay.** That single rule deletes the data-loss mechanism outright, and it makes migration idempotency stop mattering for recovery. AddaxAI already ships backup, restore and reset UI (`BackupNowDialog`, `RestoreBackupDialog`, `AppHamburger`), so there is a safe human-in-the-loop escape hatch. A desktop app should hand a corrupt database back to its owner with a precise error, not silently rewrite their verifications guessing at a fix.

Step 3's comparison primitives already exist and are proven: `test_upgrade_from_base_matches_models` (tables and columns), plus `test_upgrade_from_base_creates_every_model_index` and `test_upgrade_from_base_preserves_fk_ondelete_actions` (both added 2026-07-27). The runtime check is the same walk over `Base.metadata` versus `inspect(engine)`. Keep it deliberately loose on things SQLite fudges (server defaults, type affinity); a skipped migration essentially always shows up as a missing table, column or index.

## What this deletes

- `SCHEMA_FINGERPRINTS` (24 entries, growing by one per migration forever)
- `_Fingerprint`, `_fingerprint_satisfied`, `detect_schema_revision`, `_alembic_version_is_truthful`
- the "pick a marker column" step in the add-a-migration workflow, and the DEVELOPERS.md text describing it
- `test_every_alembic_revision_has_a_fingerprint`, `test_fingerprints_are_in_chronological_order`, `test_every_detectable_fingerprint_is_satisfied_at_head`
- the promise that every fingerprint column survives all future migrations

What replaces it: one floor constant, one verification function, and `Base.metadata` as the only source of truth for what the schema should be.

## Existing behaviour that must be preserved

Do not lose these; they are all covered by tests in `backend/tests/db/test_migrations.py` and were added in response to real incidents.

| behaviour | why it exists |
|---|---|
| Pre-floor databases are refused, not upgraded | issue #11, Arky's Linux install: marching the chain forward died with `KeyError: 'captured_at_local'` |
| A legacy database with no `alembic_version` row still gets adopted | beta databases that predate the runtime alembic wiring |
| Fresh install builds the whole schema from base | normal first run |
| A pre-upgrade backup is taken before any migration runs | `main.py` lifespan, never auto-pruned |
| Startup crashes loudly rather than continuing on a broken schema | "crash early" (CONVENTIONS.md) |

Note the ordering trap found on 2026-07-27: the lifespan checks `needs_upgrade()` **before** `init_db()`, so any re-stamp that happens inside `init_db()` is invisible to the backup decision. If the redesign moves when the stamp changes, re-check that the pre-upgrade backup still fires when it should.

## Test plan

1. Port every scenario in `test_migrations.py` to the new flow: fresh install, legacy-no-version, pre-floor refusal, healthy at head, stamp-ahead-of-schema, stamp-behind-schema.
2. New: a database stamped at head but missing a column must produce a clear error naming the column, and must **not** run any migration.
3. New: run the whole chain from base twice and assert the second run succeeds and leaves the schema identical (DDL idempotency, still worth enforcing even though recovery no longer depends on it).
4. Keep the three model-versus-schema guards as CI tests; they become the specification for the runtime check.
5. Verify against real databases: a copy of `~/AddaxAI/addaxai.db`, plus any beta-tester database available. Check row counts, `PRAGMA integrity_check`, and `PRAGMA foreign_key_check` before and after.

## Risks and open questions

- **Losing self-healing.** Today a mis-stamped database silently repairs itself. Afterwards the user sees an error and has to restore or reset. That is a deliberate trade: silent repair is what destroyed data. Worth confirming the wording of that error is something a field ecologist can act on.
- **How loose should verification be?** Too strict and SQLite quirks (type affinity, server defaults) produce false alarms on healthy databases; too loose and a skipped migration slips through. Start with tables, columns, index names and FK ondelete actions, which is what the CI guards already compare.
- **Are there users mis-stamped right now?** Unknown. Worth adding a one-off diagnostic to the report export that logs any model-versus-schema mismatch, and shipping that ahead of the redesign so we learn the real prevalence first.
- **Do not bundle this with anything else.** It changes startup for every user.

## Effort

Roughly one focused session: the module is about 600 lines and the reconcile logic is maybe 150 of them, the verification primitives already exist as tests, and the test file already contains the scenarios. The care goes into the real-database verification, not the code.

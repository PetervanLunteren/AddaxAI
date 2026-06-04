# Site + Deployment location cleanup

Summary of the refactor that reshapes how deployments are attached to
locations. Written 2026-04-20.

## Why

The old model had three tangled states hiding behind one pattern:

1. A real site with GPS coordinates (e.g. "North ridge" at 44.4, -110.5).
2. A real site without GPS (e.g. "North ridge", coords TBD).
3. The user didn't want to pick a site (unknown or multiple locations).

Cases 2 and 3 were both routed through an auto-created placeholder
Site called "Unknown" with NULL coords. That meant banners like "Add
GPS to Unknown" asked the user to put coordinates onto a placeholder
that was never a real location — polluted metadata plus bad UX.

## The new two-state model

- **Deployment with a site**: the site is a real georeferenced camera
  location. Always has valid GPS.
- **Deployment without a site** (`site_id = NULL`): the user left the
  Site field blank because the location is unknown or the folder
  contains files from multiple locations.

There is no third state. Sites without GPS no longer exist.

## What changed

### Database

- `deployments.project_id` is now a direct NOT NULL FK to
  `projects.id`. Before, deployments inherited their project through
  the site join. Now they belong to a project directly, so a
  deployment with `site_id = NULL` still has a clear project.
- `deployments.site_id` is nullable. `NULL` means "no site assigned".
- The `ON DELETE` behaviour on `deployments.site_id` switched from
  `CASCADE` to `SET NULL`. Deleting a site now orphans its
  deployments into the no-site state rather than destroying them.
- `sites.latitude` and `sites.longitude` are NOT NULL. Every site has
  real coordinates.
- Migrations that handle the data rollover:
  - `b8c9d0e1f2a3_nullable_deployment_site`: backfills `project_id`,
    makes `site_id` nullable, re-points deployments attached to the
    old auto-created "Unknown" sites, and deletes those sites.
  - `c9d0e1f2a3b4_require_site_coords`: deletes sites with NULL coords
    or exactly `(0, 0)` (the buggy old default), re-points their
    deployments to `site_id=NULL`, then adds the NOT NULL constraint
    on lat/lon and switches the FK to `SET NULL`.

### Backend

- `Deployment` model, `DeploymentCreate` / `DeploymentResponse` /
  `DeploymentInfoResponse` schemas updated: `project_id` is required,
  `site_id` / `site_name` are nullable.
- `Site` model + `SiteBase` / `SiteResponse` / `SiteInfoResponse`
  schemas updated: `latitude` and `longitude` are required.
- Null-Island guard: `SiteBase` and `SiteUpdate` now reject exactly
  `(0, 0)` via a Pydantic `model_validator` so the API returns 422
  when a client bypasses the form.
- CRUD and routers scope deployments by `Deployment.project_id`
  directly instead of joining through `Site`. This is why no-site
  deployments still belong to their project.
- Dashboard overview totals (files, events, deployments,
  observations, trap nights) count no-site deployments too, since they
  are still real deployments belonging to the project.
- Worker (`detection_worker.create_deployment`) takes `project_id` +
  optional `site_id`; dropped the old "every queue entry must have a
  site" validation.

### Frontend

- **AddDeploymentCard**: dropped the `ensureSiteId()` helper that used
  to get-or-create an "Unknown" site. Now it just sends
  `site_id: siteId ?? null`. The `sitesApi.create` call is gone.
- **SiteSelector**: placeholder is
  *"Leave blank if unknown or from multiple locations"*. Tooltip
  matches. When a site list is empty it shows
  *"No sites yet. Click `+` to add one."* (the `+` is rendered as an
  inline code pill so it doesn't collide with the em dash).
- **AddSiteModal**: coordinates are now required. Fields default to
  blank (no more `(0, 0)` pin at Null Island). The submit button
  stays disabled until both lat/lon are present and the map pin is
  placed. The zod schema rejects exactly `(0, 0)` with an inline
  "Pick a real location" error. The old "Coordinates are optional"
  help copy is gone.
- **EditDeploymentDialog**: the Site picker now has an extra
  *"No site (unknown / multiple)"* item that writes `site_id = null`.
  Radix `Select` can't bind to null so it uses a `__none__` sentinel
  under the hood.
- **DeploymentsPage**:
  - Site column renders `—` (em dash, muted) when a deployment has no
    site assigned.
  - New `site` filter with options *"No site assigned"* /
    *"Site assigned"*. Deep-linked from the banner on Map / Export
    as `?site=missing`.
  - Removed the old `gps` filter: every site has GPS now, so the only
    "missing-GPS" state is "no site".

### Banners on Map + Export

Both pages previously rendered two amber banners (one for
sites-without-GPS, one for no-site deployments). Since sites now
always have GPS, only one banner remains:

- **`NoSiteBanner`** (`frontend/src/components/sites/NoSiteBanner.tsx`):
  amber banner shown on Map, Export → Spatial, and Export →
  Camtrap-DP when the project has any deployment with `site_id =
  NULL`. Copy reads e.g. *"3 deployments aren't on the map. They
  have no site assigned."* with an
  *[→ Assign a site]* button that deep-links to
  `/projects/:id/deployments?site=missing`.
- The **Spatial** and **Camtrap-DP** download buttons stay disabled
  while that banner is active. Obs-CSV export is unaffected.
- The old `MissingCoordsBanner` file was deleted.

### Types

- `SiteResponse.latitude` / `longitude`: `number` (was `number | null`).
- `SiteInfo.latitude` / `longitude`: same.
- `DeploymentResponse.project_id`: `string` (new field).
- `DeploymentResponse.site_id`: `string | null` (was `string`).
- `DeploymentInfo.site_id` / `site_name`: both nullable.
- `DeploymentUpdate.site_id`: `string | null | undefined`.
  `undefined` means "don't touch", `null` means "clear the site".
- `DeleteDeploymentTarget.site_name`: nullable, shown as
  *"no site"* in the delete confirmation when absent.

## Verification

- Backend: 553 tests pass, `ruff check` clean.
- Frontend: `npx tsc` shows the same pre-existing error count as
  before the refactor (no new type errors introduced). `npx vite
  build` succeeds.
- Manual smoke path:
  1. Analyses page → pick a folder, leave Site blank, queue.
     Expect: queue accepts, no placeholder "Unknown" site is
     created, Deployments table shows the row with Site = `—`.
  2. Map page shows the new amber `NoSiteBanner`.
  3. Click *Assign a site* → lands on
     `/deployments?site=missing` with the row visible; edit →
     assign → save → banner disappears.
  4. Try to add a new site at lat=0, lon=0: blocked with inline
     error.

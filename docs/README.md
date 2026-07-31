# AddaxAI documentation site

The AddaxAI docs, built with [Docusaurus](https://docusaurus.io/). Deployed to
GitHub Pages at https://docs.addaxai.com/ by
`.github/workflows/docs.yml` on every push to `main` that touches `docs/`.

## Edit locally (do this, not push-and-wait)

To work on the docs, run them locally. You do **not** need to push and wait for
the GitHub Action to see your changes.

```bash
cd docs
nvm use 20
npm install      # first time only
npm start
```

This opens a live preview at http://localhost:3000/ that reloads
the moment you save a file. Editing text, CSS or a component updates in about a
second. `Ctrl+C` to stop.

Two things `npm start` does not pick up live:

- **`docusaurus.config.ts`** (navbar, footer, sidebar): stop and restart.
- **Broken links**: that check only runs at build time. Before you push, run
  one `npm run build` to catch dead links, since the deploy fails on them.

So: `npm start` while writing, one `npm run build` before you push.

## Add content

- **New page**: drop a `.md` or `.mdx` file into a folder under `docs/`. It
  appears in the sidebar automatically. Order it with `sidebar_position` in the
  frontmatter.
- **New section**: create a folder under `docs/` with a `_category_.json` file
  (copy an existing one). `position` controls where it sits in the sidebar.
- **App screenshots**: do not commit these. Add the image to the manifest at
  `src/data/screenshots.ts` (an external URL), then place it with
  `<AppShot name="..." />` in an `.mdx` page. Source PNGs live in
  `~/Desktop/addaxai-docs-screenshots/upload`.
- **Other images** (logos, diagrams): put them in `static/img/` and reference
  them as `![alt](/img/your-image.png)`.
- **Diagrams**: use a ```mermaid code fence (already enabled).
- **Interactive widgets**: write a React component under `src/components/` and
  import it into an `.mdx` page. `src/components/ModelZoo` is the worked example:
  a filterable, searchable table driven by the app's real `models.json`.
- **Per-system steps**: use `<Tabs groupId="os">` so the choice syncs across
  the page. See `docs/start-here/install.mdx`.

Broken internal links fail the build (`onBrokenLinks: throw`), so keep links
pointing at real files.

## How the model zoo gets its data

Two files feed it, and they work differently on purpose.

`scripts/sync-models.mjs` copies the repo-root `models.json` into
`src/data/models.json` before every `start` and `build` (via the npm
`prestart` / `prebuild` hooks). That generated file is gitignored, so the table
always reflects the real catalogue and never drifts. If your editor complains
that `src/data/models.json` is missing, run `npm run sync-models` once.

`scripts/fetch-species.mjs` downloads each classification model's
`taxonomy.csv` from HuggingFace and writes the species lists to
`src/data/species.json`. It runs on the same `prestart` / `prebuild` hooks, so
adding a model to the catalogue is all you have to do: the next build picks up
its species by itself.

That output **is** committed, unlike `models.json`, because it is also the
fallback. If HuggingFace is unreachable the script keeps the committed list for
each model it could not refresh, prints a `::warning::` that GitHub Actions
shows in the run summary, and exits 0. An outage therefore never blocks a
deploy and never empties the table. A model with no data at all, new and
unfetchable, is listed in `meta.unavailable` and the table says its species
list could not be fetched.

Because the committed copy is the safety net, commit what a good run writes
now and then, so the fallback does not drift far behind the catalogue.

## Build

```bash
npm run build      # outputs static site to build/
npm run serve      # preview the production build locally
```

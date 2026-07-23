# AddaxAI documentation site

The AddaxAI docs, built with [Docusaurus](https://docusaurus.io/). Deployed to
GitHub Pages at https://petervanlunteren.github.io/AddaxAI-WebUI/ by
`.github/workflows/docs.yml` on every push to `main` that touches `docs/`.

## Run locally

```bash
cd docs
nvm use 20
npm install
npm start
```

This opens a live-reloading preview at http://localhost:3000/AddaxAI-WebUI/.

## Add content

- **New page**: drop a `.md` or `.mdx` file into a folder under `docs/`. It
  appears in the sidebar automatically. Order it with `sidebar_position` in the
  frontmatter.
- **New section**: create a folder under `docs/` with a `_category_.json` file
  (copy an existing one). `position` controls where it sits in the sidebar.
- **Images**: put them in `static/img/` and reference them as
  `![alt](/img/your-image.png)`.
- **Diagrams**: use a ```mermaid code fence (already enabled).
- **Interactive widgets**: write a React component under `src/components/` and
  import it into an `.mdx` page. `src/components/ModelZoo` is the worked example:
  a filterable, searchable table driven by the app's real `models.json`.

## How the model zoo gets its data

`scripts/sync-models.mjs` copies the repo-root `models.json` into
`src/data/models.json` before every `start` and `build` (via the npm
`prestart` / `prebuild` hooks). That generated file is gitignored, so the table
always reflects the real catalogue and never drifts. If your editor complains
that `src/data/models.json` is missing, run `npm run sync-models` once.

## Build

```bash
npm run build      # outputs static site to build/
npm run serve      # preview the production build locally
```

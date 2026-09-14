// Builds the list of sections and pages the documentation home page shows,
// from the same files the sidebar is generated from: one folder per section
// with a _category_.json, one .md/.mdx per page with its frontmatter.
//
// The home page used to carry this list by hand and drifted: 8 of 27 pages
// were missing by 2026-09-14, among them every install-related help page.
// Generating it means a page dropped into docs/ shows up on the home page
// the same way it shows up in the sidebar, with nothing to remember.
//
// Runs automatically before `npm start` and `npm build` (via the prestart /
// prebuild npm hooks). The output is gitignored on purpose.
import { readdirSync, readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import matter from "gray-matter"; // what Docusaurus itself parses frontmatter with

const here = dirname(fileURLToPath(import.meta.url));
const docsDir = resolve(here, "../docs");
const dest = resolve(here, "../src/data/pages.json");

// Mirrors `routeBasePath` in docusaurus.config.ts.
const ROUTE_BASE = "/docs";

function fail(message) {
  throw new Error(`[sync-pages] ${message}`);
}

function readPage(section, file) {
  const path = join(docsDir, section, file);
  const { data } = matter(readFileSync(path, "utf8"));
  // Docusaurus derives the route from the file path unless one of these
  // overrides it. None of our pages use them; if one ever does, this has
  // to learn the rule rather than silently link to the wrong place.
  for (const key of ["slug", "id"]) {
    if (key in data) fail(`${path} sets "${key}", which this script does not handle`);
  }
  if (data.draft || data.unlisted) return null;
  if (!data.title) fail(`${path} has no title in its frontmatter`);
  if (typeof data.sidebar_position !== "number") {
    fail(`${path} has no numeric sidebar_position in its frontmatter`);
  }
  return {
    label: data.sidebar_label ?? data.title,
    to: `${ROUTE_BASE}/${section}/${file.replace(/\.mdx?$/, "")}`,
    position: data.sidebar_position,
  };
}

function readSection(section) {
  const categoryPath = join(docsDir, section, "_category_.json");
  let category;
  try {
    category = JSON.parse(readFileSync(categoryPath, "utf8"));
  } catch {
    fail(`${categoryPath} is missing or not valid JSON`);
  }
  const description = category.link?.description;
  if (!category.label || typeof category.position !== "number" || !description) {
    fail(`${categoryPath} needs label, position and link.description`);
  }
  const pages = readdirSync(join(docsDir, section))
    .filter((f) => /\.mdx?$/.test(f))
    .map((f) => readPage(section, f))
    .filter(Boolean)
    .sort((a, b) => a.position - b.position);
  if (pages.length === 0) fail(`${section}/ has no pages`);
  return {
    label: category.label,
    description,
    position: category.position,
    // The card title links to the first page, not the generated index:
    // one click should land on content.
    to: pages[0].to,
    pages: pages.map(({ label, to }) => ({ label, to })),
  };
}

const sections = readdirSync(docsDir, { withFileTypes: true })
  .filter((d) => d.isDirectory())
  .map((d) => readSection(d.name))
  .sort((a, b) => a.position - b.position)
  .map(({ position, ...rest }) => rest);

mkdirSync(dirname(dest), { recursive: true });
writeFileSync(dest, JSON.stringify(sections, null, 2) + "\n");

const total = sections.reduce((n, s) => n + s.pages.length, 0);
console.log(`[sync-pages] ${sections.length} sections, ${total} pages -> ${dest}`);

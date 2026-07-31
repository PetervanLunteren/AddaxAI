import { useMemo, useState, type ReactElement } from "react";
import modelsData from "@site/src/data/models.json";
import speciesData from "@site/src/data/species.json";
import styles from "./styles.module.css";

// Interactive, filterable model catalogue. Data comes from the repo-root
// models.json (synced into src/data at build time), so this table always
// matches what ships in the app. This is the reference pattern for future
// interactive pages (maps, charts, filters): a plain React component dropped
// into an .mdx page.
//
// Rows expand to show the model's full description plus its licence and
// citation. The table itself stays narrow enough to read without sideways
// scrolling; anything long or rarely needed lives in the expanded panel.
// This mirrors the app's own model info sheet, so docs and app agree.

type ModelType = "det" | "cls" | "emb";

interface RawModel {
  model_id: string;
  friendly_name: string;
  emoji?: string;
  developer?: string;
  owner?: string;
  region?: string;
  description?: string;
  description_short?: string;
  info_url?: string;
  license?: string;
  citation?: string;
  min_app_version?: string;
}

interface Row extends RawModel {
  type: ModelType;
}

const TYPE_LABEL: Record<ModelType, string> = {
  det: "Detection",
  cls: "Classification",
  emb: "Embedding",
};

const data = modelsData as { models: Record<ModelType, RawModel[]> };

// Classification first: it is the choice most people come here to make.
const ALL_ROWS: Row[] = (["cls", "det", "emb"] as ModelType[]).flatMap((type) =>
  (data.models[type] ?? []).map((m) => ({ ...m, type })),
);

const REGIONS: string[] = Array.from(
  new Set(ALL_ROWS.map((r) => r.region).filter((r): r is string => Boolean(r))),
).sort();

// Species a classification model can predict, from src/data/species.json
// (see scripts/fetch-species.mjs, which refreshes it on every build).
// Underscores are how the model files spell them; readers should not have to
// decode that. `unavailable` lists models whose list could not be fetched and
// had no committed copy to fall back on, so the panel can say so rather than
// look as though the model knows nothing.
const SPECIES = (speciesData as { species: Record<string, string[]> }).species;
const UNAVAILABLE = new Set(
  (speciesData as { meta?: { unavailable?: string[] } }).meta?.unavailable ?? [],
);

/** How many species to print before collapsing the tail into a count. */
const SPECIES_SHOWN = 60;

function prettySpecies(name: string): string {
  return name.replace(/_/g, " ");
}

// The catalogue stores regions lowercase ("europe"). They are proper nouns,
// so display them capitalised while filtering on the stored value.
function prettyRegion(region: string): string {
  return region.charAt(0).toUpperCase() + region.slice(1);
}

// Search matches species too, so "wolverine" answers "which models know
// this animal?" — the question people actually arrive with. Precomputed
// once because it runs against every row on every keystroke.
const HAYSTACK = new Map<string, string>(
  ALL_ROWS.map((r) => [
    `${r.type}-${r.model_id}`,
    [
      r.friendly_name,
      r.model_id,
      r.developer,
      r.description_short,
      r.description,
      r.region,
      ...(SPECIES[r.model_id] ?? []).map(prettySpecies),
    ]
      .filter(Boolean)
      .join(" ")
      .toLowerCase(),
  ]),
);

export default function ModelZoo(): ReactElement {
  const [query, setQuery] = useState("");
  const [typeFilter, setTypeFilter] = useState<ModelType | "all">("all");
  const [regionFilter, setRegionFilter] = useState<string>("all");
  const [expanded, setExpanded] = useState<Set<string>>(new Set());

  // Only classification models carry a region, so the control and the column
  // are pointless while the other two types are selected.
  const regionApplies = typeFilter === "all" || typeFilter === "cls";

  const rows = useMemo(() => {
    const q = query.trim().toLowerCase();
    return ALL_ROWS.filter((r) => {
      if (typeFilter !== "all" && r.type !== typeFilter) return false;
      if (regionApplies && regionFilter !== "all" && r.region !== regionFilter) {
        return false;
      }
      if (!q) return true;
      return (HAYSTACK.get(`${r.type}-${r.model_id}`) ?? "").includes(q);
    });
  }, [query, typeFilter, regionFilter, regionApplies]);

  const typeButtons: Array<[ModelType | "all", string]> = [
    ["all", "All"],
    ["cls", "Classification"],
    ["det", "Detection"],
    ["emb", "Embedding"],
  ];

  const columnCount = regionApplies ? 5 : 4;

  function toggle(key: string): void {
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  }

  return (
    <div className={styles.root}>
      <div className={styles.controls}>
        <input
          type="search"
          className={styles.search}
          placeholder="Search species, models, developers, regions…"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          aria-label="Search models"
        />

        <div className={styles.typeGroup} role="group" aria-label="Filter by type">
          {typeButtons.map(([value, label]) => (
            <button
              key={value}
              type="button"
              className={
                typeFilter === value
                  ? `${styles.typeButton} ${styles.typeButtonActive}`
                  : styles.typeButton
              }
              onClick={() => setTypeFilter(value)}
            >
              {label}
            </button>
          ))}
        </div>

        {/* Stays put when it does not apply, rather than disappearing and
            shifting the controls next to it. Only classification models
            carry a region. */}
        <select
          className={styles.region}
          value={regionFilter}
          onChange={(e) => setRegionFilter(e.target.value)}
          disabled={!regionApplies}
          title={
            regionApplies ? undefined : "Only classification models have a region"
          }
          aria-label="Filter by region"
        >
          <option value="all">All regions</option>
          {REGIONS.map((region) => (
            <option key={region} value={region}>
              {prettyRegion(region)}
            </option>
          ))}
        </select>
      </div>

      <div className={styles.count}>
        {rows.length} {rows.length === 1 ? "model" : "models"}
        {rows.length > 0 ? ", select one to read more" : ""}
      </div>

      <div className={styles.tableWrap}>
        <table className={styles.table}>
          <thead>
            <tr>
              <th>Model</th>
              <th>Type</th>
              <th>Developer</th>
              {regionApplies ? <th>Region</th> : null}
              <th>Summary</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => {
              const key = `${row.type}-${row.model_id}`;
              const isOpen = expanded.has(key);
              return [
                <tr key={key} className={isOpen ? styles.rowOpen : undefined}>
                  <td>
                    <button
                      type="button"
                      className={styles.nameButton}
                      onClick={() => toggle(key)}
                      aria-expanded={isOpen}
                    >
                      <span
                        className={isOpen ? styles.caretOpen : styles.caret}
                        aria-hidden="true"
                      >
                        ›
                      </span>
                      <span className={styles.name}>
                        {row.emoji ? <span>{row.emoji}</span> : null}
                        <span>{row.friendly_name}</span>
                      </span>
                    </button>
                    <code className={styles.modelId}>{row.model_id}</code>
                  </td>
                  <td>
                    <span className={styles.badge}>{TYPE_LABEL[row.type]}</span>
                  </td>
                  <td>{row.developer ?? "—"}</td>
                  {regionApplies ? (
                    <td>{row.region ? prettyRegion(row.region) : "—"}</td>
                  ) : null}
                  <td className={styles.summary}>{row.description_short ?? "—"}</td>
                </tr>,
                isOpen ? (
                  <tr key={`${key}-detail`} className={styles.detailRow}>
                    <td colSpan={columnCount}>
                      <div className={styles.detail}>
                        {row.description ? <p>{row.description}</p> : null}
                        {(() => {
                          const species = SPECIES[row.model_id];
                          if (!species || species.length === 0) {
                            return UNAVAILABLE.has(row.model_id) ? (
                              <p className={styles.speciesMissing}>
                                The species list could not be fetched for this
                                model. Open it in the app to see what it knows.
                              </p>
                            ) : null;
                          }
                          const shown = species.slice(0, SPECIES_SHOWN);
                          const rest = species.length - shown.length;
                          return (
                            <div className={styles.species}>
                              <div className={styles.speciesHead}>
                                Knows {species.length}{" "}
                                {species.length === 1 ? "label" : "labels"}
                              </div>
                              <p className={styles.speciesList}>
                                {shown.map(prettySpecies).join(", ")}
                                {rest > 0 ? `, and ${rest} more` : ""}
                              </p>
                            </div>
                          );
                        })()}
                        <dl className={styles.meta}>
                          {row.owner ? (
                            <>
                              <dt>Owner</dt>
                              <dd>{row.owner}</dd>
                            </>
                          ) : null}
                          {row.info_url ? (
                            <>
                              <dt>More info</dt>
                              <dd>
                                <a href={row.info_url} target="_blank" rel="noreferrer">
                                  {row.info_url}
                                </a>
                              </dd>
                            </>
                          ) : null}
                          {row.license ? (
                            <>
                              <dt>Licence</dt>
                              <dd>
                                <a href={row.license} target="_blank" rel="noreferrer">
                                  {row.license}
                                </a>
                              </dd>
                            </>
                          ) : null}
                          {row.citation ? (
                            <>
                              <dt>Cite</dt>
                              <dd>
                                <a href={row.citation} target="_blank" rel="noreferrer">
                                  {row.citation}
                                </a>
                              </dd>
                            </>
                          ) : null}
                        </dl>
                      </div>
                    </td>
                  </tr>
                ) : null,
              ];
            })}
            {rows.length === 0 ? (
              <tr>
                <td colSpan={columnCount} className={styles.empty}>
                  No models match your filters.
                </td>
              </tr>
            ) : null}
          </tbody>
        </table>
      </div>
    </div>
  );
}

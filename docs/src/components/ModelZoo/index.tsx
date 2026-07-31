import { useMemo, useState, type ReactElement } from "react";
import modelsData from "@site/src/data/models.json";
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

const ALL_ROWS: Row[] = (["det", "cls", "emb"] as ModelType[]).flatMap((type) =>
  (data.models[type] ?? []).map((m) => ({ ...m, type })),
);

const REGIONS: string[] = Array.from(
  new Set(ALL_ROWS.map((r) => r.region).filter((r): r is string => Boolean(r))),
).sort();

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
      const haystack = [
        r.friendly_name,
        r.model_id,
        r.developer,
        r.description_short,
        r.description,
        r.region,
      ]
        .filter(Boolean)
        .join(" ")
        .toLowerCase();
      return haystack.includes(q);
    });
  }, [query, typeFilter, regionFilter, regionApplies]);

  const typeButtons: Array<[ModelType | "all", string]> = [
    ["all", "All"],
    ["det", "Detection"],
    ["cls", "Classification"],
    ["emb", "Embedding"],
  ];

  const columnCount = regionApplies ? 7 : 6;

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
          placeholder="Search models, developers, regions…"
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

        {regionApplies ? (
          <select
            className={styles.region}
            value={regionFilter}
            onChange={(e) => setRegionFilter(e.target.value)}
            aria-label="Filter by region"
          >
            <option value="all">All regions</option>
            {REGIONS.map((region) => (
              <option key={region} value={region}>
                {region}
              </option>
            ))}
          </select>
        ) : null}
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
              <th>Min app</th>
              <th>Info</th>
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
                  {regionApplies ? <td>{row.region ?? "—"}</td> : null}
                  <td className={styles.summary}>{row.description_short ?? "—"}</td>
                  <td>{row.min_app_version ?? "—"}</td>
                  <td>
                    {row.info_url ? (
                      <a href={row.info_url} target="_blank" rel="noreferrer">
                        Link
                      </a>
                    ) : (
                      "—"
                    )}
                  </td>
                </tr>,
                isOpen ? (
                  <tr key={`${key}-detail`} className={styles.detailRow}>
                    <td colSpan={columnCount}>
                      <div className={styles.detail}>
                        {row.description ? <p>{row.description}</p> : null}
                        <dl className={styles.meta}>
                          {row.owner ? (
                            <>
                              <dt>Owner</dt>
                              <dd>{row.owner}</dd>
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

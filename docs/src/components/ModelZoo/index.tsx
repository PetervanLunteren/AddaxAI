import { useMemo, useState, type ReactElement } from "react";
import modelsData from "@site/src/data/models.json";
import styles from "./styles.module.css";

// Interactive, filterable model catalogue. Data comes from the repo-root
// models.json (synced into src/data at build time), so this table always
// matches what ships in the app. This is the reference pattern for future
// interactive pages (maps, charts, filters): a plain React component dropped
// into an .mdx page.

type ModelType = "det" | "cls" | "emb";

interface RawModel {
  model_id: string;
  friendly_name: string;
  emoji?: string;
  developer?: string;
  region?: string;
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

function LinkCell({ row }: { row: Row }): ReactElement {
  const links: Array<[string, string | undefined]> = [
    ["Info", row.info_url],
    ["License", row.license],
    ["Cite", row.citation],
  ];
  return (
    <span className={styles.links}>
      {links
        .filter(([, href]) => Boolean(href))
        .map(([label, href]) => (
          <a key={label} href={href} target="_blank" rel="noreferrer">
            {label}
          </a>
        ))}
    </span>
  );
}

export default function ModelZoo(): ReactElement {
  const [query, setQuery] = useState("");
  const [typeFilter, setTypeFilter] = useState<ModelType | "all">("all");
  const [regionFilter, setRegionFilter] = useState<string>("all");

  const rows = useMemo(() => {
    const q = query.trim().toLowerCase();
    return ALL_ROWS.filter((r) => {
      if (typeFilter !== "all" && r.type !== typeFilter) return false;
      if (regionFilter !== "all" && r.region !== regionFilter) return false;
      if (!q) return true;
      const haystack = [
        r.friendly_name,
        r.model_id,
        r.developer,
        r.description_short,
        r.region,
      ]
        .filter(Boolean)
        .join(" ")
        .toLowerCase();
      return haystack.includes(q);
    });
  }, [query, typeFilter, regionFilter]);

  const typeButtons: Array<[ModelType | "all", string]> = [
    ["all", "All"],
    ["det", "Detection"],
    ["cls", "Classification"],
    ["emb", "Embedding"],
  ];

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
      </div>

      <div className={styles.count}>
        {rows.length} {rows.length === 1 ? "model" : "models"}
      </div>

      <div className={styles.tableWrap}>
        <table className={styles.table}>
          <thead>
            <tr>
              <th>Model</th>
              <th>Type</th>
              <th>Developer</th>
              <th>Region</th>
              <th>Summary</th>
              <th>Min app</th>
              <th>Links</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={`${row.type}-${row.model_id}`}>
                <td>
                  <div className={styles.name}>
                    {row.emoji ? <span>{row.emoji}</span> : null}
                    <span>{row.friendly_name}</span>
                  </div>
                  <code className={styles.modelId}>{row.model_id}</code>
                </td>
                <td>
                  <span className={`${styles.badge} ${styles[`badge_${row.type}`]}`}>
                    {TYPE_LABEL[row.type]}
                  </span>
                </td>
                <td>{row.developer ?? "—"}</td>
                <td>{row.region ?? "—"}</td>
                <td className={styles.summary}>{row.description_short ?? "—"}</td>
                <td>{row.min_app_version ?? "—"}</td>
                <td>
                  <LinkCell row={row} />
                </td>
              </tr>
            ))}
            {rows.length === 0 ? (
              <tr>
                <td colSpan={7} className={styles.empty}>
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

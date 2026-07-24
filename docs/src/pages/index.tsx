import type { ReactNode } from "react";
import Layout from "@theme/Layout";
import Link from "@docusaurus/Link";
import useBaseUrl from "@docusaurus/useBaseUrl";
import styles from "./index.module.css";

// Documentation home. This site explains how AddaxAI works and what the
// numbers mean. It is not a sales page: the job here is to get a reader to
// the right page in one click.

interface Card {
  title: string;
  body: string;
  to: string;
  links: Array<{ label: string; to: string }>;
}

const SECTIONS: Card[] = [
  {
    title: "Start here",
    body: "New to AddaxAI. Install it, run it once, and pick the way of working that fits your project.",
    to: "/docs/start-here/what-is-addaxai",
    links: [
      { label: "What AddaxAI does", to: "/docs/start-here/what-is-addaxai" },
      { label: "Install", to: "/docs/start-here/install" },
      { label: "Choose a workflow", to: "/docs/start-here/choose-a-workflow" },
    ],
  },
  {
    title: "Guides",
    body: "Step by step for the main tasks: run an analysis, check labels, confirm counts, get your data out.",
    to: "/docs/guides/analyse-a-folder",
    links: [
      { label: "Analyse a folder", to: "/docs/guides/analyse-a-folder" },
      { label: "Build a project", to: "/docs/guides/build-a-project" },
      { label: "Check the labels", to: "/docs/guides/check-labels" },
      { label: "Export your results", to: "/docs/guides/export-results" },
    ],
  },
  {
    title: "Understanding your results",
    body: "Where the numbers come from. Read this before you use the output in a paper or a report.",
    to: "/docs/understanding/detections-events-observations",
    links: [
      { label: "Detections, events and observations", to: "/docs/understanding/detections-events-observations" },
      { label: "How trap nights are counted", to: "/docs/understanding/trap-nights" },
      { label: "Confidence and verification", to: "/docs/understanding/confidence-and-verification" },
      { label: "The charts explained", to: "/docs/understanding/insights" },
    ],
  },
  {
    title: "Reference",
    body: "Look things up: every column in the exports, every model, and what each setting changes.",
    to: "/docs/reference/export-columns",
    links: [
      { label: "Export columns", to: "/docs/reference/export-columns" },
      { label: "Model zoo", to: "/docs/reference/model-zoo" },
      { label: "Settings", to: "/docs/reference/settings" },
      { label: "Where your files live", to: "/docs/reference/file-locations" },
    ],
  },
];

function Hero(): ReactNode {
  const logo = useBaseUrl("/img/logo-wordmark.png");
  return (
    <header className={styles.hero}>
      <div className={styles.heroInner}>
        <img className={styles.heroLogo} src={logo} alt="AddaxAI" />
        <h1 className={styles.heroTitle}>AddaxAI documentation</h1>
        <p className={styles.heroSub}>
          How the app works, what each screen does, and where the numbers come
          from. Use the search box at the top if you already know what you are
          looking for.
        </p>
      </div>
    </header>
  );
}

function Sections(): ReactNode {
  return (
    <section className={styles.section}>
      <div className={styles.wide}>
        <div className={styles.cards}>
          {SECTIONS.map((c) => (
            <div key={c.title} className={styles.card}>
              <h2 className={styles.cardTitle}>
                <Link to={c.to}>{c.title}</Link>
              </h2>
              <p className={styles.cardBody}>{c.body}</p>
              <ul className={styles.cardLinks}>
                {c.links.map((l) => (
                  <li key={l.to}>
                    <Link to={l.to}>{l.label}</Link>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

function Popular(): ReactNode {
  const items = [
    { label: "Why is my trap night count different from what I expected?", to: "/docs/understanding/trap-nights" },
    { label: "What is the difference between a detection and an observation?", to: "/docs/understanding/detections-events-observations" },
    { label: "What does each column in the CSV mean?", to: "/docs/reference/export-columns" },
    { label: "Which workflow should I use?", to: "/docs/start-here/choose-a-workflow" },
    { label: "Some photos have no date. What happens to them?", to: "/docs/understanding/capture-times" },
    { label: "Something went wrong", to: "/docs/troubleshooting/" },
  ];
  return (
    <section className={`${styles.section} ${styles.sectionAlt}`}>
      <div className={styles.narrow}>
        <h2 className={styles.h2}>Common questions</h2>
        <ul className={styles.popular}>
          {items.map((i) => (
            <li key={i.to}>
              <Link to={i.to}>{i.label}</Link>
            </li>
          ))}
        </ul>
      </div>
    </section>
  );
}

export default function Home(): ReactNode {
  return (
    <Layout
      title="Documentation"
      description="How AddaxAI works, what each screen does, and where the numbers come from."
    >
      <Hero />
      <main>
        <Sections />
        <Popular />
      </main>
    </Layout>
  );
}

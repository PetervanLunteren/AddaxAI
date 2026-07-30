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
  /** Spans both columns, so an odd last card does not leave a gap. */
  wide?: boolean;
}

const SECTIONS: Card[] = [
  {
    title: "Start here",
    body: "New to AddaxAI. Install it, run it once, and pick the path that fits your workflow.",
    to: "/docs/start-here/what-is-addaxai",
    links: [
      { label: "What AddaxAI does", to: "/docs/start-here/what-is-addaxai" },
      { label: "Install", to: "/docs/start-here/install" },
      { label: "Choose a workflow", to: "/docs/start-here/choose-a-workflow" },
    ],
  },
  {
    title: "Guides",
    body: "Step by step: run an analysis, check labels and counts. In a project, results add up into charts and maps.",
    to: "/docs/guides/analyse-a-folder",
    links: [
      { label: "Analyse a folder", to: "/docs/guides/analyse-a-folder" },
      { label: "Build a project", to: "/docs/guides/build-a-project" },
      { label: "Check the labels", to: "/docs/guides/check-labels" },
      { label: "Confirm the counts", to: "/docs/guides/confirm-counts" },
      { label: "Use results in Timelapse", to: "/docs/guides/timelapse" },
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
  {
    title: "Help",
    body: "Answers to the questions people ask most, and what to do when something goes wrong.",
    to: "/docs/troubleshooting/faq",
    wide: true,
    links: [
      { label: "FAQ", to: "/docs/troubleshooting/faq" },
      { label: "Troubleshooting", to: "/docs/troubleshooting/" },
      { label: "Go back to version 6", to: "/docs/troubleshooting/go-back-to-v6" },
    ],
  },
];

function Hero(): ReactNode {
  const logo = useBaseUrl("/img/logo-wordmark.png");
  const bg = useBaseUrl("/img/home-background.webp");
  return (
    <header
      className={styles.hero}
      style={{
        backgroundImage: `linear-gradient(180deg, rgba(10,40,42,0.74), rgba(10,40,42,0.84)), url(${bg})`,
      }}
    >
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
    <section className={`${styles.section} ${styles.sectionAlt}`}>
      <div className={styles.wide}>
        <div className={styles.cards}>
          {SECTIONS.map((c) => (
            <div
              key={c.title}
              className={c.wide ? `${styles.card} ${styles.cardWide}` : styles.card}
            >
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

export default function Home(): ReactNode {
  return (
    <Layout
      title="Documentation"
      description="How AddaxAI works, what each screen does, and where the numbers come from."
    >
      <Hero />
      <main>
        <Sections />
      </main>
    </Layout>
  );
}

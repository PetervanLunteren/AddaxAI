import type { ReactNode } from "react";
import Layout from "@theme/Layout";
import Link from "@docusaurus/Link";
import useBaseUrl from "@docusaurus/useBaseUrl";
import pagesData from "@site/src/data/pages.json";
import styles from "./index.module.css";

// Documentation home. This site explains how AddaxAI works and what the
// numbers mean. It is not a sales page: the job here is to get a reader to
// the right page in one click.

interface Section {
  label: string;
  description: string;
  to: string;
  pages: Array<{ label: string; to: string }>;
}

// Written by scripts/sync-pages.mjs before every start and build, from the
// same folders and frontmatter the sidebar is generated from. Never edit
// the list here: add or reorder pages under docs/ and both update.
const SECTIONS = pagesData as Section[];

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
          {SECTIONS.map((c, i) => (
            <div
              key={c.label}
              // An odd last card spans both columns, so it leaves no gap.
              className={
                i === SECTIONS.length - 1 && SECTIONS.length % 2 === 1
                  ? `${styles.card} ${styles.cardWide}`
                  : styles.card
              }
            >
              <h2 className={styles.cardTitle}>
                <Link to={c.to}>{c.label}</Link>
              </h2>
              <p className={styles.cardBody}>{c.description}</p>
              <ul className={styles.cardLinks}>
                {c.pages.map((l) => (
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

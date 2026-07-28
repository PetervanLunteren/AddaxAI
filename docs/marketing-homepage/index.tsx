import type { ReactNode } from "react";
import Layout from "@theme/Layout";
import Link from "@docusaurus/Link";
import useBaseUrl from "@docusaurus/useBaseUrl";
import AppShot from "@site/src/components/AppShot";
import styles from "./index.module.css";

// Marketing homepage. Docs live under /docs; this is the front door for
// curious, non-technical visitors: what it does, proof it runs on their own
// computer, the download, and a look at the real interface.

const RELEASE = "https://github.com/PetervanLunteren/AddaxAI/releases/latest";
const DOWNLOADS = {
  windows: `${RELEASE}/download/AddaxAI-Setup.exe`,
  macos: `${RELEASE}/download/AddaxAI-arm64.dmg`,
  linux: `${RELEASE}/download/AddaxAI-amd64.deb`,
};

function DownloadButtons({ compact = false }: { compact?: boolean }): ReactNode {
  return (
    <div className={compact ? styles.dlRowCompact : styles.dlRow}>
      <a className={styles.dlBtn} href={DOWNLOADS.windows}>
        Download for Windows
      </a>
      <a className={styles.dlBtn} href={DOWNLOADS.macos}>
        Download for macOS
      </a>
      <a className={styles.dlBtn} href={DOWNLOADS.linux}>
        Download for Linux
      </a>
    </div>
  );
}

function Hero(): ReactNode {
  const bg = useBaseUrl("/img/home-background.webp");
  const logo = useBaseUrl("/img/logo-wordmark.png");
  return (
    <header
      className={styles.hero}
      style={{
        backgroundImage: `linear-gradient(180deg, rgba(10,40,42,0.72), rgba(10,40,42,0.82)), url(${bg})`,
      }}
    >
      <div className={styles.heroInner}>
        <img className={styles.heroLogo} src={logo} alt="AddaxAI" />
        <h1 className={styles.heroTitle}>
          Turn camera-trap photos into wildlife data
        </h1>
        <p className={styles.heroSub}>
          AddaxAI finds and identifies the animals for you. It is free, open
          source, and runs on your own computer. No code, and your photos never
          leave your machine.
        </p>
        <DownloadButtons />
        <p className={styles.heroTrust}>
          18,365 downloads · 135 countries · 166 universities
        </p>
        <a className={styles.heroScroll} href="#how">
          See how it works ↓
        </a>
      </div>
    </header>
  );
}

function Problem(): ReactNode {
  return (
    <section className={styles.section}>
      <div className={styles.narrow}>
        <h2 className={styles.h2}>Most camera-trap photos are empty</h2>
        <p className={styles.lead}>
          Camera traps make thousands of photos. Most show nothing, and sorting
          them by hand takes days. AddaxAI runs the AI for you. It finds the
          animals, people, and vehicles, then names the species. You spend your
          time on the wildlife, not the sorting.
        </p>
      </div>
    </section>
  );
}

const STEPS = [
  {
    title: "Import",
    body: "Pick a folder on your own computer. Nothing is uploaded.",
  },
  {
    title: "Analyse",
    body: "Choose the models. The AI finds the animals and names the species.",
  },
  {
    title: "Verify",
    body: "Check and correct the labels. Your edits always win over the AI.",
  },
  {
    title: "Post-process",
    body: "Get tables and species folders, or build a project with dashboards and maps.",
  },
];

function HowItWorks(): ReactNode {
  return (
    <section className={`${styles.section} ${styles.sectionAlt}`} id="how">
      <div className={styles.wide}>
        <h2 className={styles.h2}>How it works</h2>
        <ol className={styles.steps}>
          {STEPS.map((s, i) => (
            <li key={s.title} className={styles.step}>
              <span className={styles.stepNum}>{i + 1}</span>
              <h3 className={styles.stepTitle}>{s.title}</h3>
              <p className={styles.stepBody}>{s.body}</p>
            </li>
          ))}
        </ol>
      </div>
    </section>
  );
}

const SHOWCASE = [
  {
    shot: "dashboard",
    title: "See your data come alive",
    body: "Counts per species, activity through the day, and trends over the season. Filter by site, date, or species. A map shows every camera and how often animals turned up.",
  },
  {
    shot: "verify",
    title: "Check the AI, fast",
    body: "Go through detections quickly. Similarity search groups animals that look alike, so you can confirm or relabel a whole group at once.",
  },
  {
    shot: "counts",
    title: "Get your data out",
    body: "Export to Camtrap DP, a recognition file for Timelapse, species-separated folders, or images with the animals marked.",
  },
];

function Showcase(): ReactNode {
  return (
    <section className={styles.section}>
      <div className={styles.wide}>
        {SHOWCASE.map((row, i) => (
          <div
            key={row.shot}
            className={`${styles.showRow} ${i % 2 ? styles.showRowRev : ""}`}
          >
            <div className={styles.showText}>
              <h2 className={styles.h2}>{row.title}</h2>
              <p className={styles.lead}>{row.body}</p>
            </div>
            <div className={styles.showShot}>
              <AppShot name={row.shot} />
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}

function TwoModes(): ReactNode {
  return (
    <section className={`${styles.section} ${styles.sectionAlt}`}>
      <div className={styles.wide}>
        <h2 className={styles.h2}>Two ways to work</h2>
        <div className={styles.modes}>
          <div className={styles.modeCard}>
            <h3 className={styles.modeTitle}>Analyse a folder</h3>
            <p className={styles.modeBody}>
              A quick one-off run. Point at a folder, get results, move on. Good
              for a single batch of photos.
            </p>
            <Link className={styles.modeLink} to="/docs/analyse-a-folder/overview">
              Learn more
            </Link>
          </div>
          <div className={styles.modeCard}>
            <h3 className={styles.modeTitle}>Build a project</h3>
            <p className={styles.modeBody}>
              Track many cameras over time. Confirm species counts, keep a
              verification history, and watch dashboards and maps update as you
              add more.
            </p>
            <Link className={styles.modeLink} to="/docs/projects/overview">
              Learn more
            </Link>
          </div>
        </div>
      </div>
    </section>
  );
}

const STATS = [
  ["18,365", "downloads"],
  ["135", "countries"],
  ["166", "universities"],
  ["438", "affiliations"],
];

function Stats(): ReactNode {
  return (
    <section className={styles.statsBand}>
      <div className={styles.statsRow}>
        {STATS.map(([n, label]) => (
          <div key={label} className={styles.stat}>
            <div className={styles.statNum}>{n}</div>
            <div className={styles.statLabel}>{label}</div>
          </div>
        ))}
      </div>
    </section>
  );
}

const FAQ = [
  {
    q: "Does my data get uploaded anywhere?",
    a: "No. Everything runs on your own computer. Your photos never leave your machine.",
  },
  {
    q: "Is it free?",
    a: "Yes. AddaxAI is free and open source.",
  },
  {
    q: "Do I need a GPU?",
    a: "No. It runs faster with an NVIDIA GPU or Apple Silicon, but it works on any computer, just slower.",
  },
  {
    q: "Does it work on video?",
    a: "Yes. It handles both photos and videos.",
  },
  {
    q: "Can I correct the AI?",
    a: "Yes. You can check and relabel anything. Your edits always take priority over the AI.",
  },
  {
    q: "Which species can it recognise?",
    a: "Many, through a growing set of models for different regions and animal groups. See the model zoo in the docs.",
  },
  {
    q: "Which systems does it run on?",
    a: "Windows, macOS (Apple Silicon), and Linux.",
  },
];

function Faq(): ReactNode {
  return (
    <section className={styles.section}>
      <div className={styles.narrow}>
        <h2 className={styles.h2}>Common questions</h2>
        <div className={styles.faq}>
          {FAQ.map((item) => (
            <details key={item.q} className={styles.faqItem}>
              <summary className={styles.faqQ}>{item.q}</summary>
              <p className={styles.faqA}>{item.a}</p>
            </details>
          ))}
        </div>
        <p className={styles.faqMore}>
          <Link to="/docs/faq">More questions</Link>
        </p>
      </div>
    </section>
  );
}

function FinalCta(): ReactNode {
  return (
    <section className={`${styles.section} ${styles.sectionAlt} ${styles.finalCta}`}>
      <div className={styles.narrow}>
        <h2 className={styles.h2}>Ready to try it?</h2>
        <p className={styles.lead}>
          Free, open source, and runs on your own computer.
        </p>
        <DownloadButtons compact />
        <p className={styles.finalDocs}>
          <Link to="/docs/getting-started/installation">Read the docs</Link>
        </p>
      </div>
    </section>
  );
}

export default function Home(): ReactNode {
  return (
    <Layout
      title="Camera trap wildlife analysis"
      description="AddaxAI is free software that finds and identifies wildlife in camera-trap photos and videos, on your own computer."
    >
      <Hero />
      <main>
        <Problem />
        <HowItWorks />
        <Showcase />
        <TwoModes />
        <Stats />
        <Faq />
        <FinalCta />
      </main>
    </Layout>
  );
}

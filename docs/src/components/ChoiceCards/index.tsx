import type { ReactNode } from "react";
import styles from "./styles.module.css";

// Side-by-side option cards for a two-way (or N-way) decision. Used on
// "choose a workflow"; reusable anywhere two paths need comparing at a glance.

interface Choice {
  title: string;
  tagline: string;
  bestFor: string;
  gives: string;
  note?: string;
}

export default function ChoiceCards({ choices }: { choices: Choice[] }): ReactNode {
  return (
    <div className={styles.row} style={{ ["--n" as string]: choices.length }}>
      {choices.map((c) => (
        <div key={c.title} className={styles.card}>
          <h3 className={styles.title}>{c.title}</h3>
          <p className={styles.tagline}>{c.tagline}</p>
          <dl className={styles.dl}>
            <dt className={styles.dt}>Best for</dt>
            <dd className={styles.dd}>{c.bestFor}</dd>
            <dt className={styles.dt}>You get</dt>
            <dd className={styles.dd}>{c.gives}</dd>
          </dl>
          {c.note ? <p className={styles.note}>{c.note}</p> : null}
        </div>
      ))}
    </div>
  );
}

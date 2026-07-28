import type { ReactNode } from "react";
import styles from "./styles.module.css";

// A small numbered step row: icon, verb, one line. Used on the "what it does"
// page for Find / Name / Check, and reusable anywhere a short process needs to
// read at a glance instead of as a list.

const FindIcon = (
  <svg viewBox="0 0 24 24" aria-hidden="true">
    <rect x="3.5" y="4.5" width="17" height="13" rx="2" />
    <rect x="7" y="8" width="7" height="6" rx="1" className="accent" />
    <circle cx="18" cy="19.5" r="0.9" fill="currentColor" stroke="none" />
  </svg>
);

const NameIcon = (
  <svg viewBox="0 0 24 24" aria-hidden="true">
    <path d="M4 7.5 h11 M4 12 h16 M4 16.5 h8" />
    <circle cx="18.5" cy="7.5" r="2" className="accent" />
  </svg>
);

const CheckIcon = (
  <svg viewBox="0 0 24 24" aria-hidden="true">
    <path d="M20 6.5 L10 16.5 L5 11.5" />
  </svg>
);

interface Step {
  icon: ReactNode;
  verb: string;
  body: string;
}

export default function Steps({ steps }: { steps: Step[] }): ReactNode {
  return (
    <ol className={styles.row}>
      {steps.map((s, i) => (
        <li key={s.verb} className={styles.step}>
          <span className={styles.head}>
            <span className={styles.icon}>{s.icon}</span>
            <span className={styles.num}>{i + 1}</span>
          </span>
          <span className={styles.verb}>{s.verb}</span>
          <span className={styles.body}>{s.body}</span>
        </li>
      ))}
    </ol>
  );
}

// Preset icons exported so pages can pass them without redefining SVGs.
export const stepIcons = { find: FindIcon, name: NameIcon, check: CheckIcon };

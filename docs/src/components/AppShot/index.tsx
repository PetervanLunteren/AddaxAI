import type { ReactNode } from "react";
import useBaseUrl from "@docusaurus/useBaseUrl";
import { shots, isPlaceholder } from "@site/src/data/screenshots";
import styles from "./styles.module.css";

// Shared screenshot frame, driven by the src/data/screenshots manifest.
// Used by the homepage and the docs so a real image (or the placeholder)
// shows in exactly one place. useBaseUrl passes http URLs through unchanged
// and prepends the site baseUrl to local /static paths.

export default function AppShot({ name }: { name: string }): ReactNode {
  const shot = shots[name];
  const src = useBaseUrl(shot?.src ?? "/img/screenshot-placeholder.svg");
  return (
    <figure className={styles.shot}>
      <img className={styles.img} src={src} alt={shot?.alt ?? ""} loading="lazy" />
      {isPlaceholder(name) ? (
        <figcaption className={styles.note}>screenshot to be added</figcaption>
      ) : null}
    </figure>
  );
}

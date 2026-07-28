import type { ReactNode } from "react";
import { useBaseUrlUtils } from "@docusaurus/useBaseUrl";
import { shots } from "@site/src/data/screenshots";
import styles from "./styles.module.css";

// Responsive grid of app screenshots, driven by the src/data/screenshots
// manifest. Use to show several views at a glance, e.g. dashboard + insights.
// One withBaseUrl call (hook) is reused across the tiles.

export default function AppGallery({ names }: { names: string[] }): ReactNode {
  const { withBaseUrl } = useBaseUrlUtils();
  return (
    <div className={styles.grid}>
      {names.map((name) => {
        const shot = shots[name];
        const src = withBaseUrl(shot?.src ?? "/img/screenshot-placeholder.svg");
        return (
          <figure key={name} className={styles.tile}>
            <img className={styles.img} src={src} alt={shot?.alt ?? ""} loading="lazy" />
          </figure>
        );
      })}
    </div>
  );
}

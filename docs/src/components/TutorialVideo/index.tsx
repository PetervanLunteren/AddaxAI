import type { ReactNode } from "react";
import styles from "./styles.module.css";

// YouTube embed for the tutorial videos on the guide pages. Same frame
// styling as AppShot so screenshots and videos read as one family.
// youtube-nocookie.com keeps YouTube from setting cookies until the
// visitor presses play; loading="lazy" defers the iframe until it
// scrolls into view.

export default function TutorialVideo({
  id,
  title,
}: {
  // The YouTube video id, the part after `watch?v=` in the URL.
  id: string;
  // Accessible name for the iframe, e.g. "Video: analyse a folder".
  title: string;
}): ReactNode {
  return (
    <div className={styles.frame}>
      <iframe
        className={styles.player}
        src={`https://www.youtube-nocookie.com/embed/${id}`}
        title={title}
        loading="lazy"
        allow="accelerometer; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
        allowFullScreen
      />
    </div>
  );
}

import { useEffect, useRef, useState, type ReactNode } from "react";
import styles from "./styles.module.css";

// Collapsible FAQ rows that stay linkable.
//
// A plain <details> would hide the answer from a deep link: the browser
// scrolls to a closed element and the reader sees nothing. So each item
// owns its id, opens itself when the URL points at it, and writes its id
// back into the URL when opened, which is how a reader gets a link to
// copy. Section headings stay ordinary markdown, so the page's table of
// contents still lists them.

// Set while "Expand all" is flipping every row. Each row still gets a
// toggle event, and without this every one of them would claim the URL,
// leaving the reader on a link to whichever row happened to be last.
let bulkToggling = false;

interface FaqItemProps {
  /** URL fragment for this question. Keep it stable: links depend on it. */
  id: string;
  /** The question, shown on the closed row. */
  q: string;
  /**
   * Optional line under the question while the row is closed, hidden
   * once it opens. For lists of problems, put the error the reader is
   * looking at here: that, not the title, is what they match against.
   * Leave it out where the question already says everything.
   */
  summary?: string;
  children: ReactNode;
}

export function FaqItem({ id, q, summary, children }: FaqItemProps): ReactNode {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDetailsElement>(null);

  // Open when the page loads at this item, and when the hash changes
  // later (in-page links, browser back). Scrolling is left to the
  // browser on load; on a later hash change the element may already be
  // in view, so only nudge it when we opened it ourselves.
  useEffect(() => {
    const target = () => decodeURIComponent(window.location.hash.slice(1)) === id;
    if (target()) setOpen(true);

    const onHashChange = () => {
      if (!target()) return;
      setOpen(true);
      ref.current?.scrollIntoView({ block: "start", behavior: "smooth" });
    };
    window.addEventListener("hashchange", onHashChange);
    return () => window.removeEventListener("hashchange", onHashChange);
  }, [id]);

  // Opening a question puts it in the address bar, so the reader can copy
  // the link without hunting for an anchor. replaceState keeps the back
  // button meaning "the previous page", not "the previous question".
  function onToggle(e: React.SyntheticEvent<HTMLDetailsElement>): void {
    const isOpen = e.currentTarget.open;
    setOpen(isOpen);
    if (isOpen && !bulkToggling && typeof window !== "undefined") {
      window.history.replaceState(null, "", `#${id}`);
    }
  }

  return (
    <details ref={ref} id={id} className={styles.item} open={open} onToggle={onToggle}>
      <summary className={styles.summary}>
        <span className={open ? styles.caretOpen : styles.caret} aria-hidden="true">
          ›
        </span>
        <span className={styles.text}>
          <span className={styles.question}>{q}</span>
          {summary && <span className={styles.teaser}>{summary}</span>}
        </span>
        <a
          className={styles.hash}
          href={`#${id}`}
          aria-label={`Direct link to: ${q}`}
          title="Direct link to this question"
          onClick={(e) => e.stopPropagation()}
        >
          #
        </a>
      </summary>
      <div className={styles.answer}>{children}</div>
    </details>
  );
}

// Find-in-page cannot see inside a closed <details>, so give readers one
// click to lay the whole page open before they search it.
export function FaqExpandAll(): ReactNode {
  const [expanded, setExpanded] = useState(false);

  function toggle(): void {
    const next = !expanded;
    bulkToggling = true;
    document
      .querySelectorAll<HTMLDetailsElement>(`details.${styles.item}`)
      .forEach((d) => {
        d.open = next;
      });
    // Toggle events fire asynchronously, so release the guard only after
    // this task, once every row has reported in.
    setTimeout(() => {
      bulkToggling = false;
    }, 0);
    setExpanded(next);
  }

  return (
    <button type="button" className={styles.expandAll} onClick={toggle}>
      {expanded ? "Collapse all" : "Expand all"}
    </button>
  );
}

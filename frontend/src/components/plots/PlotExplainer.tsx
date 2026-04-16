/**
 * Reusable "About this plot" section for pages under /plots.
 *
 * Every plot page drops this in at the bottom, below the chart and
 * footer, and passes its own content for each of the five fixed
 * sub-sections. The fixed structure keeps the UX predictable across
 * plots: once users learn where to find "Caveats" on one plot, they
 * know where to look on the next.
 *
 * Collapse state is persisted in localStorage (not sessionStorage)
 * because it's a learned UX preference — "I've read this, hide it
 * next time" — not per-session intent like the auto-pick species.
 * Default state is collapsed; users opt in to read the science.
 */

import { useState } from "react";
import { ChevronRight, ExternalLink } from "lucide-react";

import { cn } from "../../lib/utils";

export interface PlotReference {
  /** Full citation text, author-year-title-journal style. */
  citation: string;
  /** Optional DOI or journal URL for a "→" link icon. */
  url?: string;
}

export interface PlotSettingInfluence {
  /** Short name of the project setting, e.g. "Timezone". */
  label: string;
  /** One-line explanation of how it affects this plot. */
  detail: string;
}

interface PlotExplainerProps {
  /** Stable identifier used for the localStorage persistence key. */
  plotKey: string;
  /** "What does the user see on the plot?" Plain language, no jargon. */
  what: React.ReactNode;
  /** "How is it computed?" Method summary + key parameters. */
  how: React.ReactNode;
  /** Free-form caveats, sample-size warnings, pitfalls. Rendered as-is. */
  caveats?: React.ReactNode;
  /** Project settings that flow into this plot's output. */
  settings?: PlotSettingInfluence[];
  /** Scientific references backing the plot's methods and defaults. */
  references?: PlotReference[];
}

const STORAGE_PREFIX = "addaxai:plots:";

function readStoredExpanded(plotKey: string): boolean {
  try {
    const saved = localStorage.getItem(`${STORAGE_PREFIX}${plotKey}:explainer-expanded`);
    if (saved === null) return false;
    return saved === "true";
  } catch {
    return false;
  }
}

function writeStoredExpanded(plotKey: string, expanded: boolean): void {
  try {
    localStorage.setItem(
      `${STORAGE_PREFIX}${plotKey}:explainer-expanded`,
      String(expanded),
    );
  } catch {
    /* ignore */
  }
}

interface SectionProps {
  title: string;
  children: React.ReactNode;
}

function Section({ title, children }: SectionProps) {
  return (
    <div className="space-y-2">
      <h3 className="text-sm font-semibold text-foreground">{title}</h3>
      <div className="space-y-2 text-sm text-muted-foreground">{children}</div>
    </div>
  );
}

export function PlotExplainer({
  plotKey,
  what,
  how,
  caveats,
  settings,
  references,
}: PlotExplainerProps) {
  const [expanded, setExpanded] = useState<boolean>(() => readStoredExpanded(plotKey));

  const toggle = () => {
    const next = !expanded;
    setExpanded(next);
    writeStoredExpanded(plotKey, next);
  };

  return (
    <section className="rounded-lg border bg-card">
      <button
        type="button"
        onClick={toggle}
        aria-expanded={expanded}
        className="flex w-full items-center gap-2 rounded-lg px-4 py-3 text-left text-sm font-medium hover:bg-accent"
      >
        <ChevronRight
          className={cn(
            "h-4 w-4 shrink-0 opacity-60 transition-transform",
            expanded && "rotate-90",
          )}
        />
        <span>About this plot</span>
      </button>

      {expanded && (
        <div className="space-y-6 border-t px-4 py-4">
          <Section title="What it shows">{what}</Section>
          <Section title="How it's computed">{how}</Section>

          {caveats && <Section title="Caveats">{caveats}</Section>}

          {settings && settings.length > 0 && (
            <Section title="Project settings that affect this plot">
              <ul className="list-disc space-y-1 pl-5">
                {settings.map((setting) => (
                  <li key={setting.label}>
                    <span className="font-medium text-foreground">
                      {setting.label}
                    </span>
                    {" — "}
                    {setting.detail}
                  </li>
                ))}
              </ul>
            </Section>
          )}

          {references && references.length > 0 && (
            <Section title="References">
              <ol className="list-decimal space-y-2 pl-5">
                {references.map((ref) => (
                  <li key={ref.citation} className="leading-snug">
                    {ref.citation}
                    {ref.url && (
                      <>
                        {" "}
                        <a
                          href={ref.url}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="inline-flex items-center gap-0.5 text-primary hover:underline"
                        >
                          <ExternalLink className="h-3 w-3" />
                        </a>
                      </>
                    )}
                  </li>
                ))}
              </ol>
            </Section>
          )}
        </div>
      )}
    </section>
  );
}

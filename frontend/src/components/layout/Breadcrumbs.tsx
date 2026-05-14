/**
 * Breadcrumb strip rendered above page content.
 *
 * Reads the current route via useBreadcrumbs() and renders a slim
 * navigation strip. Returns null when the registry has no entry for
 * the route (home, setup), so callers can drop this in unconditionally.
 *
 * Placement: each layout (AppLayout, single-purpose pages) renders
 * one Breadcrumbs at the top of its main content area. Inside a page
 * that already uses the canonical FRONTEND_CONVENTIONS header
 * pattern, place it before the `<header>` so the title bar still
 * sits on its own line.
 */

import { Fragment } from "react";
import { ChevronRight } from "lucide-react";
import { Link } from "react-router-dom";
import { useBreadcrumbs } from "../../lib/breadcrumbs";

export function Breadcrumbs() {
  const items = useBreadcrumbs();
  if (items.length === 0) return null;

  return (
    <nav
      aria-label="Breadcrumb"
      className="border-b bg-white/60 backdrop-blur-sm"
    >
      <ol className="mx-auto flex max-w-7xl items-center gap-1.5 px-4 py-2 text-xs sm:px-6 lg:px-8">
        {items.map((item, index) => {
          const isLast = index === items.length - 1;
          return (
            <Fragment key={`${item.label}-${index}`}>
              {index > 0 && (
                <ChevronRight className="h-3 w-3 shrink-0 text-muted-foreground/60" />
              )}
              {item.to && !isLast ? (
                <Link
                  to={item.to}
                  className="text-muted-foreground hover:text-foreground"
                >
                  {item.label}
                </Link>
              ) : (
                <span className="font-medium text-foreground">
                  {item.label}
                </span>
              )}
            </Fragment>
          );
        })}
      </ol>
    </nav>
  );
}

/**
 * Keeps one page's render error from blanking the whole window.
 *
 * React unmounts everything up to the nearest error boundary when a
 * component throws, and until 2026-11 there was none, so one broken
 * dashboard card left a white window with no sidebar and no way to the
 * page that could fix the data (beta report, 2026-10-31). Used around the
 * project pages inside AppLayout, so the sidebar survives, and once around
 * the top-level routes. Keyed on the path, so navigating to another page
 * starts clean; there is no retry button because the sidebar is the retry.
 *
 * Logging lives in main.tsx (`onCaughtError`), not here: React 19 hands a
 * caught error to that root option and never to window.onerror.
 */

import { Component, type ReactNode } from "react";
import { useLocation } from "react-router-dom";

import { Callout } from "../ui/callout";

interface BoundaryState {
  error: unknown;
}

class Boundary extends Component<{ children: ReactNode }, BoundaryState> {
  state: BoundaryState = { error: null };

  static getDerivedStateFromError(error: unknown): BoundaryState {
    return { error };
  }

  render() {
    if (this.state.error === null) return this.props.children;
    return (
      <div className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8">
        <Callout variant="error" title="This page could not be shown">
          <p>
            Something went wrong while drawing this page. Open another page
            from the sidebar, or restart AddaxAI. If it keeps happening, send
            a diagnostic report from the Troubleshooting menu.
          </p>
          <p className="mt-2 font-mono text-xs">{String(this.state.error)}</p>
        </Callout>
      </div>
    );
  }
}

export function RouteErrorBoundary({ children }: { children: ReactNode }) {
  const { pathname } = useLocation();
  return <Boundary key={pathname}>{children}</Boundary>;
}

/**
 * Keeps one page's render error from blanking the whole window.
 *
 * React unmounts everything up to the nearest error boundary when a
 * component throws, and until 2026-11 there was none, so one broken
 * dashboard card left a white window with no sidebar and no way to the
 * page that could fix the data (beta report, 2026-10-31). Used around the
 * project pages inside AppLayout, so the sidebar survives, and once around
 * the top-level routes.
 *
 * Going to another page clears the error. That is done by comparing the
 * path, not by keying the boundary on it: a key would remount everything
 * below on every navigation, the layouts too, which reopened a dismissed
 * dialog on each page change.
 *
 * Logging lives in main.tsx (`onCaughtError`), not here: React 19 hands a
 * caught error to that root option and never to window.onerror.
 */

import { Component, type ReactNode } from "react";
import { Link, useLocation } from "react-router-dom";

import { Callout } from "../ui/callout";

interface BoundaryProps {
  pathname: string;
  children: ReactNode;
}

interface BoundaryState {
  error: unknown;
}

class Boundary extends Component<BoundaryProps, BoundaryState> {
  state: BoundaryState = { error: null };

  static getDerivedStateFromError(error: unknown): BoundaryState {
    return { error };
  }

  componentDidUpdate(prevProps: BoundaryProps, prevState: BoundaryState) {
    // Only an error that was already on screen before the path changed.
    // The update that catches an error on arrival also has a new path,
    // and clearing that one would render the broken page a second time.
    if (prevState.error !== null && prevProps.pathname !== this.props.pathname) {
      this.setState({ error: null });
    }
  }

  render() {
    if (this.state.error === null) return this.props.children;
    return (
      <div className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8">
        <Callout variant="error" title="This page could not be shown">
          <p>
            Something went wrong while drawing this page. Open another page,
            or restart AddaxAI. If it keeps happening, send a diagnostic report
            from the Help menu.
          </p>
          <p className="mt-2 font-mono text-xs">{String(this.state.error)}</p>
          <Link to="/" className="mt-3 inline-block font-medium underline">
            Go to home
          </Link>
        </Callout>
      </div>
    );
  }
}

export function RouteErrorBoundary({ children }: { children: ReactNode }) {
  const { pathname } = useLocation();
  return <Boundary pathname={pathname}>{children}</Boundary>;
}

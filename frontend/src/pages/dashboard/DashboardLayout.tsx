/**
 * Dashboard shell: the page header and the Overview / Explore tab strip.
 *
 * The tabs are real routes, so a link, the back button and "open in a new
 * window" all land on the right one. The query string travels with a tab
 * switch, which keeps Explore's filters in place when you look at the
 * Overview and come back. Overview never reads it: it always shows the
 * whole project, so it cannot end up filtered by a control it does not
 * show.
 */

import { NavLink, Outlet, useLocation, useParams } from "react-router-dom";

import { cn } from "../../lib/utils";

// One line per tab, because the two tabs answer different questions.
const CAPTIONS = {
  overview: "The whole project at a glance",
  explore: "Any labels at the sites and dates you choose",
} as const;

const tabClass = ({ isActive }: { isActive: boolean }) =>
  cn(
    "-mb-px border-b-2 px-4 py-2 text-sm font-medium transition-colors",
    isActive
      ? "border-primary text-foreground"
      : "border-transparent text-muted-foreground hover:text-foreground",
  );

export default function DashboardLayout() {
  const { projectId } = useParams<{ projectId: string }>();
  const { pathname, search } = useLocation();
  if (!projectId) return null;
  const base = `/projects/${projectId}/dashboard`;
  const onExplore = pathname.endsWith("/explore");

  return (
    <div className="min-h-screen">
      <header className="border-b bg-card/80 backdrop-blur-sm">
        <div className="mx-auto max-w-7xl px-4 py-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold tracking-tight">Dashboard</h1>
              <p className="text-sm text-muted-foreground">
                {onExplore ? CAPTIONS.explore : CAPTIONS.overview}
              </p>
            </div>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-7xl space-y-6 px-4 py-8 sm:px-6 lg:px-8">
        <nav className="flex border-b" aria-label="Dashboard views">
          {/* `end` keeps Overview from matching the Explore route too. */}
          <NavLink end to={{ pathname: base, search }} className={tabClass}>
            Overview
          </NavLink>
          <NavLink to={{ pathname: `${base}/explore`, search }} className={tabClass}>
            Explore
          </NavLink>
        </nav>
        <Outlet />
      </main>
    </div>
  );
}

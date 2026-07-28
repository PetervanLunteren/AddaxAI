/**
 * Project Dashboard page — thin wrapper around ``DashboardView``.
 *
 * The body of the dashboard lives in ``DashboardView``; this page
 * just provides the research-projects chrome (``<header>`` with the
 * page title) and the outer max-width container.
 */

import { useParams } from "react-router-dom";

import { DashboardView } from "../components/dashboard/DashboardView";

export default function DashboardPage() {
  const { projectId } = useParams<{ projectId: string }>();
  if (!projectId) return null;

  return (
    <div className="min-h-screen">
      <header className="border-b bg-white/80 backdrop-blur-sm">
        <div className="mx-auto max-w-7xl px-4 py-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold tracking-tight">
                Dashboard
              </h1>
              <p className="text-sm text-muted-foreground">
                Project overview with statistics and trends
              </p>
            </div>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8">
        <DashboardView projectId={projectId} />
      </main>
    </div>
  );
}

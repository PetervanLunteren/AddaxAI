/**
 * Home screen.
 *
 * Task-based chooser per WORKFLOW_RECOMMENDATION.md: pick what you
 * want AddaxAI to do, not which mode you are in. Two equal-weight
 * paths:
 *
 * 1. Analyse a folder: a quick one-off run, results out, no setup.
 * 2. Projects: the stored, revisitable project workspace.
 *
 * The two cards differentiate on one-off vs persistent (not on
 * feature count): a folder run hands you output files and steps back,
 * a project keeps everything in the app to return to and add to.
 *
 * Both cards share the same chrome (padding, heading size, icon size,
 * single primary button) so the user is not nudged toward one over
 * the other by layout alone. New-project creation lives inside the
 * projects page itself, not on this chooser.
 *
 * Timelapse Analyser is no longer a separate mode: its launcher
 * (`AddaxAI.exe --timelapse <folder>`) now opens "Analyse a folder"
 * with the folder pre-filled, and the folder run's recognitions.json
 * is what Timelapse imports.
 */

import { useNavigate } from "react-router-dom";
import { ArrowRight, LayoutDashboard, FolderOpen } from "lucide-react";
import { Button } from "../components/ui/button";
import { Card, CardContent } from "../components/ui/card";
import { AppHamburger } from "../components/layout/AppHamburger";

export function HomePage() {
  const navigate = useNavigate();

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100">
      <header className="relative z-40 border-b bg-white/80 backdrop-blur-sm">
        <div className="mx-auto max-w-7xl px-4 py-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between gap-4">
            <div className="flex items-center gap-3">
              <img
                src="/branding/logo-mark.png"
                alt=""
                className="h-16 w-16 shrink-0"
              />
              <div>
                <h1 className="text-2xl font-bold tracking-tight">
                  What do you want to do?
                </h1>
                <p className="text-sm text-muted-foreground">
                  Pick the path that matches your task
                </p>
              </div>
            </div>
            <AppHamburger />
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8">
        <div className="grid gap-6 lg:grid-cols-2">
          <Card className="flex flex-col">
            <CardContent className="flex flex-1 flex-col gap-6 p-8">
              <div className="flex items-start gap-4">
                <div className="rounded-lg bg-primary/10 p-3">
                  <FolderOpen className="h-8 w-8 text-primary" />
                </div>
                <div className="flex-1">
                  <h2 className="text-xl font-semibold tracking-tight">
                    Analyse a folder
                  </h2>
                  <p className="mt-1 text-sm text-muted-foreground">
                    A quick one-off run. Point at a folder, get results,
                    move on.
                  </p>
                </div>
              </div>

              <p className="text-sm text-muted-foreground">
                Run the AI, review and fix the results if you want, then
                get files out: a results table, a recognition file for
                Timelapse, species-separated folders, visualised or
                blurred images. You manage the output files yourself.
              </p>

              <div className="mt-auto">
                <Button
                  size="lg"
                  onClick={() => navigate("/folder-runs/new")}
                  className="gap-2"
                >
                  Start folder analysis
                  <ArrowRight className="h-4 w-4" />
                </Button>
              </div>
            </CardContent>
          </Card>

          <Card className="flex flex-col">
            <CardContent className="flex flex-1 flex-col gap-6 p-8">
              <div className="flex items-start gap-4">
                <div className="rounded-lg bg-primary/10 p-3">
                  <LayoutDashboard className="h-8 w-8 text-primary" />
                </div>
                <div className="flex-1">
                  <h2 className="text-xl font-semibold tracking-tight">
                    Projects
                  </h2>
                  <p className="mt-1 text-sm text-muted-foreground">
                    A workspace you come back to and keep adding cameras
                    to.
                  </p>
                </div>
              </div>

              <p className="text-sm text-muted-foreground">
                Track many cameras over time: keep verification history,
                watch dashboards and maps, compare activity, and export
                to Camtrap DP. Everything stays stored and revisitable in
                the app.
              </p>

              <div className="mt-auto">
                <Button
                  size="lg"
                  onClick={() => navigate("/projects")}
                  className="gap-2"
                >
                  Open projects
                  <ArrowRight className="h-4 w-4" />
                </Button>
              </div>
            </CardContent>
          </Card>
        </div>
      </main>
    </div>
  );
}

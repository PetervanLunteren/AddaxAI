/**
 * Home screen.
 *
 * Task-based chooser per WORKFLOW_RECOMMENDATION.md: pick what you
 * want AddaxAI to do, not which mode you are in. Two equal-weight
 * paths:
 *
 * 1. Analyse a folder — the legacy point-at-a-folder workflow.
 * 2. Research projects — the existing project workspace.
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
import { ArrowRight, Camera, FolderOpen } from "lucide-react";
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
                    Run AI on one folder and save results you can use
                    right away.
                  </p>
                </div>
              </div>

              <p className="text-sm text-muted-foreground">
                Best for quick camera-trap batches, legacy AddaxAI-style
                workflows, folder separation, visualised images, people
                blurring, and CSV or JSON outputs.
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
                  <Camera className="h-8 w-8 text-primary" />
                </div>
                <div className="flex-1">
                  <h2 className="text-xl font-semibold tracking-tight">
                    Research projects
                  </h2>
                  <p className="mt-1 text-sm text-muted-foreground">
                    Manage sites, deployments, verification,
                    dashboards, insights, and exports.
                  </p>
                </div>
              </div>

              <p className="text-sm text-muted-foreground">
                Best for studies with multiple camera locations,
                repeated imports, metadata, long-term verification,
                maps, activity plots, performance checks, and
                Camtrap-DP style exports.
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

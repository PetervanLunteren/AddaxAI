/**
 * Main App component.
 *
 * Following DEVELOPERS.md principles:
 * - Simple, clear structure
 * - Type hints everywhere
 */

import { QueryClientProvider } from "@tanstack/react-query";
import { queryClient } from "./lib/query-client";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "./components/ui/card";
import { Button } from "./components/ui/button";
import { Camera, FolderTree, BarChart3, Settings } from "lucide-react";

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100">
        {/* Header */}
        <header className="border-b bg-white/80 backdrop-blur-sm">
          <div className="mx-auto max-w-7xl px-4 py-4 sm:px-6 lg:px-8">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="rounded-lg bg-primary p-2">
                  <Camera className="h-6 w-6 text-primary-foreground" />
                </div>
                <div>
                  <h1 className="text-2xl font-bold tracking-tight">
                    AddaxAI
                  </h1>
                  <p className="text-sm text-muted-foreground">
                    Camera Trap Analysis Platform
                  </p>
                </div>
              </div>
              <Button variant="outline" size="sm">
                <Settings className="h-4 w-4" />
                Settings
              </Button>
            </div>
          </div>
        </header>

        {/* Main Content */}
        <main className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8">
          {/* Welcome Section */}
          <div className="mb-8">
            <h2 className="text-3xl font-bold tracking-tight">
              Welcome to AddaxAI
            </h2>
            <p className="mt-2 text-muted-foreground">
              AI-powered wildlife monitoring and analysis for camera trap data
            </p>
          </div>

          {/* Feature Cards */}
          <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
            <Card className="transition-shadow hover:shadow-lg">
              <CardHeader>
                <div className="mb-2 inline-flex rounded-lg bg-primary/10 p-3 text-primary">
                  <FolderTree className="h-6 w-6" />
                </div>
                <CardTitle>Project Management</CardTitle>
                <CardDescription>
                  Organize camera trap deployments and manage your wildlife
                  monitoring projects
                </CardDescription>
              </CardHeader>
              <CardContent>
                <Button className="w-full">Create Project</Button>
              </CardContent>
            </Card>

            <Card className="transition-shadow hover:shadow-lg">
              <CardHeader>
                <div className="mb-2 inline-flex rounded-lg bg-primary/10 p-3 text-primary">
                  <Camera className="h-6 w-6" />
                </div>
                <CardTitle>Image Analysis</CardTitle>
                <CardDescription>
                  AI-powered species detection and classification with
                  confidence scoring
                </CardDescription>
              </CardHeader>
              <CardContent>
                <Button className="w-full" variant="outline">
                  Upload Images
                </Button>
              </CardContent>
            </Card>

            <Card className="transition-shadow hover:shadow-lg">
              <CardHeader>
                <div className="mb-2 inline-flex rounded-lg bg-primary/10 p-3 text-primary">
                  <BarChart3 className="h-6 w-6" />
                </div>
                <CardTitle>Reports & Analytics</CardTitle>
                <CardDescription>
                  Generate insights from your wildlife data with advanced
                  analytics
                </CardDescription>
              </CardHeader>
              <CardContent>
                <Button className="w-full" variant="outline">
                  View Reports
                </Button>
              </CardContent>
            </Card>
          </div>

          {/* System Status */}
          <Card className="mt-8">
            <CardHeader>
              <CardTitle>System Status</CardTitle>
              <CardDescription>
                All systems operational and ready for use
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-2 text-sm">
                <div className="flex items-center gap-2">
                  <div className="h-2 w-2 rounded-full bg-green-500"></div>
                  <span>Backend API: Connected</span>
                  <code className="ml-auto rounded bg-muted px-2 py-0.5 text-xs">
                    http://127.0.0.1:8000
                  </code>
                </div>
                <div className="flex items-center gap-2">
                  <div className="h-2 w-2 rounded-full bg-green-500"></div>
                  <span>React + TypeScript + Vite</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="h-2 w-2 rounded-full bg-green-500"></div>
                  <span>TanStack Query configured</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="h-2 w-2 rounded-full bg-green-500"></div>
                  <span>UI Components ready</span>
                </div>
              </div>
            </CardContent>
          </Card>
        </main>
      </div>
    </QueryClientProvider>
  );
}

export default App;

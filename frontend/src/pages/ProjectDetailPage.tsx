/**
 * Project Detail page showing sites for a project.
 *
 * Following DEVELOPERS.md principles:
 * - Type hints everywhere
 * - Simple, clear structure
 */

import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { useParams, useNavigate } from "react-router-dom";
import { Plus, ArrowLeft, Settings } from "lucide-react";
import { projectsApi } from "../api/projects";
import { sitesApi } from "../api/sites";
import type { SiteResponse } from "../api/types";
import { Button } from "../components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "../components/ui/card";
import { CreateSiteDialog } from "../components/sites/CreateSiteDialog";
import { EditSiteDialog } from "../components/sites/EditSiteDialog";

export function ProjectDetailPage() {
  const { projectId } = useParams<{ projectId: string }>();
  const navigate = useNavigate();
  const [createDialogOpen, setCreateDialogOpen] = useState(false);
  const [editingSite, setEditingSite] = useState<SiteResponse | null>(null);

  const { data: project } = useQuery({
    queryKey: ["projects", projectId],
    queryFn: () => projectsApi.get(projectId!),
    enabled: !!projectId,
  });

  const { data: sites, isLoading } = useQuery({
    queryKey: ["sites", projectId],
    queryFn: () => sitesApi.list(projectId),
    enabled: !!projectId,
  });

  if (!projectId) {
    return <div>Project ID missing</div>;
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100">
      <header className="border-b bg-white/80 backdrop-blur-sm">
        <div className="mx-auto max-w-7xl px-4 py-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <Button
                variant="ghost"
                size="sm"
                onClick={() => navigate("/projects")}
              >
                <ArrowLeft className="h-4 w-4" />
                Back
              </Button>
              <div>
                <h1 className="text-2xl font-bold tracking-tight">
                  {project?.name || "Loading..."}
                </h1>
                <p className="text-sm text-muted-foreground">
                  {project?.description || "Manage sites for this project"}
                </p>
              </div>
            </div>
            <Button onClick={() => setCreateDialogOpen(true)}>
              <Plus className="h-4 w-4" />
              New Site
            </Button>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8">
        {isLoading ? (
          <div className="text-center text-muted-foreground">
            Loading sites...
          </div>
        ) : sites && sites.length > 0 ? (
          <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
            {sites.map((site) => (
              <Card key={site.id} className="transition-shadow hover:shadow-lg">
                <CardHeader>
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <CardTitle>{site.name}</CardTitle>
                      <CardDescription>
                        {site.habitat_type || "No habitat type"}
                      </CardDescription>
                    </div>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => setEditingSite(site)}
                    >
                      <Settings className="h-4 w-4" />
                    </Button>
                  </div>
                </CardHeader>
                <CardContent className="space-y-1 text-sm">
                  {site.latitude && site.longitude && (
                    <p>
                      📍 {site.latitude.toFixed(4)}, {site.longitude.toFixed(4)}
                    </p>
                  )}
                  {site.elevation_m && <p>🏔️ {site.elevation_m}m elevation</p>}
                  {site.notes && (
                    <p className="text-muted-foreground">{site.notes}</p>
                  )}
                  <p className="text-xs text-muted-foreground pt-2">
                    Created: {new Date(site.created_at).toLocaleDateString()}
                  </p>
                </CardContent>
              </Card>
            ))}
          </div>
        ) : (
          <Card>
            <CardContent className="flex flex-col items-center justify-center py-12">
              <p className="mb-4 text-muted-foreground">
                No sites yet. Create your first site to get started.
              </p>
              <Button onClick={() => setCreateDialogOpen(true)}>
                <Plus className="h-4 w-4" />
                Create Site
              </Button>
            </CardContent>
          </Card>
        )}
      </main>

      <CreateSiteDialog
        projectId={projectId}
        open={createDialogOpen}
        onOpenChange={setCreateDialogOpen}
      />

      {editingSite && (
        <EditSiteDialog
          site={editingSite}
          open={!!editingSite}
          onOpenChange={(open) => !open && setEditingSite(null)}
        />
      )}
    </div>
  );
}

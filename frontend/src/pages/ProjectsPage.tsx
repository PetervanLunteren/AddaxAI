/**
 * Projects list page.
 *
 * Following DEVELOPERS.md principles:
 * - Type hints everywhere
 * - Simple, clear structure
 */

import { useEffect, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Link, useNavigate, useSearchParams } from "react-router-dom";
import { Plus, MoreVertical, Pencil, Trash2, ImageIcon } from "lucide-react";
import { projectsApi, type ProjectWithStats } from "../api/projects";
import { modelsApi } from "../api/models";
import { logger } from "../lib/logger";
import { API_BASE_URL } from "../lib/api-client";
import { Button } from "../components/ui/button";
import {
  Card,
  CardContent,
  CardFooter,
  CardHeader,
  CardTitle,
} from "../components/ui/card";
import { formatCompact } from "../lib/utils";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "../components/ui/dropdown-menu";
import { CreateProjectDialog } from "../components/projects/CreateProjectDialog";
import { EditProjectDialog } from "../components/projects/EditProjectDialog";

import { DeleteProjectDialog } from "../components/projects/DeleteProjectDialog";

export function ProjectsPage() {
  const navigate = useNavigate();
  const [searchParams, setSearchParams] = useSearchParams();
  const [createDialogOpen, setCreateDialogOpen] = useState(false);
  const [editingProject, setEditingProject] = useState<ProjectWithStats | null>(null);
  const [deletingProject, setDeletingProject] = useState<ProjectWithStats | null>(null);

  // The "New project…" app-menu item navigates here with ?new=1; open the
  // create dialog and drop the param so a refresh doesn't reopen it.
  useEffect(() => {
    if (searchParams.get("new") === "1") {
      setCreateDialogOpen(true);
      setSearchParams({}, { replace: true });
    }
  }, [searchParams, setSearchParams]);

  const { data: projects, isLoading, error } = useQuery({
    queryKey: ["projects", "research"],
    queryFn: () => projectsApi.getProjects("research"),
  });


  const { data: classificationModels = [] } = useQuery({
    queryKey: ["models", "classification"],
    queryFn: () => modelsApi.listClassificationModels(),
  });

  // Helper to get classification model name by ID
  const getClassificationModelName = (modelId: string | null) => {
    if (!modelId || modelId === "none") return "∅ Detection only";
    const model = classificationModels.find((m) => m.model_id === modelId);
    return model ? `${model.emoji} ${model.friendly_name}` : modelId;
  };

  // Log errors
  if (error) {
    logger.error("Failed to load projects", { error: error.message });
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100">
      {/* Header sits above <main> (relative z-40) so it never paints
          behind the project cards. */}
      <header className="relative z-40 border-b bg-white/80 backdrop-blur-sm">
        <div className="mx-auto max-w-7xl px-4 py-3 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between gap-4">
            <div className="flex items-center gap-3">
              <Link
                to="/"
                aria-label="Home"
                title="Home"
                className="shrink-0 rounded-lg transition-opacity hover:opacity-80"
              >
                <img
                  src="/branding/logo-mark.png"
                  alt="AddaxAI"
                  className="h-12 w-12"
                />
              </Link>
              <div>
                <h1 className="text-2xl font-bold tracking-tight">
                  Projects
                </h1>
                <p className="text-sm text-muted-foreground">
                  Manage wildlife monitoring projects
                </p>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <Button
                onClick={() => {
                  logger.info("User clicked New Project button");
                  setCreateDialogOpen(true);
                }}
              >
                <Plus className="h-4 w-4" />
                New project
              </Button>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8">
        {isLoading ? (
          <div className="text-center text-muted-foreground">
            Loading projects...
          </div>
        ) : projects && projects.length > 0 ? (
          <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
            {projects.map((project: ProjectWithStats) => (
              <Card
                key={project.id}
                className="transition-shadow hover:shadow-lg cursor-pointer"
                onClick={() => {
                  logger.info(`User navigated to project: ${project.name}`, {
                    projectId: project.id,
                  });
                  // Land on the index route; ProjectIndexRoute then
                  // redirects to Dashboard or Analyses based on whether
                  // the project has any files yet.
                  navigate(`/projects/${project.id}`);
                }}
              >
                <div className="relative">
                  {project.thumbnail_path ? (
                    <div className="aspect-video overflow-hidden rounded-t-lg">
                      <img
                        src={`${API_BASE_URL}/api/projects/${project.id}/thumbnail?v=${project.updated_at_utc}`}
                        alt={project.name}
                        className="h-full w-full object-cover"
                        loading="lazy"
                      />
                    </div>
                  ) : (
                    <div className="aspect-video rounded-t-lg bg-gradient-to-br from-slate-100 to-slate-200 flex items-center justify-center">
                      <ImageIcon className="h-10 w-10 text-slate-300" />
                    </div>
                  )}
                  <DropdownMenu>
                    <DropdownMenuTrigger asChild>
                      <Button
                        variant="secondary"
                        size="icon"
                        className="absolute top-2 right-2 h-7 w-7 rounded-full opacity-80 hover:opacity-100"
                        onClick={(e) => e.stopPropagation()}
                      >
                        <MoreVertical className="h-4 w-4" />
                      </Button>
                    </DropdownMenuTrigger>
                    <DropdownMenuContent align="end">
                      <DropdownMenuItem
                        onClick={(e) => {
                          e.stopPropagation();
                          logger.info(`User opened edit dialog for project: ${project.name}`);
                          setEditingProject(project);
                        }}
                      >
                        <Pencil className="h-4 w-4 mr-2" />
                        Edit
                      </DropdownMenuItem>
                      <DropdownMenuItem
                        onClick={(e) => {
                          e.stopPropagation();
                          logger.info(`User opened delete dialog for project: ${project.name}`);
                          setDeletingProject(project);
                        }}
                        className="text-destructive focus:text-destructive"
                      >
                        <Trash2 className="h-4 w-4 mr-2" />
                        Delete
                      </DropdownMenuItem>
                    </DropdownMenuContent>
                  </DropdownMenu>
                </div>
                <CardHeader className="pb-3">
                  <CardTitle>{project.name}</CardTitle>
                </CardHeader>
                <div className="border-t border-border/50 mx-6 mb-3" />
                <CardContent className="space-y-3 pb-3">
                  {project.description && (
                    <>
                      <p className="text-sm text-muted-foreground line-clamp-4">
                        {project.description}
                      </p>
                      <div className="border-t border-border/50" />
                    </>
                  )}

                  <div className="space-y-2">
                    <p className="text-sm text-muted-foreground">
                      {getClassificationModelName(project.classification_model_id)}
                    </p>

                  </div>
                </CardContent>
                <div className="border-t border-border/50 mx-6" />
                <CardFooter className="pt-3 pb-4 px-6">
                  <p className="text-xs text-muted-foreground">
                    {formatCompact(project.file_count)} files
                    {" · "}
                    {formatCompact(project.observation_count)} observations
                    {" · "}
                    {formatCompact(project.deployment_count)} deployments
                    {" · "}
                    {formatCompact(project.trap_nights)} trap nights
                  </p>
                </CardFooter>
              </Card>
            ))}
          </div>
        ) : (
          <Card>
            <CardContent className="flex flex-col items-center justify-center py-12">
              <p className="max-w-md text-center text-muted-foreground">
                There are no projects yet.{" "}
                <button
                  type="button"
                  onClick={() => setCreateDialogOpen(true)}
                  className="font-medium text-primary hover:underline"
                >
                  Click here
                </button>{" "}
                to create your first one.
              </p>
            </CardContent>
          </Card>
        )}
      </main>

      <CreateProjectDialog
        open={createDialogOpen}
        onOpenChange={setCreateDialogOpen}
      />

      {editingProject && (
        <EditProjectDialog
          project={editingProject}
          open={!!editingProject}
          onOpenChange={(open) => !open && setEditingProject(null)}
        />
      )}

      <DeleteProjectDialog
        project={deletingProject}
        open={!!deletingProject}
        onOpenChange={(open) => !open && setDeletingProject(null)}
      />
    </div>
  );
}

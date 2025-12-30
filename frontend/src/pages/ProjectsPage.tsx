/**
 * Projects list page.
 *
 * Following DEVELOPERS.md principles:
 * - Type hints everywhere
 * - Simple, clear structure
 */

import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { useNavigate } from "react-router-dom";
import { Plus, MoreVertical, Pencil, Copy, Trash2 } from "lucide-react";
import { projectsApi, type ProjectResponse } from "../api/projects";
import { modelsApi } from "../api/models";
import { logger } from "../lib/logger";
import { Button } from "../components/ui/button";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "../components/ui/card";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "../components/ui/dropdown-menu";
import { CreateProjectDialog } from "../components/projects/CreateProjectDialog";
import { EditProjectDialog } from "../components/projects/EditProjectDialog";
import { DuplicateProjectDialog } from "../components/projects/DuplicateProjectDialog";
import { DeleteProjectDialog } from "../components/projects/DeleteProjectDialog";

export function ProjectsPage() {
  const navigate = useNavigate();
  const [createDialogOpen, setCreateDialogOpen] = useState(false);
  const [editingProject, setEditingProject] = useState<ProjectResponse | null>(null);
  const [duplicatingProject, setDuplicatingProject] = useState<ProjectResponse | null>(null);
  const [deletingProject, setDeletingProject] = useState<ProjectResponse | null>(null);

  const { data: projects, isLoading, error } = useQuery({
    queryKey: ["projects"],
    queryFn: () => projectsApi.getProjects(),
  });


  const { data: classificationModels = [] } = useQuery({
    queryKey: ["models", "classification"],
    queryFn: () => modelsApi.listClassificationModels(),
  });

  const { data: locations } = useQuery({
    queryKey: ["speciesnet-locations"],
    queryFn: () => modelsApi.getSpeciesNetLocations(),
  });

  // Helper to get classification model name by ID
  const getClassificationModelName = (modelId: string | null) => {
    if (!modelId || modelId === "none") return "None";
    const model = classificationModels.find((m) => m.model_id === modelId);
    return model ? `${model.emoji} ${model.friendly_name}` : modelId;
  };

  // Helper to check if a model is SpeciesNet
  const isSpeciesNet = (modelId: string | null) => {
    return modelId?.toLowerCase().includes("speciesnet") ?? false;
  };

  // Helper to get country name by code
  const getCountryName = (code: string | null) => {
    if (!code || !locations) return null;
    return Object.entries(locations.countries).find(([_, c]) => c === code)?.[0];
  };

  // Helper to get state name by code
  const getStateName = (code: string | null) => {
    if (!code || !locations) return null;
    return Object.entries(locations.us_states).find(([_, c]) => c === code)?.[0];
  };

  // Log errors
  if (error) {
    logger.error("Failed to load projects", { error: error.message });
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100">
      {/* Header */}
      <header className="border-b bg-white/80 backdrop-blur-sm">
        <div className="mx-auto max-w-7xl px-4 py-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold tracking-tight">Projects</h1>
              <p className="text-sm text-muted-foreground">
                Manage your camera trap monitoring projects
              </p>
            </div>
            <Button
              onClick={() => {
                logger.info("User clicked New Project button");
                setCreateDialogOpen(true);
              }}
            >
              <Plus className="h-4 w-4" />
              New Project
            </Button>
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
            {projects.map((project: ProjectResponse) => (
              <Card
                key={project.id}
                className="transition-shadow hover:shadow-lg cursor-pointer"
                onClick={() => {
                  logger.info(`User navigated to project: ${project.name}`, {
                    projectId: project.id,
                  });
                  navigate(`/projects/${project.id}/analyses`);
                }}
              >
                <CardHeader className="pb-3">
                  <div className="flex items-center justify-between">
                    <div className="flex-1">
                      <CardTitle>{project.name}</CardTitle>
                    </div>
                    <DropdownMenu>
                      <DropdownMenuTrigger asChild>
                        <Button
                          variant="ghost"
                          size="sm"
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
                            logger.info(`User opened duplicate dialog for project: ${project.name}`);
                            setDuplicatingProject(project);
                          }}
                        >
                          <Copy className="h-4 w-4 mr-2" />
                          Duplicate
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
                </CardHeader>
                <div className="border-t border-border/50 mx-6 mb-3" />
                <CardContent className="space-y-3">
                  {project.description && (
                    <>
                      <p className="text-sm text-muted-foreground line-clamp-4">
                        {project.description}
                      </p>
                      <div className="border-t border-border/50" />
                    </>
                  )}

                  <div className="space-y-2">
                    <p className="text-sm font-medium">
                      {getClassificationModelName(project.classification_model_id)}
                    </p>

                    {isSpeciesNet(project.classification_model_id) && (
                      <>
                        <div className="border-t border-border/50" />
                        <div className="space-y-2 text-sm text-muted-foreground">
                          {getCountryName(project.country_code) && (
                            <p>{getCountryName(project.country_code)}</p>
                          )}
                          {getCountryName(project.country_code) && getStateName(project.state_code) && (
                            <div className="border-t border-border/50" />
                          )}
                          {getStateName(project.state_code) && (
                            <p>{getStateName(project.state_code)}</p>
                          )}
                        </div>
                      </>
                    )}
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        ) : (
          <Card>
            <CardContent className="flex flex-col items-center justify-center py-12">
              <p className="mb-4 text-muted-foreground">
                No projects yet. Create your first project to get started.
              </p>
              <Button onClick={() => setCreateDialogOpen(true)}>
                <Plus className="h-4 w-4" />
                Create Project
              </Button>
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

      <DuplicateProjectDialog
        project={duplicatingProject}
        open={!!duplicatingProject}
        onOpenChange={(open) => !open && setDuplicatingProject(null)}
      />

      <DeleteProjectDialog
        project={deletingProject}
        open={!!deletingProject}
        onOpenChange={(open) => !open && setDeletingProject(null)}
      />
    </div>
  );
}

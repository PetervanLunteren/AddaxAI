/**
 * Projects list page.
 *
 * Following DEVELOPERS.md principles:
 * - Type hints everywhere
 * - Simple, clear structure
 */

import { useState, useEffect } from "react";
import { useQuery } from "@tanstack/react-query";
import { useNavigate } from "react-router-dom";
import { Plus, Pencil, Copy, Trash2, Calendar, Scan, ScrollText } from "lucide-react";
import { projectsApi, type ProjectResponse } from "../api/projects";
import { modelsApi } from "../api/models";
import { logger } from "../lib/logger";
import { Button } from "../components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "../components/ui/card";
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

  // Debug: log projects data to see classification_model_id values
  useEffect(() => {
    if (projects) {
      console.log("Projects data:", projects);
      projects.forEach(p => {
        console.log(`Project "${p.name}":`, {
          detection_model_id: p.detection_model_id,
          classification_model_id: p.classification_model_id,
          taxonomy_config: p.taxonomy_config
        });
      });
    }
  }, [projects]);

  const { data: classificationModels = [] } = useQuery({
    queryKey: ["models", "classification"],
    queryFn: () => modelsApi.listClassificationModels(),
  });

  // Helper to get classification model name by ID
  const getClassificationModelName = (modelId: string | null) => {
    if (!modelId || modelId === "none") return "None";
    console.log(`Getting classification model name for:`, modelId, "Available models:", classificationModels);
    const model = classificationModels.find((m) => m.model_id === modelId);
    console.log(`Found model:`, model);
    return model ? `${model.emoji} ${model.friendly_name}` : modelId;
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
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <CardTitle>{project.name}</CardTitle>
                    </div>
                    <div className="flex items-center gap-1">
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={(e) => {
                          e.stopPropagation();
                          logger.info(`User opened edit dialog for project: ${project.name}`);
                          setEditingProject(project);
                        }}
                        title="Edit project"
                      >
                        <Pencil className="h-4 w-4" />
                      </Button>
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={(e) => {
                          e.stopPropagation();
                          logger.info(`User opened duplicate dialog for project: ${project.name}`);
                          setDuplicatingProject(project);
                        }}
                        title="Duplicate project"
                      >
                        <Copy className="h-4 w-4" />
                      </Button>
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={(e) => {
                          e.stopPropagation();
                          logger.info(`User opened delete dialog for project: ${project.name}`);
                          setDeletingProject(project);
                        }}
                        title="Delete project"
                      >
                        <Trash2 className="h-4 w-4" />
                      </Button>
                    </div>
                  </div>
                </CardHeader>
                <CardContent className="space-y-2.5">
                  <div className="space-y-2 text-sm">
                    {project.description && (
                      <div className="flex items-start gap-2.5">
                        <ScrollText className="h-4 w-4 mt-0.5 text-muted-foreground flex-shrink-0" />
                        <p className="text-muted-foreground line-clamp-2 flex-1">
                          {project.description}
                        </p>
                      </div>
                    )}

                    <div className="flex items-center gap-2.5">
                      <Scan className="h-4 w-4 text-muted-foreground flex-shrink-0" />
                      <p className="font-medium truncate">
                        {getClassificationModelName(project.classification_model_id)}
                      </p>
                    </div>

                    <div className="flex items-center gap-2.5">
                      <Calendar className="h-4 w-4 text-muted-foreground flex-shrink-0" />
                      <p className="text-sm text-muted-foreground">
                        {new Date(project.created_at).toLocaleDateString()}
                      </p>
                    </div>
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

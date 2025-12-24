/**
 * Duplicate Project Dialog.
 */

import { useEffect } from "react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import * as z from "zod";
import { Info } from "lucide-react";
import { projectsApi, type ProjectCreate, type ProjectResponse } from "../../api/projects";
import { Button } from "../ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "../ui/dialog";
import {
  Form,
  FormControl,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
} from "../ui/form";
import { Input } from "../ui/input";
import { Textarea } from "../ui/textarea";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "../ui/tooltip";

const projectSchema = z.object({
  name: z.string().min(1, "Project name is required").max(100, "Name too long"),
  description: z.string().max(500, "Description too long").optional(),
  detection_model_id: z.string().min(1, "Detection model is required"),
  classification_model_id: z.string().nullable(),
  taxonomy_config: z.object({
    selected_classes: z.array(z.string()),
  }),
});

interface DuplicateProjectDialogProps {
  project: ProjectResponse | null;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export function DuplicateProjectDialog({
  project,
  open,
  onOpenChange,
}: DuplicateProjectDialogProps) {
  const queryClient = useQueryClient();

  const form = useForm<ProjectCreate>({
    resolver: zodResolver(projectSchema),
    defaultValues: {
      name: "",
      description: "",
      detection_model_id: project?.detection_model_id || "MD5A-0-0",
      classification_model_id: project?.classification_model_id || null,
      taxonomy_config: project?.taxonomy_config || { selected_classes: [] },
    },
  });

  // Reset form when project changes
  useEffect(() => {
    if (project) {
      form.reset({
        name: "",
        description: "",
        detection_model_id: project.detection_model_id,
        classification_model_id: project.classification_model_id,
        taxonomy_config: project.taxonomy_config,
      });
    }
  }, [project, form]);

  const createMutation = useMutation({
    mutationFn: (data: ProjectCreate) => projectsApi.create(data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["projects"] });
      form.reset();
      onOpenChange(false);
    },
    onError: (error: Error) => {
      form.setError("root", {
        message: error.message || "Failed to duplicate project",
      });
    },
  });

  const onSubmit = (data: ProjectCreate) => {
    createMutation.mutate(data);
  };

  if (!project) return null;

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-2xl">
        <DialogHeader>
          <DialogTitle>Duplicate project</DialogTitle>
          <DialogDescription>
            Create a copy of "{project.name}" with the same models and taxonomy configuration
          </DialogDescription>
        </DialogHeader>

        <Form {...form}>
          <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-6">
            <TooltipProvider>
              <FormField
                control={form.control}
                name="name"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel className="flex items-center gap-1.5">
                      Project name
                      <Tooltip>
                        <TooltipTrigger asChild>
                          <Info className="h-3.5 w-3.5 text-muted-foreground cursor-help" />
                        </TooltipTrigger>
                        <TooltipContent>
                          <p className="max-w-xs">
                            A unique name for the duplicated project
                          </p>
                        </TooltipContent>
                      </Tooltip>
                    </FormLabel>
                    <FormControl>
                      <Input placeholder="e.g., Yellowstone 2024 (Copy)" {...field} />
                    </FormControl>
                    <FormMessage />
                  </FormItem>
                )}
              />

              <FormField
                control={form.control}
                name="description"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel className="flex items-center gap-1.5">
                      Description
                      <Tooltip>
                        <TooltipTrigger asChild>
                          <Info className="h-3.5 w-3.5 text-muted-foreground cursor-help" />
                        </TooltipTrigger>
                        <TooltipContent>
                          <p className="max-w-xs">
                            Optional notes about the project's purpose, location, team members, etc.
                          </p>
                        </TooltipContent>
                      </Tooltip>
                    </FormLabel>
                    <FormControl>
                      <Textarea
                        placeholder="Brief description of the project"
                        className="resize-y min-h-[80px]"
                        {...field}
                      />
                    </FormControl>
                    <FormMessage />
                  </FormItem>
                )}
              />

              <div className="rounded-lg border bg-muted/50 p-4">
                <p className="text-sm text-muted-foreground">
                  <strong>Note:</strong> The duplicated project will use the same detection model, classification model, and taxonomy configuration as the original.
                </p>
              </div>

              {form.formState.errors.root && (
                <p className="text-sm font-medium text-destructive">
                  {form.formState.errors.root.message}
                </p>
              )}

              <DialogFooter>
                <Button
                  type="button"
                  variant="outline"
                  onClick={() => onOpenChange(false)}
                >
                  Cancel
                </Button>
                <Button type="submit" disabled={createMutation.isPending}>
                  {createMutation.isPending ? "Duplicating..." : "Duplicate project"}
                </Button>
              </DialogFooter>
            </TooltipProvider>
          </form>
        </Form>
      </DialogContent>
    </Dialog>
  );
}

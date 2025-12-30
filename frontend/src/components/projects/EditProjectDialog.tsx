/**
 * Edit Project Dialog.
 */

import { useEffect, useState } from "react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import * as z from "zod";
import { Info, AlertTriangle } from "lucide-react";
import { projectsApi, type ProjectUpdate, type ProjectResponse } from "../../api/projects";
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
  FormDescription,
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
  description: z
    .string()
    .max(500, "Description too long")
    .optional()
    .transform((val) => (val === "" ? undefined : val)),
});

interface EditProjectDialogProps {
  project: ProjectResponse;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export function EditProjectDialog({
  project,
  open,
  onOpenChange,
}: EditProjectDialogProps) {
  const queryClient = useQueryClient();
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);
  const [deleteConfirmText, setDeleteConfirmText] = useState("");

  const form = useForm<ProjectUpdate>({
    resolver: zodResolver(projectSchema),
    defaultValues: {
      name: project.name,
      description: project.description || "",
    },
  });

  // Reset form when project changes
  useEffect(() => {
    form.reset({
      name: project.name,
      description: project.description || "",
    });
  }, [project, form]);

  // Reset delete confirmation when dialog opens/closes
  useEffect(() => {
    if (!open) {
      setShowDeleteConfirm(false);
      setDeleteConfirmText("");
    }
  }, [open]);

  const updateMutation = useMutation({
    mutationFn: (data: ProjectUpdate) => projectsApi.update(project.id, data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["projects"] });
      queryClient.invalidateQueries({ queryKey: ["projects", project.id] });
      onOpenChange(false);
    },
    onError: (error: Error) => {
      form.setError("root", {
        message: error.message || "Failed to update project",
      });
    },
  });


  const onSubmit = (data: ProjectUpdate) => {
    updateMutation.mutate(data);
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-xl">
        <DialogHeader>
          <DialogTitle>Edit project</DialogTitle>
          <DialogDescription>
            Update project details
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
                            A unique name for your project
                          </p>
                        </TooltipContent>
                      </Tooltip>
                    </FormLabel>
                    <FormControl>
                      <Input placeholder="e.g., Yellowstone camera trap project" {...field} />
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
                        className="resize-y"
                        rows={2}
                        maxLength={500}
                        {...field}
                      />
                    </FormControl>
                    <div className="flex items-center justify-between">
                      <FormMessage />
                      <p className={`text-xs ${
                        (field.value?.length || 0) > 450
                          ? "text-orange-600"
                          : "text-muted-foreground"
                      }`}>
                        {field.value?.length || 0} / 500
                      </p>
                    </div>
                  </FormItem>
                )}
              />

              <div className="rounded-lg border bg-muted/50 p-4">
                <p className="text-sm text-muted-foreground">
                  <strong>Note:</strong> Other project settings can be edited in the project settings page.
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
                <Button type="submit" disabled={updateMutation.isPending}>
                  {updateMutation.isPending ? "Saving..." : "Save changes"}
                </Button>
              </DialogFooter>
            </TooltipProvider>
          </form>
        </Form>
      </DialogContent>
    </Dialog>
  );
}

/**
 * Edit Project Dialog.
 */

import { useEffect, useState } from "react";
import { useForm, type Resolver } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import * as z from "zod";
import { projectsApi, type ProjectUpdate, type ProjectResponse } from "../../api/projects";
import { API_BASE_URL } from "../../lib/api-client";
import { ImageDropZone } from "./ImageDropZone";
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
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [removeImage, setRemoveImage] = useState(false);
  const [, setShowDeleteConfirm] = useState(false);
  const [, setDeleteConfirmText] = useState("");

  const form = useForm<ProjectUpdate>({
    resolver: zodResolver(projectSchema) as Resolver<ProjectUpdate>,
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

  // Reset state when dialog opens/closes
  useEffect(() => {
    if (!open) {
      setShowDeleteConfirm(false);
      setDeleteConfirmText("");
      setImageFile(null);
      setRemoveImage(false);
    }
  }, [open]);

  const updateMutation = useMutation({
    mutationFn: (data: ProjectUpdate) => projectsApi.update(project.id, data),
    onSuccess: async () => {
      try {
        if (removeImage && !imageFile) {
          await projectsApi.deleteThumbnail(project.id);
        }
        if (imageFile) {
          await projectsApi.uploadThumbnail(project.id, imageFile);
        }
      } catch (e) {
        console.error("Failed to update project image:", e);
      }
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

  const existingThumbnailUrl = project.thumbnail_path && !removeImage
    ? `${API_BASE_URL}/api/projects/${project.id}/thumbnail`
    : null;

  const onSubmit = (data: ProjectUpdate) => {
    updateMutation.mutate(data);
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-xl max-h-[85vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle>Edit project</DialogTitle>
          <DialogDescription>
            Update project details
          </DialogDescription>
        </DialogHeader>

        <Form {...form}>
          <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-6">
              <FormField
                control={form.control}
                name="name"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Project name</FormLabel>
                    <FormControl>
                      <Input placeholder="e.g., Yellowstone camera trap project" {...field} value={field.value ?? ""} />
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
                    <FormLabel>Description</FormLabel>
                    <FormControl>
                      <Textarea
                        placeholder="Notes about purpose, location, team members, etc."
                        className="resize-y"
                        rows={2}
                        maxLength={500}
                        {...field}
                        value={field.value ?? ""}
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

              <ImageDropZone
                value={imageFile}
                existingUrl={existingThumbnailUrl}
                onChange={setImageFile}
                onRemove={() => {
                  setRemoveImage(true);
                  setImageFile(null);
                }}
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
          </form>
        </Form>
      </DialogContent>
    </Dialog>
  );
}

/**
 * Create Project Dialog.
 *
 * Following DEVELOPERS.md principles:
 * - Type hints everywhere
 * - Simple, clear validation
 * - Explicit error handling
 */

import { useState, useEffect } from "react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import * as z from "zod";
import { projectsApi, type ProjectCreate } from "../../api/projects";
import { modelsApi } from "../../api/models";
import { TaxonomyEditor } from "../taxonomy/TaxonomyEditor";
import { Button } from "../ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "../ui/select";
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

const projectSchema = z.object({
  name: z.string().min(1, "Project name is required").max(100, "Name too long"),
  description: z.string().max(500, "Description too long").optional(),
  detection_model_id: z.string().min(1, "Detection model is required"),
  classification_model_id: z.string().nullable(),
  taxonomy_config: z.object({
    selected_classes: z.array(z.string()),
  }),
});

interface CreateProjectDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export function CreateProjectDialog({
  open,
  onOpenChange,
}: CreateProjectDialogProps) {
  const queryClient = useQueryClient();
  const [taxonomyEditorOpen, setTaxonomyEditorOpen] = useState(false);

  // Fetch available models
  const { data: detectionModels = [] } = useQuery({
    queryKey: ["models", "detection"],
    queryFn: () => modelsApi.listDetectionModels(),
    enabled: open,
  });

  const { data: classificationModels = [] } = useQuery({
    queryKey: ["models", "classification"],
    queryFn: () => modelsApi.listClassificationModels(),
    enabled: open,
  });

  const form = useForm<ProjectCreate>({
    resolver: zodResolver(projectSchema),
    defaultValues: {
      name: "",
      description: "",
      detection_model_id: "MD5A-0-0",
      classification_model_id: null,
      taxonomy_config: { selected_classes: [] },
    },
  });

  // Fetch taxonomy to get total class count
  const classificationModelId = form.watch("classification_model_id");
  const { data: taxonomy, isLoading: taxonomyLoading } = useQuery({
    queryKey: ["taxonomy", classificationModelId],
    queryFn: () => modelsApi.getTaxonomy(classificationModelId!),
    enabled: open && !!classificationModelId && classificationModelId !== "none",
  });

  const createMutation = useMutation({
    mutationFn: (data: ProjectCreate) => projectsApi.create(data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["projects"] });
      form.reset();
      onOpenChange(false);
    },
    onError: (error: Error) => {
      console.error("Failed to create project:", error);
      // Set form error
      form.setError("root", {
        message: error.message || "Failed to create project",
      });
    },
  });

  const onSubmit = (data: ProjectCreate) => {
    createMutation.mutate(data);
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Create new project</DialogTitle>
          <DialogDescription>
            Create a new camera trap monitoring project
          </DialogDescription>
        </DialogHeader>

        <Form {...form}>
          <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-4">
            <FormField
              control={form.control}
              name="name"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Project name</FormLabel>
                  <FormControl>
                    <Input placeholder="e.g., Yellowstone 2024" {...field} />
                  </FormControl>
                  <FormDescription>
                    A unique name for your project
                  </FormDescription>
                  <FormMessage />
                </FormItem>
              )}
            />

            <FormField
              control={form.control}
              name="description"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Description (optional)</FormLabel>
                  <FormControl>
                    <Input
                      placeholder="Brief description of the project"
                      {...field}
                    />
                  </FormControl>
                  <FormDescription>
                    Optional notes about the project's purpose or location
                  </FormDescription>
                  <FormMessage />
                </FormItem>
              )}
            />

            <FormField
              control={form.control}
              name="detection_model_id"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Detection model</FormLabel>
                  <Select onValueChange={field.onChange} defaultValue={field.value}>
                    <FormControl>
                      <SelectTrigger>
                        <SelectValue placeholder="Select detection model" />
                      </SelectTrigger>
                    </FormControl>
                    <SelectContent>
                      {detectionModels.map((model) => (
                        <SelectItem key={model.model_id} value={model.model_id}>
                          {model.emoji} {model.friendly_name}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                  <FormDescription>
                    Model to detect animals in your images
                  </FormDescription>
                  <FormMessage />
                </FormItem>
              )}
            />

            <FormField
              control={form.control}
              name="classification_model_id"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Classification model (optional)</FormLabel>
                  <Select
                    onValueChange={(value) => field.onChange(value === "none" ? null : value)}
                    defaultValue={field.value || "none"}
                  >
                    <FormControl>
                      <SelectTrigger>
                        <SelectValue placeholder="Select classification model" />
                      </SelectTrigger>
                    </FormControl>
                    <SelectContent>
                      {classificationModels.map((model) => (
                        <SelectItem key={model.model_id} value={model.model_id}>
                          {model.emoji} {model.friendly_name}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                  <FormDescription>
                    Model to identify species after detection
                  </FormDescription>
                  <FormMessage />
                </FormItem>
              )}
            />

            {/* Taxonomy configuration button */}
            {form.watch("classification_model_id") && form.watch("classification_model_id") !== "none" && (
              <div className="space-y-2">
                <label className="text-sm font-medium">Species taxonomy</label>
                <div className="flex items-center gap-2">
                  <Button
                    type="button"
                    variant="outline"
                    onClick={() => setTaxonomyEditorOpen(true)}
                  >
                    Select species
                  </Button>
                  <span className="text-sm text-muted-foreground">
                    Selected: {form.watch("taxonomy_config")?.selected_classes?.length || taxonomy?.all_classes?.length || 0} of {taxonomy?.all_classes?.length || 0}
                  </span>
                </div>
              </div>
            )}

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
                {createMutation.isPending ? "Creating..." : "Create project"}
              </Button>
            </DialogFooter>
          </form>
        </Form>
      </DialogContent>

      {/* Taxonomy Editor */}
      <TaxonomyEditor
        open={taxonomyEditorOpen}
        onOpenChange={setTaxonomyEditorOpen}
        modelId={form.watch("classification_model_id")}
        selectedClasses={form.watch("taxonomy_config")?.selected_classes || []}
        onSave={(selectedClasses) => {
          form.setValue("taxonomy_config", { selected_classes: selectedClasses });
        }}
      />
    </Dialog>
  );
}

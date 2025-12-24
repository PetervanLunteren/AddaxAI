/**
 * Create Project Dialog.
 *
 * Following DEVELOPERS.md principles:
 * - Type hints everywhere
 * - Simple, clear validation
 * - Explicit error handling
 */

import { useState, useEffect, useRef } from "react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import * as z from "zod";
import { Info, CheckSquare, Square, ChevronDown, ChevronRight } from "lucide-react";
import { projectsApi, type ProjectCreate } from "../../api/projects";
import { modelsApi } from "../../api/models";
import type { TaxonomyNode } from "../../api/types";
import { TreeNode } from "../taxonomy/TreeNode";
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
import { Textarea } from "../ui/textarea";
import { ScrollArea } from "../ui/scroll-area";
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

interface CreateProjectDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export function CreateProjectDialog({
  open,
  onOpenChange,
}: CreateProjectDialogProps) {
  const queryClient = useQueryClient();
  const [selectedClasses, setSelectedClasses] = useState<Set<string>>(new Set());
  const [expandedNodes, setExpandedNodes] = useState<Set<string>>(new Set());
  const [initializedForModel, setInitializedForModel] = useState<string | null>(null);

  // Fetch available models (already sorted alphabetically by backend)
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

  const allClasses = taxonomy?.all_classes || [];
  const tree = taxonomy?.tree || [];

  // Reset when model changes to none or null
  useEffect(() => {
    if (!classificationModelId || classificationModelId === "none") {
      setSelectedClasses(new Set());
      setExpandedNodes(new Set());
      setInitializedForModel(null);
      form.setValue("taxonomy_config", { selected_classes: [] });
    }
  }, [classificationModelId, form]);

  // Initialize with all classes when taxonomy loads for a new model
  useEffect(() => {
    if (
      classificationModelId &&
      classificationModelId !== "none" &&
      taxonomy?.all_classes &&
      !taxonomyLoading &&
      initializedForModel !== classificationModelId
    ) {
      const newSelected = new Set(taxonomy.all_classes);
      setSelectedClasses(newSelected);
      form.setValue("taxonomy_config", { selected_classes: taxonomy.all_classes });
      setInitializedForModel(classificationModelId);
    }
  }, [classificationModelId, taxonomy?.all_classes, taxonomyLoading, initializedForModel, form]);

  // Tree handlers
  const handleToggle = (nodeId: string, checked: boolean) => {
    const newSelected = new Set(selectedClasses);

    const findAndToggleNode = (nodes: TaxonomyNode[]): boolean => {
      for (const node of nodes) {
        if (node.id === nodeId) {
          toggleNodeAndDescendants(node, checked, newSelected);
          return true;
        }
        if (node.children && findAndToggleNode(node.children)) {
          return true;
        }
      }
      return false;
    };

    findAndToggleNode(tree);
    setSelectedClasses(newSelected);
    form.setValue("taxonomy_config", { selected_classes: Array.from(newSelected) });
  };

  const toggleNodeAndDescendants = (
    node: TaxonomyNode,
    checked: boolean,
    selectedSet: Set<string>
  ) => {
    if (!node.children || node.children.length === 0) {
      if (checked) {
        selectedSet.add(node.id);
      } else {
        selectedSet.delete(node.id);
      }
    } else {
      for (const child of node.children) {
        toggleNodeAndDescendants(child, checked, selectedSet);
      }
    }
  };

  const handleSelectAll = () => {
    const newSelected = new Set(allClasses);
    setSelectedClasses(newSelected);
    form.setValue("taxonomy_config", { selected_classes: allClasses });
  };

  const handleDeselectAll = () => {
    setSelectedClasses(new Set());
    form.setValue("taxonomy_config", { selected_classes: [] });
  };

  const handleExpandAll = () => {
    const allNodeIds = new Set<string>();
    const collectAllNodeIds = (nodes: TaxonomyNode[]) => {
      for (const node of nodes) {
        allNodeIds.add(node.id);
        if (node.children) {
          collectAllNodeIds(node.children);
        }
      }
    };
    collectAllNodeIds(tree);
    setExpandedNodes(allNodeIds);
  };

  const handleCollapseAll = () => {
    setExpandedNodes(new Set());
  };

  const handleExpand = (nodeId: string, expanded: boolean) => {
    const newExpanded = new Set(expandedNodes);
    if (expanded) {
      newExpanded.add(nodeId);
    } else {
      newExpanded.delete(nodeId);
    }
    setExpandedNodes(newExpanded);
  };

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
      <DialogContent className="max-w-3xl max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle>Create new project</DialogTitle>
          <DialogDescription>
            Create a new camera trap monitoring project
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
                      <Input placeholder="e.g., Yellowstone 2024" {...field} />
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
                        className="resize-y min-h-12"
                        {...field}
                      />
                    </FormControl>
                    <FormMessage />
                  </FormItem>
                )}
              />

              <FormField
                control={form.control}
                name="detection_model_id"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel className="flex items-center gap-1.5">
                      Detection model
                      <Tooltip>
                        <TooltipTrigger asChild>
                          <Info className="h-3.5 w-3.5 text-muted-foreground cursor-help" />
                        </TooltipTrigger>
                        <TooltipContent>
                          <p className="max-w-xs">
                            The AI model that will scan your data for the presence of animals, persons, and vehicles. It does not identify the animals, it just labels them as a generic "animal"
                          </p>
                        </TooltipContent>
                      </Tooltip>
                    </FormLabel>
                    <Select onValueChange={field.onChange} defaultValue={field.value}>
                      <FormControl>
                        <SelectTrigger className="h-12">
                          <SelectValue placeholder="Select detection model" />
                        </SelectTrigger>
                      </FormControl>
                      <SelectContent>
                        {detectionModels.map((model) => (
                          <SelectItem key={model.model_id} value={model.model_id}>
                            <div className="flex flex-col gap-0.5 items-start text-left">
                              <div>{model.emoji} {model.friendly_name}</div>
                              <div className="text-xs text-muted-foreground">
                                {model.description.length > 50
                                  ? `${model.description.substring(0, 50)}...`
                                  : model.description}
                              </div>
                            </div>
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                    <FormMessage />
                  </FormItem>
                )}
              />

              <FormField
                control={form.control}
                name="classification_model_id"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel className="flex items-center gap-1.5">
                      Classification model
                      <Tooltip>
                        <TooltipTrigger asChild>
                          <Info className="h-3.5 w-3.5 text-muted-foreground cursor-help" />
                        </TooltipTrigger>
                        <TooltipContent>
                          <p className="max-w-xs">
                            After detecting animals, this model will identify the species.
                            Choose a model trained on species from your geographic region.
                          </p>
                        </TooltipContent>
                      </Tooltip>
                    </FormLabel>
                    <Select
                      onValueChange={(value) => field.onChange(value === "none" ? null : value)}
                      defaultValue={field.value || "none"}
                    >
                      <FormControl>
                        <SelectTrigger className="h-12">
                          <SelectValue placeholder="Select classification model" />
                        </SelectTrigger>
                      </FormControl>
                      <SelectContent>
                        {classificationModels.map((model) => (
                          <SelectItem key={model.model_id} value={model.model_id}>
                            <div className="flex flex-col gap-0.5 items-start text-left">
                              <div>{model.emoji} {model.friendly_name}</div>
                              <div className="text-xs text-muted-foreground">
                                {model.description.length > 50
                                  ? `${model.description.substring(0, 50)}...`
                                  : model.description}
                              </div>
                            </div>
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                    <FormMessage />
                  </FormItem>
                )}
              />

              {/* Species selection - inline */}
              {classificationModelId && classificationModelId !== "none" && (
                <div className="space-y-2">
                  <FormLabel className="flex items-center gap-1.5">
                    Species selection
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <Info className="h-3.5 w-3.5 text-muted-foreground cursor-help" />
                      </TooltipTrigger>
                      <TooltipContent>
                        <p className="max-w-xs">
                          Select which species the model should identify. All species are selected by default,
                          but narrowing this to only species present in your study area will improve
                          classification accuracy.
                        </p>
                      </TooltipContent>
                    </Tooltip>
                  </FormLabel>

                  <div className="space-y-2 border rounded-md p-3">
                    <div className="flex items-center justify-between gap-2 flex-wrap">
                      <Button
                        type="button"
                        variant="outline"
                        size="sm"
                        onClick={handleSelectAll}
                        className="flex-1 min-w-[100px]"
                      >
                        <CheckSquare className="h-4 w-4 mr-1.5" />
                        Select all
                      </Button>
                      <Button
                        type="button"
                        variant="outline"
                        size="sm"
                        onClick={handleDeselectAll}
                        className="flex-1 min-w-[100px]"
                      >
                        <Square className="h-4 w-4 mr-1.5" />
                        Deselect all
                      </Button>
                      <Button
                        type="button"
                        variant="outline"
                        size="sm"
                        onClick={handleExpandAll}
                        className="flex-1 min-w-[100px]"
                      >
                        <ChevronDown className="h-4 w-4 mr-1.5" />
                        Expand all
                      </Button>
                      <Button
                        type="button"
                        variant="outline"
                        size="sm"
                        onClick={handleCollapseAll}
                        className="flex-1 min-w-[100px]"
                      >
                        <ChevronRight className="h-4 w-4 mr-1.5" />
                        Collapse all
                      </Button>
                    </div>

                    <p className="text-sm text-muted-foreground">
                      Currently selected: {selectedClasses.size} of {allClasses.length}
                    </p>

                    {taxonomyLoading ? (
                      <div className="text-sm text-muted-foreground py-4">
                        Loading taxonomy...
                      </div>
                    ) : tree.length > 0 ? (
                      <ScrollArea className="h-[300px] border rounded-md p-4 bg-background">
                        <div className="space-y-1">
                          {tree.map((node) => (
                            <TreeNode
                              key={node.id}
                              node={node}
                              selectedClasses={selectedClasses}
                              expandedNodes={expandedNodes}
                              onToggle={handleToggle}
                              onExpand={handleExpand}
                              level={0}
                            />
                          ))}
                        </div>
                      </ScrollArea>
                    ) : (
                      <div className="text-sm text-muted-foreground py-4">
                        No taxonomy available for this model
                      </div>
                    )}
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
            </TooltipProvider>
          </form>
        </Form>
      </DialogContent>
    </Dialog>
  );
}

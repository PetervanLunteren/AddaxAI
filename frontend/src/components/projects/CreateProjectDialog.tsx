/**
 * Create Project Dialog.
 *
 * Following DEVELOPERS.md principles:
 * - Type hints everywhere
 * - Simple, clear validation
 * - Explicit error handling
 */

import { useEffect, useState } from "react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import * as z from "zod";
import { Info, Check, ChevronsUpDown, InfoIcon } from "lucide-react";
import { projectsApi, type ProjectCreate } from "../../api/projects";
import { modelsApi } from "../../api/models";
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
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "../ui/tooltip";
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
} from "../ui/command";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "../ui/popover";
import { ModelInfoSheet } from "../models/ModelInfoSheet";
import { cn } from "../../lib/utils";

const projectSchema = z.object({
  name: z.string().min(1, "Project name is required").max(100, "Name too long"),
  description: z.string().max(500, "Description too long").optional(),
  detection_model_id: z.literal("MD5A-0-0"),
  classification_model_id: z.string().min(1, "Classification model is required"),
  excluded_classes: z.array(z.string()),
  country_code: z.string().optional().nullable(),
  state_code: z.string().optional().nullable(),
  detection_threshold: z.number().min(0).max(1),
  event_smoothing: z.boolean(),
  taxonomic_rollup: z.boolean(),
  taxonomic_rollup_threshold: z.number().min(0.1).max(1.0),
  independence_interval: z.number().min(0),
}).refine(
  (data) => {
    // If SpeciesNet is selected, country must be provided
    const isSpeciesNet = data.classification_model_id?.toLowerCase().includes("speciesnet");
    if (isSpeciesNet && !data.country_code) {
      return false;
    }
    return true;
  },
  {
    message: "Country is required for SpeciesNet models",
    path: ["country_code"],
  }
).refine(
  (data) => {
    // If SpeciesNet is selected and country is USA, state must be provided
    const isSpeciesNet = data.classification_model_id?.toLowerCase().includes("speciesnet");
    if (isSpeciesNet && data.country_code === "USA" && !data.state_code) {
      return false;
    }
    return true;
  },
  {
    message: "State is required when USA is selected",
    path: ["state_code"],
  }
);

interface CreateProjectDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export function CreateProjectDialog({
  open,
  onOpenChange,
}: CreateProjectDialogProps) {
  const queryClient = useQueryClient();
  const [countryOpen, setCountryOpen] = useState(false);
  const [stateOpen, setStateOpen] = useState(false);
  const [showModelInfo, setShowModelInfo] = useState(false);

  // Fetch available classification models (already sorted alphabetically by backend)
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
      classification_model_id: "",
      excluded_classes: [],
      country_code: null,
      state_code: null,
      detection_threshold: 0.5,
      event_smoothing: true,
      taxonomic_rollup: true,
      taxonomic_rollup_threshold: 0.65,
      independence_interval: 1800, // Will be converted from minutes in UI
    },
  });

  // Watch classification model and country changes
  const classificationModelId = form.watch("classification_model_id");
  const countryCode = form.watch("country_code");

  // Check if current model is SpeciesNet
  const isSpeciesNet = classificationModelId?.toLowerCase().includes("speciesnet");

  // Fetch locations for SpeciesNet models
  const { data: locations } = useQuery({
    queryKey: ["speciesnet-locations"],
    queryFn: () => modelsApi.getSpeciesNetLocations(),
    enabled: isSpeciesNet && open,
  });

  // Clear state_code when country changes away from USA
  useEffect(() => {
    if (countryCode !== "USA" && form.getValues("state_code")) {
      form.setValue("state_code", null);
    }
  }, [countryCode, form]);

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
      <DialogContent className="max-w-xl max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle>Create new project</DialogTitle>
          <DialogDescription>
            Projects organize your camera trap sites, deployments, and analysis settings
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
                            The AI model that will identify species in your camera trap images.
                            Choose a model trained on species from your geographic region.
                          </p>
                        </TooltipContent>
                      </Tooltip>
                    </FormLabel>
                    <div className="flex gap-2">
                      <Select
                        onValueChange={field.onChange}
                        defaultValue={field.value}
                      >
                        <FormControl>
                          <SelectTrigger>
                            <SelectValue placeholder="Select classification model">
                            {field.value && (() => {
                              const selectedModel = classificationModels.find(
                                (m) => m.model_id === field.value
                              );
                              if (!selectedModel) return null;
                              return (
                                <div className="flex flex-col items-start py-1">
                                  <div>
                                    {selectedModel.emoji} {selectedModel.friendly_name}
                                  </div>
                                  {selectedModel.description_short && (
                                    <div className="text-xs text-muted-foreground">
                                      {selectedModel.description_short}
                                    </div>
                                  )}
                                </div>
                              );
                            })()}
                          </SelectValue>
                        </SelectTrigger>
                      </FormControl>
                      <SelectContent>
                        {classificationModels
                          .filter((model) => model.model_id !== "none")
                          .map((model) => (
                            <SelectItem key={model.model_id} value={model.model_id}>
                              {model.emoji} {model.friendly_name}
                              {model.description_short && (
                                <>
                                  <br />
                                  <span className="text-xs text-muted-foreground">{model.description_short}</span>
                                </>
                              )}
                            </SelectItem>
                          ))}
                      </SelectContent>
                    </Select>
                    <Button
                      type="button"
                      variant="outline"
                      size="icon"
                      onClick={() => setShowModelInfo(true)}
                      disabled={!field.value}
                      title="View model information"
                    >
                      <InfoIcon className="h-4 w-4" />
                    </Button>
                  </div>
                  <FormMessage />
                </FormItem>
              )}
            />

              {/* Country Selection (SpeciesNet only) */}
              {isSpeciesNet && locations && (
                <FormField
                  control={form.control}
                  name="country_code"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel className="flex items-center gap-1.5">
                        Country
                        <Tooltip>
                          <TooltipTrigger asChild>
                            <Info className="h-3.5 w-3.5 text-muted-foreground cursor-help" />
                          </TooltipTrigger>
                          <TooltipContent>
                            <p className="max-w-xs">
                              SpeciesNet uses geographic location to improve species predictions.
                              Select the country where your camera traps are located.
                            </p>
                          </TooltipContent>
                        </Tooltip>
                      </FormLabel>
                      <Popover open={countryOpen} onOpenChange={setCountryOpen}>
                        <PopoverTrigger asChild>
                          <FormControl>
                            <Button
                              variant="outline"
                              role="combobox"
                              className={cn(
                                "w-full justify-between",
                                !field.value && "text-muted-foreground"
                              )}
                            >
                              {field.value
                                ? Object.entries(locations.countries).find(
                                    ([_, code]) => code === field.value
                                  )?.[0]
                                : "Select country"}
                              <ChevronsUpDown className="ml-2 h-4 w-4 shrink-0 opacity-50" />
                            </Button>
                          </FormControl>
                        </PopoverTrigger>
                        <PopoverContent className="w-[500px] p-0">
                          <Command>
                            <CommandInput placeholder="Search countries..." />
                            <CommandList>
                              <CommandEmpty>No country found.</CommandEmpty>
                              <CommandGroup>
                                {Object.entries(locations.countries).map(([name, code]) => (
                                  <CommandItem
                                    key={code}
                                    value={name}
                                    onSelect={() => {
                                      form.setValue("country_code", code);
                                      setCountryOpen(false);
                                    }}
                                  >
                                    <Check
                                      className={cn(
                                        "mr-2 h-4 w-4",
                                        field.value === code ? "opacity-100" : "opacity-0"
                                      )}
                                    />
                                    {name}
                                  </CommandItem>
                                ))}
                              </CommandGroup>
                            </CommandList>
                          </Command>
                        </PopoverContent>
                      </Popover>
                      <FormMessage />
                    </FormItem>
                  )}
                />
              )}

              {/* State Selection (USA only) */}
              {isSpeciesNet && countryCode === "USA" && locations && (
                <FormField
                  control={form.control}
                  name="state_code"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel className="flex items-center gap-1.5">
                        State
                        <Tooltip>
                          <TooltipTrigger asChild>
                            <Info className="h-3.5 w-3.5 text-muted-foreground cursor-help" />
                          </TooltipTrigger>
                          <TooltipContent>
                            <p className="max-w-xs">
                              Select a US state for more specific SpeciesNet predictions.
                            </p>
                          </TooltipContent>
                        </Tooltip>
                      </FormLabel>
                      <Popover open={stateOpen} onOpenChange={setStateOpen}>
                        <PopoverTrigger asChild>
                          <FormControl>
                            <Button
                              variant="outline"
                              role="combobox"
                              className={cn(
                                "w-full justify-between",
                                !field.value && "text-muted-foreground"
                              )}
                            >
                              {field.value
                                ? Object.entries(locations.us_states).find(
                                    ([_, code]) => code === field.value
                                  )?.[0]
                                : "Select state"}
                              <ChevronsUpDown className="ml-2 h-4 w-4 shrink-0 opacity-50" />
                            </Button>
                          </FormControl>
                        </PopoverTrigger>
                        <PopoverContent className="w-[500px] p-0">
                          <Command>
                            <CommandInput placeholder="Search states..." />
                            <CommandList>
                              <CommandEmpty>No state found.</CommandEmpty>
                              <CommandGroup>
                                {Object.entries(locations.us_states).map(([name, code]) => (
                                  <CommandItem
                                    key={code}
                                    value={name}
                                    onSelect={() => {
                                      form.setValue("state_code", code);
                                      setStateOpen(false);
                                    }}
                                  >
                                    <Check
                                      className={cn(
                                        "mr-2 h-4 w-4",
                                        field.value === code ? "opacity-100" : "opacity-0"
                                      )}
                                    />
                                    {name}
                                  </CommandItem>
                                ))}
                              </CommandGroup>
                            </CommandList>
                          </Command>
                        </PopoverContent>
                      </Popover>
                      <FormMessage />
                    </FormItem>
                  )}
                />
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

      {/* Model Info Sheet */}
      <ModelInfoSheet
        modelId={form.watch("classification_model_id")}
        open={showModelInfo}
        onOpenChange={setShowModelInfo}
      />
    </Dialog>
  );
}

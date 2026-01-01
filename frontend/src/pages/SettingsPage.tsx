/**
 * Project Settings Page.
 *
 * Following DEVELOPERS.md principles:
 * - Type hints everywhere
 * - Simple, clear structure
 * - Explicit error handling
 */

import { useEffect, useState } from "react";
import { useParams } from "react-router-dom";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import * as z from "zod";
import { Save, RotateCcw, Settings2, Check, ChevronsUpDown, ListTodo, InfoIcon } from "lucide-react";
import { projectsApi, type ProjectUpdate } from "../api/projects";
import { modelsApi } from "../api/models";
import { SpeciesSelectionModal } from "../components/taxonomy/SpeciesSelectionModal";
import { ModelInfoSheet } from "../components/models/ModelInfoSheet";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "../components/ui/tooltip";
import { Button } from "../components/ui/button";
import { Slider } from "../components/ui/slider";
import { Switch } from "../components/ui/switch";
import { Input } from "../components/ui/input";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "../components/ui/card";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "../components/ui/select";
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
} from "../components/ui/command";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "../components/ui/popover";
import {
  Form,
  FormControl,
  FormDescription,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
} from "../components/ui/form";
import { cn } from "../lib/utils";

const settingsSchema = z.object({
  detection_model_id: z.string().min(1, "Detection model is required"),
  classification_model_id: z.string().min(1, "Classification model is required"),
  excluded_classes: z.array(z.string()),
  country_code: z.string().optional().nullable(),
  state_code: z.string().optional().nullable(),
  detection_threshold: z.number().min(0).max(1),
  event_smoothing: z.boolean(),
  taxonomic_rollup: z.boolean(),
  taxonomic_rollup_threshold: z.number().min(0.1).max(1.0),
  independence_interval: z.number().min(0),
});

type SettingsFormData = z.infer<typeof settingsSchema>;

export default function SettingsPage() {
  const { projectId } = useParams<{ projectId: string }>();
  const queryClient = useQueryClient();
  const [excludedClasses, setExcludedClasses] = useState<string[]>([]);
  const [speciesModalOpen, setSpeciesModalOpen] = useState(false);
  const [countryOpen, setCountryOpen] = useState(false);
  const [stateOpen, setStateOpen] = useState(false);
  const [showModelInfo, setShowModelInfo] = useState(false);
  const [selectedModelId, setSelectedModelId] = useState<string | null>(null);

  // Fetch current project
  const { data: project, isLoading: projectLoading } = useQuery({
    queryKey: ["projects", projectId],
    queryFn: () => projectsApi.get(projectId!),
    enabled: !!projectId,
  });

  // Fetch available models
  const { data: detectionModels = [] } = useQuery({
    queryKey: ["models", "detection"],
    queryFn: () => modelsApi.listDetectionModels(),
  });

  const { data: classificationModels = [] } = useQuery({
    queryKey: ["models", "classification"],
    queryFn: () => modelsApi.listClassificationModels(),
  });

  const form = useForm<SettingsFormData>({
    resolver: zodResolver(settingsSchema),
    defaultValues: {
      detection_model_id: "MD5A-0-0",
      classification_model_id: "",
      excluded_classes: [],
      country_code: null,
      state_code: null,
      detection_threshold: 0.5,
      event_smoothing: true,
      taxonomic_rollup: true,
      taxonomic_rollup_threshold: 0.65,
      independence_interval: 1800,
    },
  });

  // Update form values when project loads
  useEffect(() => {
    if (project) {
      form.reset({
        detection_model_id: project.detection_model_id,
        classification_model_id: project.classification_model_id || "",
        excluded_classes: project.excluded_classes || [],
        country_code: project.country_code || null,
        state_code: project.state_code || null,
        detection_threshold: project.detection_threshold,
        event_smoothing: project.event_smoothing,
        taxonomic_rollup: project.taxonomic_rollup,
        taxonomic_rollup_threshold: project.taxonomic_rollup_threshold,
        independence_interval: project.independence_interval,
      });

      // WORKAROUND: Set state_code again after a tick to ensure the field is rendered
      // This handles the race condition where the state field is conditionally rendered
      // based on country_code === "USA"
      if (project.state_code) {
        setTimeout(() => {
          form.setValue("state_code", project.state_code);
        }, 0);
      }
    }
  }, [project, form]);

  // Watch classification model changes
  const classificationModelId = form.watch("classification_model_id");
  const countryCode = form.watch("country_code");

  // Check if current model is SpeciesNet
  const isSpeciesNet = classificationModelId?.toLowerCase().includes("speciesnet");

  // Fetch taxonomy for selected classification model (non-SpeciesNet only)
  const { data: taxonomy } = useQuery({
    queryKey: ["taxonomy", classificationModelId],
    queryFn: () => modelsApi.getTaxonomy(classificationModelId!),
    enabled: !!classificationModelId && !isSpeciesNet,
  });

  // Fetch locations for SpeciesNet models
  const { data: locations } = useQuery({
    queryKey: ["speciesnet-locations"],
    queryFn: () => modelsApi.getSpeciesNetLocations(),
    enabled: isSpeciesNet,
  });

  // Initialize excludedClasses state when project loads
  useEffect(() => {
    if (project) {
      const savedExcluded = project.excluded_classes || [];
      setExcludedClasses(savedExcluded);
    }
  }, [project]);

  // Clear state_code when country changes away from USA
  useEffect(() => {
    if (countryCode !== "USA" && form.getValues("state_code")) {
      form.setValue("state_code", null, { shouldDirty: true });
    }
  }, [countryCode, form]);

  // Clear excluded_classes when classification model changes
  useEffect(() => {
    if (classificationModelId && taxonomy?.all_classes) {
      // Filter excluded_classes to only keep species that exist in the new model
      const currentExcluded = form.getValues("excluded_classes");
      const validExcluded = currentExcluded.filter(cls =>
        taxonomy.all_classes.includes(cls)
      );

      // Only update if some species were removed
      if (validExcluded.length !== currentExcluded.length) {
        form.setValue("excluded_classes", validExcluded, { shouldDirty: true });
        setExcludedClasses(validExcluded);
      }
    }
  }, [classificationModelId, taxonomy, form]);

  // Update mutation
  const updateMutation = useMutation({
    mutationFn: (data: ProjectUpdate) => projectsApi.update(projectId!, data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["projects"] });
      queryClient.invalidateQueries({ queryKey: ["projects", projectId] });
      // Reset form dirty state
      form.reset(form.getValues());
    },
    onError: (error: Error) => {
      form.setError("root", {
        message: error.message || "Failed to update project settings",
      });
    },
  });

  const onSubmit = (data: SettingsFormData) => {
    updateMutation.mutate(data);
  };

  const handleReset = () => {
    if (project) {
      form.reset({
        detection_model_id: project.detection_model_id,
        classification_model_id: project.classification_model_id || "",
        excluded_classes: project.excluded_classes || [],
        country_code: project.country_code || null,
        state_code: project.state_code || null,
        detection_threshold: project.detection_threshold,
        event_smoothing: project.event_smoothing,
        taxonomic_rollup: project.taxonomic_rollup,
        taxonomic_rollup_threshold: project.taxonomic_rollup_threshold,
        independence_interval: project.independence_interval,
      });
      setExcludedClasses(project.excluded_classes || []);
    }
  };

  if (projectLoading) {
    return (
      <div className="p-8">
        <div className="text-muted-foreground">Loading project settings...</div>
      </div>
    );
  }

  if (!project) {
    return (
      <div className="p-8">
        <div className="text-destructive">Project not found</div>
      </div>
    );
  }

  const isDirty = form.formState.isDirty;

  return (
    <div className="p-8 bg-gradient-to-br from-slate-50 to-slate-100 min-h-screen">
      <div className="mx-auto max-w-5xl space-y-6">
        {/* Header */}
        <div>
          <h1 className="text-2xl font-bold tracking-tight">Project settings</h1>
          <p className="text-sm text-muted-foreground mt-1">
            Configure AI models, species selection, and analysis parameters
          </p>
        </div>

        {/* Settings form */}
        <TooltipProvider>
          <Form {...form}>
            <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-6" key={project?.id}>
            {/* Card 1: Models */}
            <Card>
              <CardHeader>
                <CardTitle>Models</CardTitle>
                <CardDescription>
                  Models used to detect objects and classify species. Changes apply to new analyses only and do not reprocess existing results.
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-0 divide-y">
                {/* Detection Model */}
                <FormField
                  control={form.control}
                  name="detection_model_id"
                  render={({ field }) => (
                    <div className="grid grid-cols-[55%_1fr] gap-8 py-6">
                      <div className="space-y-1">
                        <FormLabel>Detection model</FormLabel>
                        <FormDescription className="text-sm">
                          Used to find animals, people, and vehicles.
                        </FormDescription>
                      </div>
                      <div className="space-y-2">
                        <div className="flex gap-2 items-stretch">
                          <Select onValueChange={field.onChange} value={field.value}>
                            <FormControl>
                              <SelectTrigger>
                                <SelectValue placeholder="Select detection model">
                                  {field.value && (() => {
                                    const selectedModel = detectionModels.find(
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
                              {detectionModels.map((model) => (
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
                          <Tooltip>
                            <TooltipTrigger asChild>
                              <span className="self-center">
                                <Button
                                  type="button"
                                  variant="outline"
                                  className="px-3"
                                  onClick={() => {
                                    if (field.value) {
                                      setSelectedModelId(field.value);
                                      setShowModelInfo(true);
                                    }
                                  }}
                                  disabled={!field.value}
                                >
                                  <InfoIcon className="h-4 w-4" />
                                </Button>
                              </span>
                            </TooltipTrigger>
                            <TooltipContent>
                              <p>
                                {field.value
                                  ? "View model information"
                                  : "Select a detection model to view details"}
                              </p>
                            </TooltipContent>
                          </Tooltip>
                        </div>
                        <FormMessage />
                      </div>
                    </div>
                  )}
                />

                {/* Classification Model */}
                <FormField
                  control={form.control}
                  name="classification_model_id"
                  render={({ field }) => (
                    <div className="grid grid-cols-[55%_1fr] gap-8 py-6">
                      <div className="space-y-1">
                        <FormLabel>Classification model</FormLabel>
                        <FormDescription className="text-sm">
                          Used to identify species for detected animals.
                        </FormDescription>
                      </div>
                      <div className="space-y-2">
                        <div className="flex gap-2 items-stretch">
                          <Select
                            key={field.value}
                            onValueChange={field.onChange}
                            value={field.value}
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
                          <Tooltip>
                            <TooltipTrigger asChild>
                              <span className="self-center">
                                <Button
                                  type="button"
                                  variant="outline"
                                  className="px-3"
                                  onClick={() => {
                                    if (field.value) {
                                      setSelectedModelId(field.value);
                                      setShowModelInfo(true);
                                    }
                                  }}
                                  disabled={!field.value}
                                >
                                  <InfoIcon className="h-4 w-4" />
                                </Button>
                              </span>
                            </TooltipTrigger>
                            <TooltipContent>
                              <p>
                                {field.value
                                  ? "View model information"
                                  : "Select a classification model to view details"}
                              </p>
                            </TooltipContent>
                          </Tooltip>
                        </div>
                        <FormMessage />
                      </div>
                    </div>
                  )}
                />

              </CardContent>
            </Card>

            {/* Card 2: Geographic location (SpeciesNet) OR Species selection (other models) */}
            {classificationModelId && isSpeciesNet && locations && (
              <Card>
                <CardHeader>
                  <CardTitle>Geographic location</CardTitle>
                  <CardDescription>
                    Select the location used for SpeciesNet predictions. Changes apply to new analyses only and do not reprocess existing results.
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-0 divide-y">
                  {/* Country Selection */}
                  <FormField
                    control={form.control}
                    name="country_code"
                    render={({ field }) => (
                      <div className="grid grid-cols-[55%_1fr] gap-8 py-6">
                        <div className="space-y-1">
                          <FormLabel>Country</FormLabel>
                          <FormDescription className="text-sm">
                            Select the country where your camera traps are located.
                          </FormDescription>
                        </div>
                        <div className="space-y-2">
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
                            <PopoverContent className="w-[400px] p-0">
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
                                          form.setValue("country_code", code, { shouldDirty: true });
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
                        </div>
                      </div>
                    )}
                  />

                  {/* State Selection (USA only) */}
                  {countryCode === "USA" && (
                    <FormField
                      control={form.control}
                      name="state_code"
                      render={({ field }) => (
                        <div className="grid grid-cols-[55%_1fr] gap-8 py-6">
                          <div className="space-y-1">
                            <FormLabel>State</FormLabel>
                            <FormDescription className="text-sm">
                              Select a US state for more specific predictions.
                            </FormDescription>
                          </div>
                          <div className="space-y-2">
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
                              <PopoverContent className="w-[400px] p-0">
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
                                            form.setValue("state_code", code, { shouldDirty: true });
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
                          </div>
                        </div>
                      )}
                    />
                  )}
                </CardContent>
              </Card>
            )}

            {classificationModelId && !isSpeciesNet && taxonomy && (
              <Card>
                <CardHeader>
                  <CardTitle>Species selection</CardTitle>
                  <CardDescription>
                    Control which species can be predicted
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-[55%_1fr] gap-8">
                    <div className="space-y-1">
                      <FormLabel>Species selection</FormLabel>
                      <FormDescription className="text-sm">
                        Limit predictions to species expected in your project area to reduce false positives.
                      </FormDescription>
                    </div>
                    <div className="space-y-1">
                      <Button
                        type="button"
                        variant="outline"
                        onClick={() => setSpeciesModalOpen(true)}
                        className="w-full min-h-14 flex flex-col items-start justify-center gap-1 text-left"
                      >
                        <div className="flex items-center gap-2">
                          <ListTodo className="h-4 w-4" />
                          <span>Select species</span>
                        </div>
                        <span className="text-xs text-muted-foreground">
                          Currently included {(taxonomy.all_classes?.length || 0) - excludedClasses.length} of {taxonomy.all_classes?.length || 0}
                        </span>
                      </Button>
                    </div>
                  </div>
                </CardContent>
              </Card>
            )}

            {/* Card 3: Analysis and counting */}
            <Card>
              <CardHeader>
                <CardTitle>Analysis and counting</CardTitle>
                <CardDescription>
                  Control how detections are filtered, grouped, and aggregated. Changes apply to all analyses (past and future) and affect how data is interpreted, not the underlying detections.
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-0 divide-y">
                {/* Detection Threshold */}
                <FormField
                  control={form.control}
                  name="detection_threshold"
                  render={({ field }) => (
                    <div className="grid grid-cols-[55%_1fr] gap-8 py-6">
                      <div className="space-y-1">
                        <FormLabel>Detection confidence threshold</FormLabel>
                        <FormDescription className="text-sm">
                          Hide and exclude detections below this value. Applies to existing and new analyses.
                        </FormDescription>
                      </div>
                      <div className="space-y-2">
                        <div className="flex items-center justify-between">
                          <Slider
                            min={0.1}
                            max={1.0}
                            step={0.01}
                            value={[field.value]}
                            onValueChange={(vals) => field.onChange(vals[0])}
                            className="flex-1 mr-4"
                          />
                          <span className="text-sm font-medium min-w-[3rem] text-right">{field.value.toFixed(2)}</span>
                        </div>
                        <FormMessage />
                      </div>
                    </div>
                  )}
                />

                {/* Independence Interval */}
                <FormField
                  control={form.control}
                  name="independence_interval"
                  render={({ field }) => (
                    <div className="grid grid-cols-[55%_1fr] gap-8 py-6">
                      <div className="space-y-1">
                        <FormLabel>Independence interval</FormLabel>
                        <FormDescription className="text-sm">
                          Group detections into one event when they occur within this time gap.
                        </FormDescription>
                      </div>
                      <div className="space-y-2">
                        <div className="flex items-center gap-2 max-w-xs">
                          <Input
                            type="number"
                            min={0}
                            value={Math.round(field.value / 60)}
                            onChange={(e) => field.onChange(parseInt(e.target.value || "0") * 60)}
                            className="flex-1"
                          />
                          <span className="text-sm text-muted-foreground">minutes</span>
                        </div>
                        <FormMessage />
                      </div>
                    </div>
                  )}
                />

                {/* Event Smoothing */}
                <FormField
                  control={form.control}
                  name="event_smoothing"
                  render={({ field }) => (
                    <div className="grid grid-cols-[55%_1fr] gap-8 py-6">
                      <div className="space-y-1">
                        <FormLabel>Event smoothing</FormLabel>
                        <FormDescription className="text-sm">
                          Reduce noisy labels within an event by averaging predictions.
                        </FormDescription>
                      </div>
                      <div className="flex items-center">
                        <Switch
                          checked={field.value}
                          onCheckedChange={field.onChange}
                        />
                      </div>
                    </div>
                  )}
                />

                {/* Taxonomic Rollup */}
                <FormField
                  control={form.control}
                  name="taxonomic_rollup"
                  render={({ field }) => (
                    <div className="grid grid-cols-[55%_1fr] gap-8 py-6">
                      <div className="space-y-1">
                        <FormLabel>Taxonomic rollup</FormLabel>
                        <FormDescription className="text-sm">
                          If the model's confidence at the species level is below 0.65, it rolls up to the next higher taxonomic level at which the summed confidence reaches 0.65.
                        </FormDescription>
                      </div>
                      <div className="flex items-center">
                        <Switch
                          checked={field.value}
                          onCheckedChange={field.onChange}
                        />
                      </div>
                    </div>
                  )}
                />

              </CardContent>
            </Card>

            {form.formState.errors.root && (
              <p className="text-sm font-medium text-destructive">
                {form.formState.errors.root.message}
              </p>
            )}

            {/* Action buttons */}
            <div className="flex items-center justify-between">
              <Button
                type="button"
                variant="ghost"
                onClick={() => {
                  const currentClassificationModel = form.getValues("classification_model_id");
                  const currentExcludedClasses = form.getValues("excluded_classes");
                  const currentCountryCode = form.getValues("country_code");
                  const currentStateCode = form.getValues("state_code");
                  form.reset({
                    detection_model_id: "MD5A-0-0",
                    classification_model_id: currentClassificationModel,
                    excluded_classes: currentExcludedClasses,
                    country_code: currentCountryCode,
                    state_code: currentStateCode,
                    detection_threshold: 0.5,
                    event_smoothing: true,
                    taxonomic_rollup: true,
                    taxonomic_rollup_threshold: 0.65,
                    independence_interval: 1800,
                  });
                }}
                disabled={updateMutation.isPending}
              >
                Restore defaults
              </Button>
              <div className="flex items-center gap-3">
                <Button
                  type="button"
                  variant="outline"
                  onClick={handleReset}
                  disabled={!isDirty || updateMutation.isPending}
                >
                  <RotateCcw className="h-4 w-4 mr-2" />
                  Reset
                </Button>
                <Button
                  type="submit"
                  disabled={!isDirty || updateMutation.isPending}
                >
                  <Save className="h-4 w-4 mr-2" />
                  {updateMutation.isPending ? "Saving..." : "Save changes"}
                </Button>
              </div>
            </div>
          </form>
        </Form>
        </TooltipProvider>

        {/* Model Info Sheet */}
        <ModelInfoSheet
          modelId={selectedModelId}
          open={showModelInfo}
          onOpenChange={setShowModelInfo}
        />

        {/* Species Selection Modal */}
        {classificationModelId && taxonomy && (
          <SpeciesSelectionModal
            modelId={classificationModelId}
            excludedClasses={excludedClasses}
            onExclusionChange={(classes) => {
              setExcludedClasses(classes);
              form.setValue("excluded_classes", classes, { shouldDirty: true });
            }}
            open={speciesModalOpen}
            onOpenChange={setSpeciesModalOpen}
            totalSpeciesCount={taxonomy.all_classes?.length || 0}
          />
        )}
      </div>
    </div>
  );
}

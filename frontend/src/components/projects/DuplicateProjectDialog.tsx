/**
 * Duplicate Project Dialog.
 *
 * Creates a new project from an existing one's structure. Mirrors the
 * create-project fields (prefilled from the source) plus checkboxes for what
 * to carry over: processing settings, sites, and the source's deployments
 * re-queued for reprocessing. Analyzed results are never copied across
 * projects. Project names are unique, so a new name is required.
 */

import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import * as z from "zod";
import { toast } from "sonner";
import { projectsApi, type ProjectWithStats } from "../../api/projects";
import { modelsApi } from "../../api/models";
import { ImageDropZone } from "./ImageDropZone";
import { Button } from "../ui/button";
import { Input } from "../ui/input";
import { Textarea } from "../ui/textarea";
import { Checkbox } from "../ui/checkbox";
import { Callout } from "../ui/callout";
import { SelectItem } from "../ui/select";
import { ClassificationModelGroupedItems } from "../models/ClassificationModelGroupedItems";
import { ModelSelect } from "../models/ModelSelect";
import { NoClassifierNotice } from "../models/NoClassifierNotice";
import { ModelInfoSheet } from "../models/ModelInfoSheet";
import {
  LabelSelectionField,
  toApiCountryCode,
  useLabelSelectionCaption,
} from "../taxonomy/LabelSelectionField";
import { FieldHeader } from "../ui/field-header";
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

const duplicateSchema = z.object({
  name: z.string().min(1, "Project name is required").max(100, "Name too long"),
  description: z.string().max(500, "Description too long").optional(),
  classification_model_id: z.string().nullable().optional(),
  excluded_classes: z.array(z.string()),
  country_code: z.string().optional().nullable(),
  state_code: z.string().optional().nullable(),
});

type DuplicateForm = z.infer<typeof duplicateSchema>;

interface DuplicateProjectDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  source: ProjectWithStats | null;
}

function CopyRow({
  checked,
  onChange,
  title,
  caption,
}: {
  checked: boolean;
  onChange: (v: boolean) => void;
  title: string;
  caption: string;
}) {
  return (
    <label className="flex items-start gap-3 cursor-pointer">
      <Checkbox checked={checked} onCheckedChange={onChange} className="mt-0.5" />
      <span className="space-y-0.5">
        <span className="block text-sm font-medium">{title}</span>
        <span className="block text-xs text-muted-foreground">{caption}</span>
      </span>
    </label>
  );
}

export function DuplicateProjectDialog({
  open,
  onOpenChange,
  source,
}: DuplicateProjectDialogProps) {
  const queryClient = useQueryClient();
  const navigate = useNavigate();

  const [imageFile, setImageFile] = useState<File | null>(null);
  const [showModelInfo, setShowModelInfo] = useState(false);
  const [copySettings, setCopySettings] = useState(true);
  const [copySites, setCopySites] = useState(true);
  const [copyDeployments, setCopyDeployments] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const form = useForm<DuplicateForm>({
    resolver: zodResolver(duplicateSchema),
    defaultValues: {
      name: "",
      description: "",
      classification_model_id: null,
      excluded_classes: [],
      country_code: null,
      state_code: null,
    },
  });

  // Prefill from the source when the dialog opens.
  useEffect(() => {
    if (!open || !source) return;
    form.reset({
      // Left empty so the placeholder shows the suggestion; the user types a
      // new (unique) name rather than accepting a presumptuous default.
      name: "",
      description: "",
      classification_model_id: source.classification_model_id ?? null,
      excluded_classes: source.excluded_classes ?? [],
      country_code: source.country_code ?? null,
      state_code: source.state_code ?? null,
    });
    setImageFile(null);
    setCopySettings(true);
    setCopySites(true);
    setCopyDeployments(true);
    setError(null);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, source]);

  const { data: classificationModels = [] } = useQuery({
    queryKey: ["models", "classification"],
    queryFn: () => modelsApi.listClassificationModels(),
    enabled: open,
  });

  const classificationModelId = form.watch("classification_model_id");
  const excludedClasses = form.watch("excluded_classes");
  const hasClassifier =
    !!classificationModelId && classificationModelId !== "none";
  const labelCaption = useLabelSelectionCaption(
    hasClassifier ? classificationModelId! : "",
  );

  const { data: taxonomy } = useQuery({
    queryKey: ["taxonomy", classificationModelId],
    queryFn: () => modelsApi.getTaxonomy(classificationModelId!),
    enabled: open && hasClassifier,
  });

  // Independent: copying Deployments without Sites just means the re-queued
  // folders come in without a site assignment (which you can set later).

  const mutation = useMutation({
    mutationFn: async (data: DuplicateForm) => {
      const created = await projectsApi.duplicate(source!.id, {
        name: data.name.trim(),
        description: data.description?.trim() || null,
        classification_model_id: hasClassifier
          ? data.classification_model_id
          : null,
        excluded_classes: data.excluded_classes,
        // ALL is a form-only sentinel; the API knows ISO codes or null.
        country_code: toApiCountryCode(data.country_code),
        state_code: data.state_code ?? null,
        copy_settings: copySettings,
        copy_sites: copySites,
        copy_deployments: copyDeployments,
      });
      if (imageFile) {
        try {
          await projectsApi.uploadThumbnail(created.id, imageFile);
        } catch (e) {
          console.error("Failed to upload project image:", e);
        }
      }
      return created;
    },
    onSuccess: (created) => {
      queryClient.invalidateQueries({ queryKey: ["projects"] });
      toast.success(`Created "${created.name}"`);
      onOpenChange(false);
      navigate(`/projects/${created.id}/dashboard`);
    },
    onError: (e: Error) => {
      setError(e.message || "Could not duplicate project");
    },
  });

  if (!source) return null;

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-2xl max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle>Duplicate project</DialogTitle>
          <DialogDescription>
            Create a new project from "{source.name}". Pick what to carry over.
          </DialogDescription>
        </DialogHeader>

        <Form {...form}>
          <form
            onSubmit={form.handleSubmit((data) => mutation.mutate(data))}
            className="space-y-6 py-2"
          >
            <Callout variant="info" size="compact">
              Use this to start a new project from the same baseline, not to
              clone the original. It copies settings and metadata only, never
              analysis results, verifications, confirmed counts, or your images
              and videos.
            </Callout>

            <FormField
              control={form.control}
              name="name"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Project name</FormLabel>
                  <FormControl>
                    <Input
                      placeholder={`${source.name} (copy)`}
                      {...field}
                    />
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
                      rows={2}
                      maxLength={500}
                      placeholder={`Duplicate of ${source.name}`}
                      {...field}
                      value={field.value ?? ""}
                    />
                  </FormControl>
                  <FormMessage />
                </FormItem>
              )}
            />

            <FormField
              control={form.control}
              name="classification_model_id"
              render={({ field }) => (
                <FormItem>
                  <FieldHeader
                    label={<FormLabel>Classification model</FormLabel>}
                    caption="The AI model that identifies species in your images. Pick one trained for your region."
                  />
                  <ModelSelect
                    value={field.value ?? "none"}
                    onValueChange={(val) =>
                      field.onChange(val === "none" ? null : val)
                    }
                    models={classificationModels}
                    placeholder="Select classification model"
                    noneValue="none"
                    noneLabel="No classification model"
                    onShowInfo={() => setShowModelInfo(true)}
                  >
                    <SelectItem value="none">
                      ∅ No classification model
                      <br />
                      <span className="text-xs text-muted-foreground">
                        Run animal detector only, identify species manually
                      </span>
                    </SelectItem>
                    <ClassificationModelGroupedItems
                      models={classificationModels.filter(
                        (m) => m.model_id !== "none",
                      )}
                    />
                  </ModelSelect>
                  {!hasClassifier && <NoClassifierNotice />}
                  <FormMessage />
                </FormItem>
              )}
            />

            {hasClassifier && taxonomy && (
              <FormItem>
                <FieldHeader
                  label={<FormLabel>Label selection</FormLabel>}
                  caption={labelCaption}
                />
                <LabelSelectionField
                  modelId={classificationModelId!}
                  excludedClasses={excludedClasses}
                  allClasses={taxonomy.all_classes ?? []}
                  countryCode={form.watch("country_code")}
                  stateCode={form.watch("state_code")}
                  onExclusionChange={(classes) =>
                    form.setValue("excluded_classes", classes, {
                      shouldDirty: true,
                    })
                  }
                  onLocationChange={(c, s) => {
                    form.setValue("country_code", c, { shouldDirty: true });
                    form.setValue("state_code", s, { shouldDirty: true });
                  }}
                />
              </FormItem>
            )}

            <ImageDropZone
              value={imageFile}
              existingUrl={null}
              onChange={setImageFile}
            />

            <div className="space-y-3 rounded-md border p-3">
              <p className="text-sm font-medium">
                Carry over from "{source.name}"
              </p>
              <CopyRow
                checked={copySettings}
                onChange={setCopySettings}
                title="Settings"
                caption="Detection and embedding models, batch sizes, frame rate, detection threshold, smoothing, rollup, independence interval, and other processing options. Off starts from defaults."
              />
              <CopyRow
                checked={copySites}
                onChange={setCopySites}
                title="Sites"
                caption="All camera sites including metadata like name, location, habitat, tags."
              />
              <CopyRow
                checked={copyDeployments}
                onChange={setCopyDeployments}
                title="Deployments"
                caption="Existing deployments can't be copied with their results. Their folders are added to this project's queue to reprocess."
              />
            </div>

            {error && (
              <p className="text-sm font-medium text-destructive">{error}</p>
            )}

            <DialogFooter>
              <Button
                type="button"
                variant="outline"
                onClick={() => onOpenChange(false)}
              >
                Cancel
              </Button>
              <Button type="submit" disabled={mutation.isPending}>
                {mutation.isPending ? "Duplicating..." : "Duplicate project"}
              </Button>
            </DialogFooter>
          </form>
        </Form>
      </DialogContent>

      <ModelInfoSheet
        modelId={classificationModelId ?? null}
        open={showModelInfo}
        onOpenChange={setShowModelInfo}
      />
    </Dialog>
  );
}

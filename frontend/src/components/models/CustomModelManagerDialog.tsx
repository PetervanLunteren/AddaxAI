import { useEffect, useMemo, useRef, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { FolderOpen, Pencil, Plus, Trash2 } from "lucide-react";
import { toast } from "sonner";

import { modelsApi } from "@/api/models";
import type {
  CustomModelCreateRequest,
  CustomDetectorBackend,
  CustomModelInfo,
  CustomModelInspectResponse,
  CustomModelType,
  CustomModelUpdateRequest,
  ModelInfo,
} from "@/api/types";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";

const REGION_OPTIONS = ["global", "africa", "americas", "asia", "europe", "oceania"] as const;
const RFDETR_CLASSES = [
  "RFDETRBase", "RFDETRNano", "RFDETRSmall", "RFDETRMedium", "RFDETRLarge",
  "RFDETRXLarge", "RFDETR2XLarge", "RFDETRSegNano", "RFDETRSegSmall",
  "RFDETRSegMedium", "RFDETRSegLarge", "RFDETRSegXLarge", "RFDETRSeg2XLarge",
  "RFDETRKeypointPreview", "RFDETRSegPreview",
];

interface Props {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  modelType: CustomModelType;
  onCreated?: (model: CustomModelInfo) => void;
  onAnyCreated?: (model: CustomModelInfo) => void;
}

function getError(error: unknown): string {
  return error instanceof Error ? error.message : "Model management failed";
}

export function CustomModelManagerDialog({
  open,
  onOpenChange,
  modelType,
  onCreated,
  onAnyCreated,
}: Props) {
  const queryClient = useQueryClient();
  const inspectGeneration = useRef(0);
  const [type, setType] = useState<CustomModelType | "">(modelType);
  const [listType, setListType] = useState<CustomModelType>(modelType);
  const [sourcePath, setSourcePath] = useState("");
  const [inspection, setInspection] = useState<CustomModelInspectResponse | null>(null);
  const [friendlyName, setFriendlyName] = useState("");
  const [env, setEnv] = useState("");
  const [modelFname, setModelFname] = useState("");
  const [description, setDescription] = useState("");
  const [developer, setDeveloper] = useState("");
  const [infoUrl, setInfoUrl] = useState("");
  const [backend, setBackend] = useState<CustomDetectorBackend | "">("");
  const [classNamesText, setClassNamesText] = useState("");
  const [datasetFile, setDatasetFile] = useState("");
  const [detectorClass, setDetectorClass] = useState("");
  const [detectorVariant, setDetectorVariant] = useState("");
  const [detectorConfig, setDetectorConfig] = useState("");
  const [region, setRegion] = useState("");
  const [fullImage, setFullImage] = useState(false);
  const [editing, setEditing] = useState<CustomModelInfo | null>(null);
  const [editDraft, setEditDraft] = useState<CustomModelUpdateRequest>({});
  const [errorText, setErrorText] = useState<string | null>(null);

  const { data, isLoading, isError, error } = useQuery({
    queryKey: ["custom-models"],
    queryFn: modelsApi.listCustomModels,
    enabled: open,
  });
  const environments = inspection?.environments ?? data?.environments ?? [];

  useEffect(() => {
    inspectGeneration.current += 1;
    if (!open) return;
    setType("");
    setListType(modelType);
    setEditing(null);
    setErrorText(null);
    setInspection(null);
    setSourcePath("");
  }, [open, modelType]);

  const refreshModelLists = async () => {
    await Promise.all([
      queryClient.invalidateQueries({ queryKey: ["custom-models"] }),
      queryClient.invalidateQueries({ queryKey: ["models"] }),
    ]);
  };

  const createMutation = useMutation({
    mutationFn: modelsApi.createCustomModel,
    onSuccess: async (model) => {
      await refreshModelLists();
      toast.success(
        onAnyCreated
          ? `${model.friendly_name} added and selected. Save project settings to keep this choice.`
          : `${model.friendly_name} added`,
      );
      setErrorText(null);
      resetCreateForm();
      if (model.type === modelType) onCreated?.(model);
      onAnyCreated?.(model);
      onOpenChange(false);
    },
    onError: (error) => setErrorText(getError(error)),
  });

  const updateMutation = useMutation({
    mutationFn: ({ id, changes }: { id: string; changes: CustomModelUpdateRequest }) =>
      modelsApi.updateCustomModel(id, changes),
    onSuccess: async () => {
      await refreshModelLists();
      toast.success("Model details updated");
      setEditing(null);
      setErrorText(null);
    },
    onError: (error) => setErrorText(getError(error)),
  });

  const deleteMutation = useMutation({
    mutationFn: modelsApi.deleteCustomModel,
    onSuccess: async () => {
      await refreshModelLists();
      toast.success("Model removed");
      setErrorText(null);
    },
    onError: (error) => setErrorText(getError(error)),
  });

  function resetCreateForm() {
    setSourcePath("");
    setInspection(null);
    setType("");
    setFriendlyName("");
    setEnv("");
    setModelFname("");
    setDescription("");
    setDeveloper("");
    setInfoUrl("");
    setClassNamesText("");
    setDatasetFile("");
    setDetectorConfig("");
    setDetectorClass("");
    setDetectorVariant("");
    setRegion("");
    setFullImage(false);
    setBackend("");
  }

  function clearInferredValues() {
    inspectGeneration.current += 1;
    setInspection(null);
    setType("");
    setFriendlyName("");
    setEnv("");
    setModelFname("");
    setBackend("");
    setClassNamesText("");
    setDatasetFile("");
    setDetectorClass("");
    setDetectorVariant("");
    setDetectorConfig("");
    setDescription("");
    setDeveloper("");
    setInfoUrl("");
    setRegion("");
    setFullImage(false);
    setErrorText(null);
  }

  function updateSourcePath(value: string) {
    clearInferredValues();
    setSourcePath(value);
  }

  function applyInspection(result: CustomModelInspectResponse) {
    setInspection(result);
    setSourcePath(result.source_path);
    const folderName = result.source_path.replace(/[\\/]+$/, "").split(/[\\/]/).pop() ?? "";
    setFriendlyName(folderName);
    setType(result.suggested_type ?? "");
    setEnv(result.suggested_env ?? "");
    setModelFname(result.suggested_model_fname ?? "");
    setBackend(result.suggested_detector_backend ?? "");
    setDetectorClass(result.suggested_detector_model_class ?? "");
    setDetectorVariant(result.suggested_detector_model_variant ?? "");
    setDetectorConfig(result.suggested_detector_config_fname ?? "");
    setDatasetFile(result.suggested_class_names_source ?? "");
    setClassNamesText(
      result.suggested_class_names
        ? JSON.stringify(result.suggested_class_names, null, 2)
        : "",
    );
    setErrorText(null);
  }

  const inspectMutation = useMutation({
    mutationFn: ({ path }: { path: string; generation: number }) => modelsApi.inspectCustomModel(path),
    onSuccess: (result, variables) => {
      if (
        !open ||
        inspectGeneration.current !== variables.generation ||
        sourcePath.trim().toLowerCase() !== variables.path.toLowerCase()
      ) return;
      applyInspection(result);
    },
    onError: (error, variables) => {
      if (!open || inspectGeneration.current !== variables.generation) return;
      setInspection(null);
      setErrorText(getError(error));
    },
  });

  async function chooseFolder() {
    if (!window.electronAPI?.selectFolder) return;
    try {
      const path = await window.electronAPI.selectFolder();
      if (path) updateSourcePath(path);
    } catch (error) {
      setErrorText(getError(error));
    }
  }

  function createModel(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    event.stopPropagation();
    setErrorText(null);
    if (!inspection || !type || !sourcePath.trim() || !friendlyName.trim() || !env) {
      setErrorText("Inspect a model folder, then complete the unresolved required fields.");
      return;
    }
    if (inspection.source_path.toLowerCase() !== sourcePath.trim().toLowerCase()) {
      setErrorText("Inspect this folder again before registering it.");
      return;
    }
    if (!modelFname.trim()) {
      setErrorText("Choose a model weight file.");
      return;
    }
    if (type === "classification" && !inspection.classifier_inference_compatible) {
      setErrorText("This folder does not contain a compatible classification inference.py.");
      return;
    }
    let classNames: Record<string, string> | string[] | undefined;
    if (type === "detection" && classNamesText.trim()) {
      try {
        const parsed: unknown = JSON.parse(classNamesText);
        if (
          !Array.isArray(parsed) &&
          (parsed === null || typeof parsed !== "object" ||
            !Object.values(parsed).every((value) => typeof value === "string"))
        ) {
          throw new Error("Class labels must be a JSON list or string map.");
        }
        if (Array.isArray(parsed) && !parsed.every((value) => typeof value === "string" && value.trim())) {
          throw new Error("Class labels must be non-empty strings.");
        }
        if (!Array.isArray(parsed) && Object.values(parsed).some((value) => !value.trim())) {
          throw new Error("Class labels must be non-empty strings.");
        }
        classNames = parsed as Record<string, string> | string[];
      } catch (error) {
        setErrorText(getError(error) === "Unexpected end of JSON input" ? "Class labels are not valid JSON." : getError(error));
        return;
      }
    }
    const payload: CustomModelCreateRequest = {
      type,
      source_path: sourcePath.trim(),
      friendly_name: friendlyName.trim(),
      env,
      model_fname: modelFname.trim() || undefined,
      description,
      developer,
      info_url: infoUrl,
    };
    if (type === "classification") {
      payload.region = (region || undefined) as ModelInfo["region"];
      payload.full_image_cls = fullImage;
    } else {
      if (!backend) {
        setErrorText("Choose the detector backend.");
        return;
      }
      payload.detector_backend = backend;
      payload.class_names = classNames;
      if (backend === "rfdetr") {
        if (!detectorClass) {
          setErrorText("Choose the RF-DETR model class.");
          return;
        }
        payload.detector_model_class = detectorClass;
      }
      if (backend === "rtdetr") {
        if (!detectorVariant) {
          setErrorText("Choose the RT-DETR variant.");
          return;
        }
        payload.detector_model_variant = detectorVariant;
      }
      if (backend === "rtdetrv2") {
        if (!detectorConfig.trim()) {
          setErrorText("Choose the RT-DETRv2 YAML config.");
          return;
        }
        payload.detector_config_fname = detectorConfig.trim();
      }
    }
    createMutation.mutate(payload);
  }

  function beginEdit(model: CustomModelInfo) {
    setEditing(model);
    setEditDraft({
      friendly_name: model.friendly_name,
      description: model.description,
      description_short: model.description_short ?? "",
      developer: model.developer ?? "",
      owner: model.owner ?? "",
      citation: model.citation ?? "",
      license: model.license ?? "",
      info_url: model.info_url ?? "",
      emoji: model.emoji ?? "",
      region: model.region ?? null,
      example_image_url: model.example_image_url ?? "",
    });
    setErrorText(null);
  }

  function saveEdit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    event.stopPropagation();
    if (!editing) return;
    updateMutation.mutate({ id: editing.model_id, changes: editDraft });
  }

  const matchingModels = useMemo(
    () => (data?.models ?? []).filter((model) => model.type === listType),
    [data?.models, listType],
  );

  function selectDataset(path: string) {
    setDatasetFile(path);
    const candidate = inspection?.dataset_candidates.find((row) => row.path === path);
    setClassNamesText(candidate ? JSON.stringify(candidate.class_names, null, 2) : "");
  }

  function inspectFolder() {
    const selectedPath = sourcePath.trim();
    if (!selectedPath) {
      setErrorText("Choose a model folder first.");
      return;
    }
    setErrorText(null);
    const generation = ++inspectGeneration.current;
    inspectMutation.mutate({ path: selectedPath, generation });
  }

  const hasAmbiguousLabels = Boolean(inspection && inspection.dataset_candidates.length > 1 &&
    new Set(inspection.dataset_candidates.map((candidate) => JSON.stringify(candidate.class_names))).size > 1);
  const registeredModelsAvailable = (data?.models.length ?? 0) > 0;
  const classNamesCount = countClassNames(classNamesText);
  const hasInvalidClassNames = type === "detection" && classNamesText.trim().length > 0 && classNamesCount === 0;
  const needsInput = inspection
    ? inspection.type_candidates.length === 0
      ? ["This folder does not match a supported model pack. Choose a folder with model weights, and for classification include an AddaxAI-compatible inference.py."]
      : [
          ...(!type ? ["Choose whether this is a detection or classification model."] : []),
          ...(type === "classification" && !inspection.classifier_inference_compatible
            ? ["Add an AddaxAI-compatible inference.py to register this as a classification model."]
            : []),
          ...(!modelFname.trim()
            ? [inspection.weights.length ? "Choose a model weight file." : "Add a supported model weight file."]
            : []),
          ...(!env ? ["Choose an inference environment."] : []),
          ...(type === "detection" && !backend ? ["Choose the detection backend."] : []),
          ...(type === "detection" && backend === "rfdetr" && !detectorClass
            ? ["Choose the RF-DETR model class."]
            : []),
          ...(type === "detection" && backend === "rtdetr" && !detectorVariant
            ? ["Choose the RT-DETR model variant."]
            : []),
          ...(type === "detection" && backend === "rtdetrv2" && !detectorConfig.trim()
            ? ["Choose the RT-DETRv2 YAML config."]
            : []),
          ...(type === "detection" && hasAmbiguousLabels && !classNamesText.trim()
            ? ["Choose the dataset YAML that contains this model's class labels, or enter labels in Advanced settings."]
            : []),
          ...(hasInvalidClassNames ? ["Fix the class-label JSON in Advanced settings."] : []),
        ]
    : [];
  const canRegister = Boolean(
    inspection &&
    inspection.source_path.toLowerCase() === sourcePath.trim().toLowerCase() &&
    type && inspection.type_candidates.includes(type) &&
    friendlyName.trim() && env && environments.includes(env) && modelFname.trim() &&
    (type !== "classification" || inspection.classifier_inference_compatible) &&
    (type !== "detection" || backend) &&
    (type !== "detection" || backend !== "rfdetr" || detectorClass) &&
    (type !== "detection" || backend !== "rtdetr" || detectorVariant) &&
    (type !== "detection" || backend !== "rtdetrv2" || detectorConfig.trim()) &&
    !hasInvalidClassNames &&
    (!hasAmbiguousLabels || classNamesText.trim()),
  );

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-h-[90vh] max-w-3xl overflow-y-auto">
        <DialogHeader>
          <DialogTitle>Manage custom models</DialogTitle>
          <DialogDescription>
            Add a local model pack or change its display details. Weights and inference settings stay fixed after registration.
          </DialogDescription>
        </DialogHeader>

        <div className={registeredModelsAvailable ? "grid gap-6 md:grid-cols-[minmax(0,1fr)_minmax(0,1.1fr)]" : "space-y-6"}>
          {registeredModelsAvailable ? <section className="space-y-3">
            <h3 className="text-sm font-semibold">Registered models</h3>
            <Field label="Show registered">
              <Select value={listType} onValueChange={(value) => setListType(value as CustomModelType)}>
                <SelectTrigger><SelectValue /></SelectTrigger><SelectContent>
                  <SelectItem value="detection">Detection</SelectItem>
                  <SelectItem value="classification">Classification</SelectItem>
                </SelectContent>
              </Select>
            </Field>
            {isError ? <p role="alert" className="text-sm text-destructive">Could not refresh custom models: {getError(error)}</p> : null}
            {matchingModels.length === 0 ? <p className="text-sm text-muted-foreground">No custom {listType} models registered.</p> : null}
            <div className="space-y-2">
              {matchingModels.map((model) => (
                <div key={model.model_id} className="flex items-center justify-between gap-2 rounded-md border p-3">
                  <div className="min-w-0">
                    <p className="truncate text-sm font-medium">{model.emoji} {model.friendly_name}</p>
                    <p className="truncate text-xs text-muted-foreground">{model.model_id} · {model.detector_backend ? formatDetectorBackend(model.detector_backend) : formatEnvironment(model.env)}</p>
                  </div>
                  <div className="flex shrink-0 gap-1">
                    <Button type="button" variant="outline" size="icon" aria-label={`Edit ${model.friendly_name}`} onClick={() => beginEdit(model)}>
                      <Pencil />
                    </Button>
                    <Button type="button" variant="destructive" size="icon" aria-label={`Delete ${model.friendly_name}`} disabled={deleteMutation.isPending} onClick={() => {
                      if (window.confirm(`Remove ${model.friendly_name}?`)) deleteMutation.mutate(model.model_id);
                    }}>
                      <Trash2 />
                    </Button>
                  </div>
                </div>
              ))}
            </div>
          </section> : null}

          <section className="space-y-3">
            <h3 className="text-sm font-semibold">{editing ? `Edit ${editing.friendly_name}` : "Register a model pack"}</h3>
            {isError && !registeredModelsAvailable ? (
              <p role="alert" className="text-sm text-destructive">Could not load custom models: {getError(error)}</p>
            ) : null}
            {!isLoading && !isError && !registeredModelsAvailable ? (
              <p className="text-sm text-muted-foreground">No custom models registered yet.</p>
            ) : null}
            {editing ? (
              <form className="space-y-3" onSubmit={saveEdit}>
                <Field label="Name"><Input required value={editDraft.friendly_name ?? ""} onChange={(e) => setEditDraft({ ...editDraft, friendly_name: e.target.value })} /></Field>
                <Field label="Description"><Textarea value={editDraft.description ?? ""} onChange={(e) => setEditDraft({ ...editDraft, description: e.target.value })} /></Field>
                <Field label="Short description"><Input value={editDraft.description_short ?? ""} onChange={(e) => setEditDraft({ ...editDraft, description_short: e.target.value })} /></Field>
                <Field label="Developer"><Input value={editDraft.developer ?? ""} onChange={(e) => setEditDraft({ ...editDraft, developer: e.target.value })} /></Field>
                <Field label="Owner"><Input value={editDraft.owner ?? ""} onChange={(e) => setEditDraft({ ...editDraft, owner: e.target.value })} /></Field>
                <Field label="Citation"><Input value={editDraft.citation ?? ""} onChange={(e) => setEditDraft({ ...editDraft, citation: e.target.value })} /></Field>
                <Field label="License"><Input value={editDraft.license ?? ""} onChange={(e) => setEditDraft({ ...editDraft, license: e.target.value })} /></Field>
                <Field label="Information URL"><Input value={editDraft.info_url ?? ""} onChange={(e) => setEditDraft({ ...editDraft, info_url: e.target.value })} /></Field>
                <Field label="Emoji"><Input value={editDraft.emoji ?? ""} onChange={(e) => setEditDraft({ ...editDraft, emoji: e.target.value })} /></Field>
                {editing.type === "classification" ? (
                  <Field label="Region">
                    <Select value={editDraft.region ?? "none"} onValueChange={(value) => setEditDraft({ ...editDraft, region: value === "none" ? null : value as ModelInfo["region"] })}>
                      <SelectTrigger><SelectValue /></SelectTrigger><SelectContent>
                        <SelectItem value="none">Unspecified</SelectItem>
                        {REGION_OPTIONS.map((value) => <SelectItem key={value} value={value}>{value}</SelectItem>)}
                      </SelectContent>
                    </Select>
                  </Field>
                ) : null}
                <Field label="Example image URL"><Input value={editDraft.example_image_url ?? ""} onChange={(e) => setEditDraft({ ...editDraft, example_image_url: e.target.value })} /></Field>
                <div className="flex gap-2">
                  <Button type="submit" disabled={updateMutation.isPending}>Save details</Button>
                  <Button type="button" variant="outline" onClick={() => setEditing(null)}>Cancel</Button>
                </div>
              </form>
            ) : (
              <form className="space-y-3" onSubmit={createModel}>
                <Field label="Model folder">
                  <div className="flex gap-2">
                    <Input
                      className="min-w-0 flex-1"
                      required
                      value={sourcePath}
                      title={sourcePath || undefined}
                      placeholder="Choose or enter the local model folder"
                      disabled={inspectMutation.isPending || createMutation.isPending}
                      onChange={(event) => updateSourcePath(event.target.value)}
                    />
                    {window.electronAPI?.selectFolder ? (
                      <Button
                        type="button"
                        variant="outline"
                        size="icon"
                        aria-label="Choose model folder"
                        disabled={inspectMutation.isPending || createMutation.isPending}
                        onClick={chooseFolder}
                      ><FolderOpen /></Button>
                    ) : null}
                    <Button
                      type="button"
                      variant="outline"
                      disabled={!sourcePath.trim() || inspectMutation.isPending}
                      onClick={inspectFolder}
                    >{inspectMutation.isPending ? "Inspecting…" : "Inspect folder"}</Button>
                  </div>
                  {!window.electronAPI?.selectFolder ? (
                    <p className="mt-1 text-xs text-muted-foreground">
                      Paste the full folder path from Windows Explorer on this PC.
                    </p>
                  ) : null}
                </Field>
                {errorText ? <p role="alert" className="text-sm text-destructive">{errorText}</p> : null}
                {inspection ? (
                  <div className="space-y-4 rounded-md border p-3">
                    <div>
                      <p className="font-medium">
                        {inspection.suggested_type
                          ? `Looks like a ${formatModelType(inspection.suggested_type)} model pack`
                          : "Choose the model type from the inspection results"}
                      </p>
                      <p className="text-sm text-muted-foreground">
                        {inspection.total_file_count} files · {formatBytes(inspection.total_size_bytes)} will be copied
                      </p>
                      {type === "detection" ? (
                        <p className="text-sm text-muted-foreground">
                          {classNamesCount > 0
                            ? `${classNamesCount} class labels found${datasetFile ? ` in ${datasetFile}` : ""}.`
                            : "No class labels found. If the checkpoint does not include them, add labels in Advanced settings."}
                        </p>
                      ) : null}
                    </div>

                    {inspection.type_candidates.length === 0 ? (
                      <p role="alert" className="text-sm text-destructive">
                        This folder has no supported model structure. Choose another folder with a supported weight file; classification packs also need an AddaxAI-compatible inference.py.
                      </p>
                    ) : null}

                    {type && type !== modelType ? (
                      <p className="rounded-md bg-muted p-3 text-sm">
                        {onAnyCreated
                          ? `This is a ${formatModelType(type)} model. It will be selected in the ${formatModelType(type)} setting. Save project settings to keep this selection.`
                          : `This is a ${formatModelType(type)} model. After registration, choose it under the ${formatModelType(type)} model setting.`}
                      </p>
                    ) : null}
                    {type && onAnyCreated && type === modelType ? (
                      <p className="text-sm text-muted-foreground">
                        After registration it will be selected in the {formatModelType(type)} setting. Save project settings to keep this choice.
                      </p>
                    ) : null}
                    {needsInput.length > 0 || inspection.warnings.length > 0 ? (
                      <div className="rounded-md border border-amber-500/50 bg-amber-50 p-3 text-sm text-amber-950">
                        {needsInput.length > 0 ? (
                          <>
                            <p className="font-medium">Needs your input</p>
                            <ul className="mt-1 list-disc space-y-1 pl-5">
                              {needsInput.map((item) => <li key={item}>{item}</li>)}
                            </ul>
                          </>
                        ) : null}
                        {inspection.warnings.length > 0 ? (
                          <>
                            <p className={needsInput.length > 0 ? "mt-2 font-medium" : "font-medium"}>Inspection notes</p>
                            <ul className="mt-1 list-disc space-y-1 pl-5">
                              {inspection.warnings.map((warning) => <li key={warning}>{warning}</li>)}
                            </ul>
                          </>
                        ) : null}
                      </div>
                    ) : null}

                    {inspection.type_candidates.length > 1 ? (
                      <Field label="Model type">
                        <Select value={type || "choose-type"} onValueChange={(value) => {
                          const nextType = value === "choose-type" ? "" : value as CustomModelType;
                          setType(nextType);
                          if (inspection.suggested_env_source === "manifest") {
                            setEnv(inspection.suggested_env ?? "");
                          } else if (
                            inspection.suggested_env_source === "rtdetrv2_config" &&
                            nextType === "detection"
                          ) {
                            setEnv(inspection.suggested_env ?? "");
                          } else {
                            setEnv("");
                          }
                        }}>
                          <SelectTrigger><SelectValue placeholder="Choose model type" /></SelectTrigger><SelectContent>
                            <SelectItem value="choose-type" disabled>Choose model type</SelectItem>
                            {inspection.type_candidates.includes("detection") || inspection.type_candidates.length === 0 ? (
                              <SelectItem value="detection">Detection</SelectItem>
                            ) : null}
                            {inspection.type_candidates.includes("classification") || inspection.type_candidates.length === 0 ? (
                              <SelectItem value="classification">Classification</SelectItem>
                            ) : null}
                          </SelectContent>
                        </Select>
                      </Field>
                    ) : null}
                    {type === "classification" && !inspection.classifier_inference_compatible ? (
                      <p role="alert" className="text-sm text-destructive">
                        This folder does not have an AddaxAI-compatible inference.py, so it cannot be registered as a classifier.
                      </p>
                    ) : null}

                    <Field label="Model name">
                      <Input required value={friendlyName} onChange={(event) => setFriendlyName(event.target.value)} />
                    </Field>
                    {inspection.weights.length > 1 ? (
                      <Field label="Weights file">
                        <Select value={modelFname || "choose-weight"} onValueChange={(value) => setModelFname(value === "choose-weight" ? "" : value)}>
                          <SelectTrigger><SelectValue placeholder="Choose weights" /></SelectTrigger><SelectContent>
                            <SelectItem value="choose-weight" disabled>Choose weights</SelectItem>
                            {inspection.weights.map((value) => <SelectItem key={value} value={value}>{value}</SelectItem>)}
                          </SelectContent>
                        </Select>
                      </Field>
                    ) : (
                      <Field label="Weights file">
                        <Input
                          required
                          value={modelFname}
                          readOnly={inspection.weights.length === 1}
                          placeholder="Relative path to a supported weight file"
                          onChange={(event) => setModelFname(event.target.value)}
                        />
                      </Field>
                    )}
                    <Field label="Inference environment">
                      <Select value={env || "choose-env"} onValueChange={(value) => setEnv(value === "choose-env" ? "" : value)}>
                        <SelectTrigger><SelectValue placeholder="Choose an environment" /></SelectTrigger><SelectContent>
                          <SelectItem value="choose-env" disabled>Choose an environment</SelectItem>
                          {environments.map((value) => <SelectItem key={value} value={value}>{formatEnvironment(value)}</SelectItem>)}
                        </SelectContent>
                      </Select>
                    </Field>

                    {type === "detection" ? (
                      <>
                        <Field label="Detection backend">
                          <Select value={backend || "choose-backend"} onValueChange={(value) => {
                            const nextBackend = value === "choose-backend" ? "" : value as CustomDetectorBackend;
                            setBackend(nextBackend);
                            setDetectorClass("");
                            setDetectorVariant("");
                            if (value !== "rtdetrv2") setDetectorConfig("");
                            if (inspection.suggested_env_source === "rtdetrv2_config") {
                              setEnv(nextBackend === "rtdetrv2" ? inspection.suggested_env ?? "" : "");
                            }
                          }}>
                            <SelectTrigger><SelectValue placeholder="Choose a backend" /></SelectTrigger><SelectContent>
                              <SelectItem value="choose-backend" disabled>Choose a backend</SelectItem>
                              {(["yolo", "rfdetr", "rtdetr", "rtdetrv2"] as const).map((value) => (
                                <SelectItem key={value} value={value}>{formatDetectorBackend(value)}</SelectItem>
                              ))}
                            </SelectContent>
                          </Select>
                          {inspection.suggested_detector_backend ? (
                            <p className="mt-1 text-xs text-muted-foreground">Detected from pack contents: {formatDetectorBackend(inspection.suggested_detector_backend)}.</p>
                          ) : null}
                        </Field>
                        {backend === "rfdetr" ? (
                          <Field label="RF-DETR model class">
                            <Select value={detectorClass || "choose-class"} onValueChange={(value) => setDetectorClass(value === "choose-class" ? "" : value)}>
                              <SelectTrigger><SelectValue placeholder="Choose model class" /></SelectTrigger><SelectContent>
                                <SelectItem value="choose-class" disabled>Choose model class</SelectItem>
                                {RFDETR_CLASSES.map((value) => <SelectItem key={value} value={value}>{value}</SelectItem>)}
                              </SelectContent>
                            </Select>
                          </Field>
                        ) : null}
                        {backend === "rtdetr" ? (
                          <Field label="RT-DETR variant">
                            <Select value={detectorVariant || "choose-variant"} onValueChange={(value) => setDetectorVariant(value === "choose-variant" ? "" : value)}>
                              <SelectTrigger><SelectValue placeholder="Choose variant" /></SelectTrigger><SelectContent>
                                <SelectItem value="choose-variant" disabled>Choose variant</SelectItem>
                                <SelectItem value="MDV6-apa-rtdetr-c">MDV6-apa-rtdetr-c</SelectItem>
                                <SelectItem value="MDV6-apa-rtdetr-e">MDV6-apa-rtdetr-e</SelectItem>
                              </SelectContent>
                            </Select>
                          </Field>
                        ) : null}
                        {backend === "rtdetrv2" ? (
                          inspection.detector_config_candidates.length > 0 ? (
                            <Field label="RT-DETRv2 YAML config">
                              <Select value={detectorConfig || "choose-config"} onValueChange={(value) => setDetectorConfig(value === "choose-config" ? "" : value)}>
                                <SelectTrigger><SelectValue placeholder="Choose a config" /></SelectTrigger><SelectContent>
                                  <SelectItem value="choose-config" disabled>Choose a config</SelectItem>
                                  {inspection.detector_config_candidates.map((value) => <SelectItem key={value} value={value}>{value}</SelectItem>)}
                                </SelectContent>
                              </Select>
                            </Field>
                          ) : (
                            <Field label="RT-DETRv2 YAML config (relative path)"><Input required value={detectorConfig} onChange={(event) => setDetectorConfig(event.target.value)} /></Field>
                          )
                        ) : null}
                        {inspection.dataset_candidates.length > 1 && hasAmbiguousLabels ? (
                          <Field label="Class labels source">
                            <Select value={datasetFile || "choose-label-source"} onValueChange={(value) => selectDataset(value === "choose-label-source" ? "" : value)}>
                              <SelectTrigger><SelectValue placeholder="Choose dataset labels" /></SelectTrigger><SelectContent>
                                <SelectItem value="choose-label-source" disabled>Choose dataset labels</SelectItem>
                                {inspection.dataset_candidates.map((candidate) => <SelectItem key={candidate.path} value={candidate.path}>{candidate.path}</SelectItem>)}
                              </SelectContent>
                            </Select>
                          </Field>
                        ) : null}
                      </>
                    ) : null}

                    {type === "classification" ? (
                      <p className="text-sm text-muted-foreground">
                        Compatible AddaxAI inference.py found. Classification packs use the existing packaged environments.
                      </p>
                    ) : null}

                    <details className="rounded-md border p-3">
                      <summary className="cursor-pointer text-sm font-medium">Advanced settings</summary>
                      <div className="mt-3 space-y-3">
                        {type === "detection" ? (
                          <Field label="Class labels (JSON; optional if the checkpoint contains labels)">
                            <Textarea
                              value={classNamesText}
                              placeholder={'{"0":"animal","1":"person"}'}
                              onChange={(event) => setClassNamesText(event.target.value)}
                            />
                          </Field>
                        ) : null}
                        {type === "classification" ? (
                          <>
                            <Field label="Region">
                              <Select value={region || "none"} onValueChange={(value) => setRegion(value === "none" ? "" : value)}>
                                <SelectTrigger><SelectValue /></SelectTrigger><SelectContent>
                                  <SelectItem value="none">Unspecified</SelectItem>
                                  {REGION_OPTIONS.map((value) => <SelectItem key={value} value={value}>{value}</SelectItem>)}
                                </SelectContent>
                              </Select>
                            </Field>
                            <label className="flex items-center gap-2 text-sm">
                              <Checkbox checked={fullImage} onCheckedChange={(value) => setFullImage(value === true)} />
                              Classifies the full image without a detector
                            </label>
                          </>
                        ) : null}
                        <Field label="Description"><Textarea value={description} onChange={(event) => setDescription(event.target.value)} /></Field>
                        <Field label="Developer"><Input value={developer} onChange={(event) => setDeveloper(event.target.value)} /></Field>
                        <Field label="Information URL"><Input value={infoUrl} onChange={(event) => setInfoUrl(event.target.value)} /></Field>
                      </div>
                    </details>

                    <details className="rounded-md border p-3">
                      <summary className="cursor-pointer text-sm font-medium">
                        Files to be copied ({inspection.files.length}{inspection.files_truncated ? "+" : ""})
                      </summary>
                      <ul className="mt-2 max-h-40 space-y-1 overflow-y-auto text-xs text-muted-foreground">
                        {inspection.files.map((file) => (
                          <li key={file.path} className="flex justify-between gap-3">
                            <span className="break-all">{file.path}</span><span className="shrink-0">{formatBytes(file.size_bytes)}</span>
                          </li>
                        ))}
                      </ul>
                      {inspection.files_truncated ? <p className="mt-2 text-xs text-muted-foreground">More files will also be retained.</p> : null}
                    </details>

                  </div>
                ) : null}
                <DialogFooter>
                  <Button
                    type="submit"
                    disabled={createMutation.isPending || inspectMutation.isPending || !canRegister}
                  >
                    <Plus /> {type === modelType || onAnyCreated ? "Register and select" : "Register"}
                  </Button>
                </DialogFooter>
              </form>
            )}
            {errorText && editing ? <p role="alert" className="text-sm text-destructive">{errorText}</p> : null}
          </section>
        </div>
      </DialogContent>
    </Dialog>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="space-y-1.5">
      <Label>{label}</Label>
      <div>{children}</div>
    </div>
  );
}

function formatBytes(value: number): string {
  if (value < 1024) return `${value} B`;
  const units = ["KB", "MB", "GB", "TB"];
  let amount = value / 1024;
  let unit = 0;
  while (amount >= 1024 && unit < units.length - 1) {
    amount /= 1024;
    unit += 1;
  }
  return `${amount.toFixed(amount >= 10 ? 0 : 1)} ${units[unit]}`;
}

function formatModelType(value: CustomModelType): string {
  return value === "detection" ? "detection" : "classification";
}

function formatDetectorBackend(value: CustomDetectorBackend | "megadetector"): string {
  switch (value) {
    case "megadetector": return "MegaDetector";
    case "yolo": return "YOLO";
    case "rfdetr": return "RF-DETR";
    case "rtdetr": return "RT-DETR";
    case "rtdetrv2": return "RT-DETRv2";
  }
}

function formatEnvironment(value: string): string {
  if (value === "addaxai-base") return "AddaxAI base environment";
  if (value === "rtdetr") return "RT-DETR environment";
  return value.split(/[-_]/).map((part) => part ? part[0].toUpperCase() + part.slice(1) : part).join(" ");
}

function countClassNames(value: string): number {
  if (!value.trim()) return 0;
  try {
    const parsed: unknown = JSON.parse(value);
    if (Array.isArray(parsed)) {
      return parsed.every((item) => typeof item === "string" && item.trim()) ? parsed.length : 0;
    }
    if (parsed && typeof parsed === "object") {
      const labels = Object.values(parsed);
      return labels.length > 0 && labels.every((item) => typeof item === "string" && item.trim())
        ? labels.length
        : 0;
    }
  } catch {
    return 0;
  }
  return 0;
}

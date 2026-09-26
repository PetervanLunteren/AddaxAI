import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { FileUp, FolderOpen, Pencil, Plus, Trash2, X } from "lucide-react";
import { toast } from "sonner";

import { modelsApi, uploadCustomModelFile } from "@/api/models";
import type {
  CustomModelCreateRequest,
  CustomDetectorBackend,
  CustomModelInfo,
  CustomModelInspectResponse,
  CustomModelRegistrationRole,
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
const NEW_DETECTOR_BACKENDS = ["yolo", "rtdetrv2"] as const;

interface Props {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  modelType: CustomModelType;
  onCreated?: (model: CustomModelInfo, role?: CustomModelRegistrationRole) => void;
  onAnyCreated?: (model: CustomModelInfo, role?: CustomModelRegistrationRole) => void;
  canSelectBoth?: boolean;
}

function getError(error: unknown): string {
  return error instanceof Error ? error.message : "Model management failed";
}

function willSelectCreatedModel(
  type: CustomModelType | "",
  modelType: CustomModelType,
  role: CustomModelRegistrationRole | "",
  hasAnyCreated: boolean,
  canSelectBoth: boolean,
): boolean {
  return Boolean(
    hasAnyCreated ||
      (type &&
        ((type === modelType &&
          !(type === "classification" && role === "both" && !canSelectBoth)) ||
          (role === "both" && canSelectBoth))),
  );
}

export function CustomModelManagerDialog({
  open,
  onOpenChange,
  modelType,
  onCreated,
  onAnyCreated,
  canSelectBoth = false,
}: Props) {
  const queryClient = useQueryClient();
  const inspectGeneration = useRef(0);
  const uploadGeneration = useRef(0);
  const sourcePathRef = useRef("");
  const pendingWeightRef = useRef("");
  const stagedWeightFilenameRef = useRef("");
  const uploadIdRef = useRef<string | null>(null);
  const uploadAbortRef = useRef<AbortController | null>(null);
  const weightInputRef = useRef<HTMLInputElement>(null);
  const companionInputRef = useRef<HTMLInputElement>(null);
  const [type, setType] = useState<CustomModelType | "">(modelType);
  const [registrationRole, setRegistrationRole] = useState<CustomModelRegistrationRole | "">("");
  const [listType, setListType] = useState<CustomModelType>(modelType);
  const [sourcePath, setSourcePath] = useState("");
  const [uploadId, setUploadId] = useState<string | null>(null);
  const [uploadingName, setUploadingName] = useState("");
  const [uploadProgress, setUploadProgress] = useState<{ loaded: number; total: number } | null>(null);
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
  const [detectorConfig, setDetectorConfig] = useState("");
  const [detectorConfigTemplate, setDetectorConfigTemplate] = useState("");
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

  const cancelUploadSession = useCallback(() => {
    uploadGeneration.current += 1;
    uploadAbortRef.current?.abort();
    uploadAbortRef.current = null;
    const previousUploadId = uploadIdRef.current;
    uploadIdRef.current = null;
    stagedWeightFilenameRef.current = "";
    setUploadId(null);
    setUploadProgress(null);
    setUploadingName("");
    if (previousUploadId) {
      void modelsApi.cancelCustomModelUpload(previousUploadId).catch(() => undefined);
    }
  }, []);

  function clearUploadAfterCreate() {
    uploadAbortRef.current = null;
    uploadIdRef.current = null;
    setUploadId(null);
    setUploadProgress(null);
    setUploadingName("");
  }

  function handleDialogOpenChange(nextOpen: boolean) {
    if (!nextOpen) cancelUploadSession();
    onOpenChange(nextOpen);
  }

  useEffect(() => {
    inspectGeneration.current += 1;
    if (!open) {
      cancelUploadSession();
      return;
    }
    setType("");
    setRegistrationRole("");
    setListType(modelType);
    setEditing(null);
    setErrorText(null);
    setInspection(null);
    cancelUploadSession();
    setSourcePath("");
    sourcePathRef.current = "";
  }, [cancelUploadSession, open, modelType]);

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
      const selectedAfterCreate = willSelectCreatedModel(
        type,
        modelType,
        registrationRole,
        Boolean(onAnyCreated),
        canSelectBoth,
      );
      toast.success(
        selectedAfterCreate
          ? `${model.friendly_name} added and selected. Save this form to keep the choice.`
          : `${model.friendly_name} added`,
      );
      setErrorText(null);
      clearUploadAfterCreate();
      resetCreateForm();
      if (model.type === modelType || (registrationRole === "both" && modelType === "classification")) {
        onCreated?.(model, registrationRole || undefined);
      }
      onAnyCreated?.(model, registrationRole || undefined);
      onOpenChange(false);
    },
    onError: (error) => {
      if (!uploadIdRef.current) {
        setErrorText(getError(error));
        return;
      }
      cancelUploadSession();
      sourcePathRef.current = "";
      pendingWeightRef.current = "";
      setSourcePath("");
      setInspection(null);
      clearInferredValues();
      setErrorText(
        `${getError(error)} Uploaded files were discarded. Choose the weight file again and reselect any companion files before registering.`,
      );
    },
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
    sourcePathRef.current = "";
    pendingWeightRef.current = "";
    stagedWeightFilenameRef.current = "";
    setSourcePath("");
    setInspection(null);
    setType("");
    setRegistrationRole("");
    setFriendlyName("");
    setEnv("");
    setModelFname("");
    setDescription("");
    setDeveloper("");
    setInfoUrl("");
    setClassNamesText("");
    setDatasetFile("");
    setDetectorConfig("");
    setDetectorConfigTemplate("");
    setRegion("");
    setFullImage(false);
    setBackend("");
  }

  function clearInferredValues() {
    inspectGeneration.current += 1;
    stagedWeightFilenameRef.current = "";
    setInspection(null);
    setType("");
    setRegistrationRole("");
    setFriendlyName("");
    setEnv("");
    setModelFname("");
    setBackend("");
    setClassNamesText("");
    setDatasetFile("");
    setDetectorConfig("");
    setDetectorConfigTemplate("");
    setDescription("");
    setDeveloper("");
    setInfoUrl("");
    setRegion("");
    setFullImage(false);
    setErrorText(null);
  }

  function updateSourcePath(value: string) {
    if (uploadIdRef.current) cancelUploadSession();
    clearInferredValues();
    sourcePathRef.current = value;
    setSourcePath(value);
  }

  function applyInspection(result: CustomModelInspectResponse) {
    setInspection(result);
    sourcePathRef.current = result.source_path;
    setSourcePath(result.source_path);
    const folderName = result.source_path.replace(/[\\/]+$/, "").split(/[\\/]/).pop() ?? "";
    const stagedFilename = uploadIdRef.current ? stagedWeightFilenameRef.current : "";
    setFriendlyName(stagedFilename ? filenameStem(stagedFilename) : folderName);
    const initialRole = result.suggested_type ?? "";
    setType(initialRole);
    setRegistrationRole(initialRole);
    setEnv(result.suggested_env ?? "");
    setModelFname(result.suggested_model_fname ?? "");
    setBackend(
      result.suggested_detector_backend === "yolo" || result.suggested_detector_backend === "rtdetrv2"
        ? result.suggested_detector_backend
        : "",
    );
    setDetectorConfig(result.suggested_detector_config_fname ?? "");
    setDatasetFile(result.suggested_class_names_source ?? "");
    setDetectorConfigTemplate("");
    setClassNamesText(result.suggested_class_names ? classNamesToLines(result.suggested_class_names) : "");
    setErrorText(null);
  }

  const inspectMutation = useMutation({
    mutationFn: ({ path }: { path: string; generation: number }) => modelsApi.inspectCustomModel(path),
    onSuccess: (result, variables) => {
      if (
        !open ||
        inspectGeneration.current !== variables.generation ||
        sourcePathRef.current.trim().toLowerCase() !== variables.path.toLowerCase()
      ) return;
      applyInspection(result);
      if (pendingWeightRef.current && result.weights.includes(pendingWeightRef.current)) {
        setModelFname(pendingWeightRef.current);
      }
      pendingWeightRef.current = "";
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
    if (!inspection || !type || !registrationRole || !sourcePath.trim() || !friendlyName.trim() || !env) {
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
    if (!registrationRole) {
      setErrorText("Choose whether to register this pack for detection, classification, or both.");
      return;
    }
    let classNames: Record<string, string> | undefined;
    if (type === "detection" && classNamesText.trim()) {
      try {
        classNames = linesToClassNames(classNamesText, requiresExplicitClassIds);
      } catch (error) {
        setErrorText(getError(error));
        return;
      }
    }
    const payload: CustomModelCreateRequest = {
      type,
      ...(uploadIdRef.current
        ? { upload_id: uploadIdRef.current }
        : { source_path: sourcePath.trim() }),
      friendly_name: friendlyName.trim(),
      env,
      model_fname: modelFname.trim() || undefined,
      description,
      developer,
      info_url: infoUrl,
    };
    if (type === "detection") {
      payload.classification_uses_detection_classes = registrationRole === "both";
    }
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
      if (backend === "rtdetrv2") {
        if (!detectorConfig.trim() && !detectorConfigTemplate) {
          setErrorText("Choose an RT-DETRv2 YAML config or architecture template.");
          return;
        }
        if (detectorConfig.trim()) payload.detector_config_fname = detectorConfig.trim();
        if (detectorConfigTemplate) payload.detector_config_template = detectorConfigTemplate;
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
    setClassNamesText(candidate ? classNamesToLines(candidate.class_names) : "");
  }

  function inspectPath(path: string) {
    sourcePathRef.current = path;
    setSourcePath(path);
    setErrorText(null);
    const generation = ++inspectGeneration.current;
    inspectMutation.mutate({ path, generation });
  }

  async function uploadSelectedFile(file: File, isWeight: boolean): Promise<boolean> {
    if (isWeight || uploadAbortRef.current) cancelUploadSession();
    const generation = uploadGeneration.current;
    if (isWeight) {
      clearInferredValues();
      pendingWeightRef.current = file.name;
      stagedWeightFilenameRef.current = file.name;
      setFriendlyName(filenameStem(file.name));
    }
    setUploadingName(file.name);
    setUploadProgress({ loaded: 0, total: file.size });
    setErrorText(null);

    try {
      let sessionId = uploadIdRef.current;
      let stagingPath = sourcePathRef.current;
      if (!sessionId) {
        const session = await modelsApi.startCustomModelUpload();
        if (generation !== uploadGeneration.current || !open) {
          await modelsApi.cancelCustomModelUpload(session.upload_id).catch(() => undefined);
          return false;
        }
        sessionId = session.upload_id;
        stagingPath = session.source_path;
        uploadIdRef.current = sessionId;
        sourcePathRef.current = stagingPath;
        setUploadId(sessionId);
        setSourcePath(stagingPath);
      }
      const controller = new AbortController();
      uploadAbortRef.current = controller;
      await uploadCustomModelFile(sessionId, file, (loaded, total) => {
        setUploadProgress({ loaded, total });
      }, controller.signal);
      if (generation !== uploadGeneration.current || !open) return false;
      uploadAbortRef.current = null;
      setUploadProgress(null);
      setUploadingName("");
      inspectPath(stagingPath);
      return true;
    } catch (error) {
      if (error instanceof DOMException && error.name === "AbortError") return false;
      cancelUploadSession();
      setInspection(null);
      sourcePathRef.current = "";
      setSourcePath("");
      clearInferredValues();
      setErrorText(getError(error));
      return false;
    }
  }

  async function chooseWeightFile() {
    if (window.electronAPI?.openFile) {
      try {
        const path = await window.electronAPI.openFile({
          title: "Choose model weight file",
          filters: [{ name: "Model weights", extensions: ["pt", "pth", "ckpt", "h5", "hdf5", "keras", "onnx", "pb", "safetensors", "tflite"] }],
        });
        if (!path) return;
        cancelUploadSession();
        clearInferredValues();
        const normalized = path.replace(/\\/g, "/");
        const separator = normalized.lastIndexOf("/");
        pendingWeightRef.current = normalized.slice(separator + 1);
        inspectPath(separator > 0 ? normalized.slice(0, separator) : ".");
      } catch (error) {
        setErrorText(getError(error));
      }
      return;
    }
    weightInputRef.current?.click();
  }

  async function onWeightInputChange(event: React.ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (file) await uploadSelectedFile(file, true);
  }

  async function onCompanionInputChange(event: React.ChangeEvent<HTMLInputElement>) {
    const files = Array.from(event.target.files ?? []);
    event.target.value = "";
    if (!uploadIdRef.current || !sourcePathRef.current) return;
    for (const file of files) {
      const completed = await uploadSelectedFile(file, false);
      if (!completed) break;
    }
  }

  function inspectFolder() {
    const selectedPath = sourcePath.trim();
    if (!selectedPath) {
      setErrorText("Choose a model folder first.");
      return;
    }
    setErrorText(null);
    inspectPath(selectedPath);
  }

  const hasAmbiguousLabels = Boolean(inspection && inspection.dataset_candidates.length > 1 &&
    new Set(inspection.dataset_candidates.map((candidate) => JSON.stringify(candidate.class_names))).size > 1);
  const selectedDataset = inspection?.dataset_candidates.find((candidate) => candidate.path === datasetFile);
  const suggestedNames = selectedDataset?.class_names ?? inspection?.suggested_class_names ?? undefined;
  const requiresExplicitClassIds = Boolean(suggestedNames && !hasZeroBasedSequentialIds(suggestedNames));
  const hasNonNumericClassIds = Boolean(suggestedNames && Object.keys(suggestedNames).some((id) => !/^\d+$/.test(id)));
  const registeredModelsAvailable = (data?.models.length ?? 0) > 0;
  const classNamesCount = countClassNames(classNamesText, requiresExplicitClassIds);
  const hasInvalidClassNames = type === "detection" && classNamesText.trim().length > 0 && classNamesCount === 0;
  const needsInput = inspection
    ? inspection.type_candidates.length === 0
      ? ["This folder does not match a supported model pack. Choose a folder with model weights, and for classification include an AddaxAI-compatible inference.py."]
      : [
          ...(!registrationRole ? ["Choose whether to register this pack for detection, classification, or both."] : []),
          ...(type === "classification" && !inspection.classifier_inference_compatible
            ? ["Add an AddaxAI-compatible inference.py to register this as a classification model."]
            : []),
          ...(!modelFname.trim()
            ? [inspection.weights.length ? "Choose a model weight file." : "Add a supported model weight file."]
            : []),
          ...(!env ? [type === "classification" ? "Choose the classifier environment in Advanced settings." : "Choose a supported detection backend to select its packaged environment."] : []),
          ...(type === "detection" && !backend ? ["Choose the detection backend."] : []),
          ...(type === "detection" && backend === "rtdetrv2" && !detectorConfig.trim() && !detectorConfigTemplate
            ? ["Choose an RT-DETRv2 YAML config or architecture template."]
            : []),
          ...(type === "detection" && hasAmbiguousLabels && !classNamesText.trim()
            ? ["Choose the dataset YAML that contains this model's class labels, or enter one class name per line."]
            : []),
          ...(type === "detection" && requiresExplicitClassIds && !hasNonNumericClassIds && !classNamesText.trim()
            ? ["Keep the existing numeric class IDs by entering each line as ID: name."]
            : []),
          ...(type === "detection" && hasNonNumericClassIds
            ? ["This pack has non-numeric class IDs; AddaxAI detector packs require numeric class IDs."]
            : []),
          ...(type === "detection" && !classNamesText.trim() && !hasAmbiguousLabels
            ? ["Enter one class name per line so AddaxAI can reuse the detector's labels as classifications."]
            : []),
          ...(hasInvalidClassNames
            ? [requiresExplicitClassIds
                ? "Use unique class names and canonical IDs (0, 1, ...), written as ID: name."
                : "Enter non-empty, unique class names, one per line."]
            : []),
        ]
    : [];
  const canRegister = Boolean(
    inspection &&
    inspection.source_path.toLowerCase() === sourcePath.trim().toLowerCase() &&
    type && registrationRole && inspection.type_candidates.includes(type) &&
    friendlyName.trim() && env && environments.includes(env) && modelFname.trim() &&
    (type !== "classification" || inspection.classifier_inference_compatible) &&
    (type !== "detection" || backend) &&
    (type !== "detection" || backend !== "rtdetrv2" || detectorConfig.trim() || detectorConfigTemplate) &&
    (type !== "detection" || classNamesCount > 0) &&
    !hasInvalidClassNames &&
    (!hasAmbiguousLabels || classNamesText.trim()),
  );

  return (
    <Dialog open={open} onOpenChange={handleDialogOpenChange}>
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
            {listType === "classification" ? (
              <p className="rounded-md border bg-muted/40 p-3 text-xs text-muted-foreground">
                A custom Detection model with class labels can also be selected under Classification in Project settings. Select the same model ID to reuse its detected class and confidence; no duplicate registration or extra inference is needed.
              </p>
            ) : null}
            {isError ? <p role="alert" className="text-sm text-destructive">Could not refresh custom models: {getError(error)}</p> : null}
            {matchingModels.length === 0 ? <p className="text-sm text-muted-foreground">No custom {listType} models registered.</p> : null}
            <div className="space-y-2">
              {matchingModels.map((model) => (
                <div key={model.model_id} className="flex items-center justify-between gap-2 rounded-md border p-3">
                  <div className="min-w-0">
                    <p className="truncate text-sm font-medium">{model.emoji} {model.friendly_name}</p>
                    <p className="truncate text-xs text-muted-foreground">{model.model_id} · {model.detector_backend ? formatDetectorBackend(model.detector_backend) : formatEnvironment(model.env)}</p>
                    {model.classification_uses_detection_classes ? (
                      <p className="text-xs text-muted-foreground">Also selectable under Classification with the same ID; detector classes and confidence are reused without another inference.</p>
                    ) : null}
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
                <input
                  ref={weightInputRef}
                  type="file"
                  accept=".pt,.pth,.ckpt,.h5,.hdf5,.keras,.onnx,.pb,.safetensors,.tflite"
                  className="hidden"
                  onChange={onWeightInputChange}
                />
                <input
                  ref={companionInputRef}
                  type="file"
                  multiple
                  className="hidden"
                  onChange={onCompanionInputChange}
                />
                <div className="rounded-md border border-primary/30 bg-primary/5 p-4">
                  <p className="font-medium">Start by choosing the model weight file</p>
                  <p className="mt-1 text-sm text-muted-foreground">
                    {window.electronAPI?.openFile
                      ? "AddaxAI will inspect the selected file's folder. The files listed below will be copied into its managed model folder."
                      : "Choose a weight file to stream it to this PC. Browsers cannot read neighboring files, so add any inference.py, dataset YAML, or model config with Add companion files."}
                  </p>
                  <div className="mt-3 flex flex-wrap gap-2">
                    <Button
                      type="button"
                      disabled={Boolean(uploadProgress) || inspectMutation.isPending || createMutation.isPending}
                      onClick={chooseWeightFile}
                    ><FileUp /> Choose weight file</Button>
                    {uploadId ? (
                      <Button type="button" variant="outline" disabled={Boolean(uploadProgress) || inspectMutation.isPending} onClick={() => companionInputRef.current?.click()}>
                        Add companion files
                      </Button>
                    ) : null}
                    {uploadId ? (
                      <Button type="button" variant="ghost" disabled={createMutation.isPending} onClick={() => {
                        cancelUploadSession();
                        clearInferredValues();
                        sourcePathRef.current = "";
                        setSourcePath("");
                      }}><X /> Cancel selected files</Button>
                    ) : null}
                  </div>
                  {uploadId && !uploadProgress ? (
                    <p className="mt-2 text-sm text-muted-foreground">Weight and selected companion files are staged temporarily on this PC.</p>
                  ) : null}
                  {uploadProgress ? (
                    <div className="mt-3 space-y-2" aria-live="polite">
                      <p className="text-sm">Uploading {uploadingName}: {formatBytes(uploadProgress.loaded)} / {formatBytes(uploadProgress.total)}</p>
                      <progress className="w-full" max={Math.max(uploadProgress.total, 1)} value={uploadProgress.loaded} />
                      <Button type="button" size="sm" variant="outline" onClick={() => {
                        cancelUploadSession();
                        setInspection(null);
                        clearInferredValues();
                        sourcePathRef.current = "";
                        setSourcePath("");
                      }}>Cancel upload</Button>
                    </div>
                  ) : null}
                </div>
                <details className="rounded-md border p-3">
                  <summary className="cursor-pointer text-sm font-medium">Or inspect an existing model folder</summary>
                  <div className="mt-3 space-y-2">
                    <Field label="Model folder">
                      <div className="flex gap-2">
                        <Input
                          className="min-w-0 flex-1"
                          value={uploadId ? "Selected files are staged on this PC" : sourcePath}
                          title={uploadId ? undefined : sourcePath || undefined}
                          placeholder="Choose or enter the local model folder"
                          disabled={Boolean(uploadId) || inspectMutation.isPending || createMutation.isPending}
                          onChange={(event) => updateSourcePath(event.target.value)}
                        />
                        {window.electronAPI?.selectFolder ? (
                          <Button type="button" variant="outline" size="icon" aria-label="Choose model folder" disabled={Boolean(uploadId) || inspectMutation.isPending || createMutation.isPending} onClick={chooseFolder}>
                            <FolderOpen />
                          </Button>
                        ) : null}
                        <Button type="button" variant="outline" disabled={Boolean(uploadId) || !sourcePath.trim() || inspectMutation.isPending} onClick={inspectFolder}>
                          {inspectMutation.isPending ? "Inspecting…" : "Inspect folder"}
                        </Button>
                      </div>
                      {!window.electronAPI?.selectFolder ? (
                        <p className="mt-1 text-xs text-muted-foreground">Paste the full folder path from Windows Explorer on this PC.</p>
                      ) : null}
                    </Field>
                  </div>
                </details>
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
                        {inspection.total_file_count} files · {formatBytes(inspection.total_size_bytes)} will be copied from the selected folder
                      </p>
                      {inspection.total_file_count > 1 ? (
                        <p className="mt-1 text-xs text-muted-foreground">
                          All listed files will be copied, including any extra weights and supporting files.
                        </p>
                      ) : null}
                      {type === "detection" ? (
                        <p className="text-sm text-muted-foreground">
                          {classNamesCount > 0
                            ? `${classNamesCount} class labels found${datasetFile ? ` in ${datasetFile}` : ""}.`
                            : "No class labels found. Enter one class name per line below; IDs will be assigned in order."}
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

                    <Field label="Register for">
                      <Select value={registrationRole || "choose-role"} onValueChange={(value) => {
                        const nextRole = value === "choose-role" ? "" : value as CustomModelRegistrationRole;
                        setRegistrationRole(nextRole);
                        const nextType: CustomModelType | "" = nextRole === "classification"
                          ? "classification"
                          : nextRole ? "detection" : "";
                        setType(nextType);
                        if (nextType === "detection") setEnv(detectorEnvironment(backend));
                        else if (nextType === "classification") {
                          setEnv(inspection.suggested_env_source === "manifest" ? inspection.suggested_env ?? "" : "");
                        } else setEnv("");
                      }}>
                        <SelectTrigger><SelectValue placeholder="Choose how to use this pack" /></SelectTrigger><SelectContent>
                          <SelectItem value="choose-role" disabled>Choose how to use this pack</SelectItem>
                          {inspection.type_candidates.includes("detection") ? (
                            <>
                              <SelectItem value="detection">Detection only</SelectItem>
                              <SelectItem value="both">Detection and Classification (reuse detector classes; no second inference)</SelectItem>
                            </>
                          ) : null}
                          {inspection.type_candidates.includes("classification") && inspection.classifier_inference_compatible ? (
                            <SelectItem value="classification">Classification only (AddaxAI inference.py)</SelectItem>
                          ) : null}
                        </SelectContent>
                      </Select>
                    </Field>
                    {inspection.type_candidates.includes("detection") && !inspection.classifier_inference_compatible ? (
                      <p className="text-sm text-muted-foreground">
                        Classification only requires an AddaxAI-compatible <code>inference.py</code>, which this pack does not have. Choose <strong>Detection and Classification</strong> to reuse the detector classes without a second inference.
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
                    {type === "detection" ? (
                      <>
                        <Field label="Detection backend">
                          <Select value={backend || "choose-backend"} onValueChange={(value) => {
                            const nextBackend = value === "choose-backend" ? "" : value as CustomDetectorBackend;
                            setBackend(nextBackend);
                            if (value !== "rtdetrv2") setDetectorConfig("");
                            setEnv(detectorEnvironment(nextBackend));
                          }}>
                            <SelectTrigger><SelectValue placeholder="Choose a backend" /></SelectTrigger><SelectContent>
                              <SelectItem value="choose-backend" disabled>Choose a backend</SelectItem>
                              {NEW_DETECTOR_BACKENDS.map((value) => (
                                <SelectItem key={value} value={value}>{formatDetectorBackend(value)}</SelectItem>
                              ))}
                            </SelectContent>
                          </Select>
                          {inspection.suggested_detector_backend ? (
                            <p className="mt-1 text-xs text-muted-foreground">Detected from pack contents: {formatDetectorBackend(inspection.suggested_detector_backend)}.</p>
                          ) : null}
                        </Field>
                        <p className="text-xs text-muted-foreground">
                          {backend === "yolo"
                            ? "YOLO runs in the packaged PyTorch environment."
                            : backend === "rtdetrv2"
                              ? <>Official PyTorch RT-DETRv2 runs in the packaged RT-DETR environment. Its YAML must match the checkpoint architecture. See the <a className="underline" href="https://github.com/lyuwenyu/RT-DETR/tree/main/rtdetrv2_pytorch" target="_blank" rel="noreferrer">official PyTorch implementation</a>.</>
                              : "Choose a supported backend; AddaxAI selects its packaged environment automatically."}
                        </p>
                    {backend === "rtdetrv2" ? (
                      <>
                      {inspection.detector_config_candidates.length > 0 ? (
                            <Field label="RT-DETRv2 YAML config">
                              <Select value={detectorConfig || "choose-config"} onValueChange={(value) => {
                                setDetectorConfig(value === "choose-config" ? "" : value);
                                setDetectorConfigTemplate("");
                              }}>
                                <SelectTrigger><SelectValue placeholder="Choose a config" /></SelectTrigger><SelectContent>
                                  <SelectItem value="choose-config" disabled>Choose a config</SelectItem>
                                  {inspection.detector_config_candidates.map((value) => <SelectItem key={value} value={value}>{value}</SelectItem>)}
                                </SelectContent>
                              </Select>
                            </Field>
                      ) : (
                        <Field label="RT-DETRv2 architecture">
                          <Select value={detectorConfigTemplate || "choose-template"} onValueChange={(value) => {
                            setDetectorConfigTemplate(value === "choose-template" ? "" : value);
                            setDetectorConfig("");
                          }}>
                            <SelectTrigger><SelectValue placeholder="Choose an architecture" /></SelectTrigger><SelectContent>
                              <SelectItem value="choose-template" disabled>Choose an architecture</SelectItem>
                              {inspection.detector_config_templates.map((value) => (
                                <SelectItem key={value} value={value}>{formatRtdetrv2Template(value)}</SelectItem>
                              ))}
                            </SelectContent>
                          </Select>
                          <p className="mt-1 text-xs text-muted-foreground">
                            This creates a safe AddaxAI config with your class count and pretrained weights disabled. Choose the backbone that matches the checkpoint.
                          </p>
                        </Field>
                      )}
                      <p className="text-xs text-muted-foreground">
                        If the matching YAML is not already in the selected pack, choose <strong>Add companion files</strong> and add it before registering. For the MDV6-c checkpoint, use the <a className="underline" href="/model-configs/MDV6-apa-rtdetr-c.yml" download>MDV6-c example inference YAML</a>.
                      </p>
                      </>
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

                    {type === "detection" ? (
                      <Field label="Class names (one per line)">
                        <Textarea
                          rows={4}
                          value={classNamesText}
                          placeholder={"fox\ndeer\nvehicle"}
                          onChange={(event) => setClassNamesText(event.target.value)}
                        />
                        <p className="mt-1 text-xs text-muted-foreground">
                          {requiresExplicitClassIds
                            ? "Keep the numeric ID: prefix on every line to preserve the model's class mapping."
                            : "Each non-empty line becomes the next class ID, starting at 0."}
                        </p>
                      </Field>
                    ) : null}

                    <details className="rounded-md border p-3">
                      <summary className="cursor-pointer text-sm font-medium">Advanced settings</summary>
                      <div className="mt-3 space-y-3">
                        {type === "classification" ? (
                          <>
                            <Field label="Inference environment">
                              <Select value={env || "choose-env"} onValueChange={(value) => setEnv(value === "choose-env" ? "" : value)}>
                                <SelectTrigger><SelectValue placeholder="Choose an environment" /></SelectTrigger><SelectContent>
                                  <SelectItem value="choose-env" disabled>Choose an environment</SelectItem>
                                  {environments.map((value) => <SelectItem key={value} value={value}>{formatEnvironment(value)}</SelectItem>)}
                                </SelectContent>
                              </Select>
                            </Field>
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
                    disabled={createMutation.isPending || inspectMutation.isPending || Boolean(uploadProgress) || !canRegister}
                  >
                    <Plus /> {willSelectCreatedModel(type, modelType, registrationRole, Boolean(onAnyCreated), canSelectBoth) ? "Register and select" : "Register"}
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

function filenameStem(filename: string): string {
  const extensionStart = filename.lastIndexOf(".");
  return extensionStart > 0 ? filename.slice(0, extensionStart) : filename;
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

function detectorEnvironment(value: CustomDetectorBackend | ""): string {
  if (value === "yolo") return "pytorch";
  if (value === "rtdetrv2") return "rtdetr";
  return "";
}

function formatRtdetrv2Template(value: string): string {
  const stem = value.replace(/\.ya?ml$/i, "");
  const parts = stem.replace(/^rtdetrv2_/i, "").split("_");
  let backbone: string;
  let details: string[];

  if (parts[0]?.toLowerCase() === "hgnetv2" && parts[1]) {
    backbone = `HGNetv2 ${parts[1].toUpperCase()}`;
    details = parts.slice(2);
  } else {
    const resnet = parts[0]?.match(/^r(\d+)vd$/i);
    if (!resnet) return `${stem} (${value})`;
    backbone = `ResNet ${resnet[1]} VD`;
    details = parts.slice(1);
  }

  const recipeIndex = details.findIndex((part) => /^\d+(?:x|e)$/i.test(part));
  const recipe = recipeIndex >= 0 ? `${details[recipeIndex].toUpperCase()} recipe` : "";
  const dataset = details.find((part) => /^(?:coco|voc)$/i.test(part))?.toUpperCase() ?? "";
  const variants = details
    .filter((part, index) => index !== recipeIndex && !/^(?:coco|voc)$/i.test(part))
    .map((part) => part.toUpperCase());
  const description = [backbone, ...variants, recipe, dataset].filter(Boolean).join(" · ");
  return `${description} (${value})`;
}

function classNamesToLines(classNames: Record<string, string>): string {
  const sequential = hasZeroBasedSequentialIds(classNames);
  return Object.entries(classNames)
    .sort(([left], [right]) => {
      const leftId = Number(left);
      const rightId = Number(right);
      if (Number.isFinite(leftId) && Number.isFinite(rightId)) return leftId - rightId;
      return left.localeCompare(right);
    })
    .map(([id, name]) => sequential ? name : `${id}: ${name}`)
    .join("\n");
}

function hasZeroBasedSequentialIds(classNames: Record<string, string>): boolean {
  const ids = Object.keys(classNames);
  return ids.length > 0 && ids.every((id, index) => id === String(index));
}

function linesToClassNames(value: string, requireExplicitIds = false): Record<string, string> {
  const lines = value.split(/\r?\n/).map((line) => line.trim()).filter(Boolean);
  if (lines.length === 0) throw new Error("Enter at least one class name, one per line.");
  const explicit = lines.map((line) => line.match(/^(\d+)\s*:\s*(.+)$/));
  if (explicit.every(Boolean)) {
    const ids = explicit.map((match) => match![1]);
    const names = explicit.map((match) => match![2].trim());
    if (ids.some((id) => !/^(0|[1-9]\d*)$/.test(id))) {
      throw new Error("Class IDs must use canonical non-negative integers without leading zeroes.");
    }
    validateUniqueClassNames(names);
    return Object.fromEntries(ids.map((id, index) => [id, names[index]]));
  }
  if (explicit.some(Boolean) || requireExplicitIds) {
    throw new Error("Keep a numeric ID: name prefix on every line to preserve this model's class IDs.");
  }
  validateUniqueClassNames(lines);
  return Object.fromEntries(lines.map((name, index) => [String(index), name]));
}

function validateUniqueClassNames(names: string[]): void {
  const normalized = names.map((name) => name.trim().toLowerCase());
  if (normalized.some((name) => !name) || new Set(normalized).size !== normalized.length) {
    throw new Error("Class names must be non-empty and unique, ignoring letter case.");
  }
}

function countClassNames(value: string, requireExplicitIds = false): number {
  if (!value.trim()) return 0;
  try { return Object.keys(linesToClassNames(value, requireExplicitIds)).length; } catch { return 0; }
}

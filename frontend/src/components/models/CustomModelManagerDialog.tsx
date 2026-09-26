import { useEffect, useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { FolderOpen, Pencil, Plus, Trash2 } from "lucide-react";
import { toast } from "sonner";

import { modelsApi } from "@/api/models";
import type {
  CustomModelCreateRequest,
  CustomModelInfo,
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
  const [type, setType] = useState<CustomModelType>(modelType);
  const [sourcePath, setSourcePath] = useState("");
  const [friendlyName, setFriendlyName] = useState("");
  const [env, setEnv] = useState("");
  const [modelFname, setModelFname] = useState("");
  const [description, setDescription] = useState("");
  const [developer, setDeveloper] = useState("");
  const [infoUrl, setInfoUrl] = useState("");
  const [backend, setBackend] = useState<"yolo" | "rfdetr" | "rtdetr" | "rtdetrv2">("yolo");
  const [classNamesText, setClassNamesText] = useState("");
  const [detectorClass, setDetectorClass] = useState("RFDETRMedium");
  const [detectorVariant, setDetectorVariant] = useState("MDV6-apa-rtdetr-c");
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
  const environments = data?.environments ?? [];

  useEffect(() => {
    if (environments.length && !environments.includes(env)) setEnv(environments[0]);
  }, [environments, env]);

  useEffect(() => {
    if (!open) return;
    setType(modelType);
    setEditing(null);
    setErrorText(null);
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
      toast.success(`${model.friendly_name} added`);
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
    setFriendlyName("");
    setModelFname("");
    setDescription("");
    setDeveloper("");
    setInfoUrl("");
    setClassNamesText("");
    setDetectorConfig("");
    setRegion("");
    setFullImage(false);
    setBackend("yolo");
  }

  async function chooseFolder() {
    if (!window.electronAPI?.selectFolder) return;
    try {
      const path = await window.electronAPI.selectFolder();
      if (path) setSourcePath(path);
    } catch (error) {
      setErrorText(getError(error));
    }
  }

  function createModel(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    event.stopPropagation();
    setErrorText(null);
    if (!sourcePath.trim() || !friendlyName.trim() || !env) {
      setErrorText("Choose a model folder, name, and packaged environment.");
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
      payload.detector_backend = backend || "yolo";
      payload.class_names = classNames;
      if (backend === "rfdetr") payload.detector_model_class = detectorClass;
      if (backend === "rtdetr") payload.detector_model_variant = detectorVariant;
      if (backend === "rtdetrv2") payload.detector_config_fname = detectorConfig.trim();
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
    () => (data?.models ?? []).filter((model) => model.type === type),
    [data?.models, type],
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

        <div className="grid gap-6 md:grid-cols-[minmax(0,1fr)_minmax(0,1.1fr)]">
          <section className="space-y-3">
            <h3 className="text-sm font-semibold">Registered {type} models</h3>
            {isLoading ? <p className="text-sm text-muted-foreground">Loading models…</p> : null}
            {isError ? (
              <p role="alert" className="text-sm text-destructive">
                Could not load custom models: {getError(error)}
              </p>
            ) : null}
            {!isLoading && !isError && matchingModels.length === 0 ? (
              <p className="text-sm text-muted-foreground">No custom models registered yet.</p>
            ) : null}
            <div className="space-y-2">
              {matchingModels.map((model) => (
                <div key={model.model_id} className="flex items-center justify-between gap-2 rounded-md border p-3">
                  <div className="min-w-0">
                    <p className="truncate text-sm font-medium">{model.emoji} {model.friendly_name}</p>
                    <p className="truncate text-xs text-muted-foreground">{model.model_id} · {model.detector_backend ?? model.env}</p>
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
          </section>

          <section className="space-y-3">
            <h3 className="text-sm font-semibold">{editing ? `Edit ${editing.friendly_name}` : "Register a model pack"}</h3>
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
                <Field label="Pack type">
                  <Select value={type} onValueChange={(value) => {
                    const nextType = value as CustomModelType;
                    setType(nextType);
                    if (nextType === "detection") setBackend("yolo");
                  }}>
                    <SelectTrigger><SelectValue /></SelectTrigger><SelectContent>
                      <SelectItem value="detection">Detection</SelectItem>
                      <SelectItem value="classification">Classification</SelectItem>
                    </SelectContent>
                  </Select>
                </Field>
                <Field label="Model folder">
                  <div className="flex gap-2">
                    <Input required value={sourcePath} placeholder="Absolute path to the local pack folder" onChange={(e) => setSourcePath(e.target.value)} />
                    {window.electronAPI?.selectFolder ? <Button type="button" variant="outline" size="icon" aria-label="Choose model folder" onClick={chooseFolder}><FolderOpen /></Button> : null}
                  </div>
                </Field>
                <Field label="Name"><Input required value={friendlyName} onChange={(e) => setFriendlyName(e.target.value)} /></Field>
                <Field label="Packaged inference environment">
                  <Select value={env} onValueChange={setEnv} disabled={!environments.length}>
                    <SelectTrigger><SelectValue placeholder="Choose environment" /></SelectTrigger><SelectContent>
                      {environments.map((value) => <SelectItem key={value} value={value}>{value}</SelectItem>)}
                    </SelectContent>
                  </Select>
                </Field>
                <Field label="Weights file (relative path; leave blank if the folder has one weights file)"><Input value={modelFname} onChange={(e) => setModelFname(e.target.value)} /></Field>
                <Field label="Description"><Textarea value={description} onChange={(e) => setDescription(e.target.value)} /></Field>
                <Field label="Developer"><Input value={developer} onChange={(e) => setDeveloper(e.target.value)} /></Field>
                <Field label="Information URL"><Input value={infoUrl} onChange={(e) => setInfoUrl(e.target.value)} /></Field>
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
                      <Checkbox checked={fullImage} onCheckedChange={setFullImage} />
                      Classifies the full image without a detector
                    </label>
                    <p className="text-xs text-muted-foreground">The folder must include an AddaxAI-compatible inference.py.</p>
                  </>
                ) : (
                  <>
                    <Field label="Detection backend">
                      <Select value={backend || "yolo"} onValueChange={(value) => setBackend(value as typeof backend)}>
                        <SelectTrigger><SelectValue /></SelectTrigger><SelectContent>
                          <SelectItem value="yolo">YOLO</SelectItem>
                          <SelectItem value="rfdetr">RF-DETR</SelectItem>
                          <SelectItem value="rtdetr">RT-DETR</SelectItem>
                          <SelectItem value="rtdetrv2">RT-DETRv2</SelectItem>
                        </SelectContent>
                      </Select>
                    </Field>
                    {backend === "rfdetr" ? (
                      <Field label="RF-DETR model class">
                        <Select value={detectorClass} onValueChange={setDetectorClass}>
                          <SelectTrigger><SelectValue /></SelectTrigger><SelectContent>
                            {RFDETR_CLASSES.map((value) => <SelectItem key={value} value={value}>{value}</SelectItem>)}
                          </SelectContent>
                        </Select>
                      </Field>
                    ) : null}
                    {backend === "rtdetr" ? (
                      <Field label="RT-DETR variant">
                        <Select value={detectorVariant} onValueChange={setDetectorVariant}>
                          <SelectTrigger><SelectValue /></SelectTrigger><SelectContent>
                            <SelectItem value="MDV6-apa-rtdetr-c">MDV6-apa-rtdetr-c</SelectItem>
                            <SelectItem value="MDV6-apa-rtdetr-e">MDV6-apa-rtdetr-e</SelectItem>
                          </SelectContent>
                        </Select>
                      </Field>
                    ) : null}
                    {backend === "rtdetrv2" ? <Field label="RT-DETRv2 YAML config (relative path)"><Input required value={detectorConfig} onChange={(e) => setDetectorConfig(e.target.value)} /></Field> : null}
                    <Field label="Class labels (JSON list or string map, optional)"><Textarea value={classNamesText} placeholder={'{"0":"animal","1":"person"}'} onChange={(e) => setClassNamesText(e.target.value)} /></Field>
                  </>
                )}
                {errorText ? <p role="alert" className="text-sm text-destructive">{errorText}</p> : null}
                <DialogFooter>
                  <Button type="submit" disabled={createMutation.isPending || !environments.length}>
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

/**
 * Add Deployment Dialog.
 *
 * Allows users to queue a deployment for ML analysis.
 * Site is automatically set to "Unknown Site" for MVP simplicity.
 */

import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import * as z from "zod";
import { useParams } from "react-router-dom";
import { jobsApi } from "../../api/jobs";
import type {
  JobCreate,
  DetectionModel,
  ClassificationModel,
  DeploymentAnalysisPayload,
} from "../../api/types";
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
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "../ui/select";

const deploymentSchema = z.object({
  folder_path: z.string().min(1, "Folder path is required"),
  detection_model: z.enum(["MegaDetector 5A", "MegaDetector v1000 Redwood"]),
  classification_model: z.enum(["Europe", "Africa"]),
});

type DeploymentFormValues = z.infer<typeof deploymentSchema>;

interface AddDeploymentDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export function AddDeploymentDialog({
  open,
  onOpenChange,
}: AddDeploymentDialogProps) {
  const { projectId } = useParams<{ projectId: string }>();
  const queryClient = useQueryClient();

  const form = useForm<DeploymentFormValues>({
    resolver: zodResolver(deploymentSchema),
    defaultValues: {
      folder_path: "",
      detection_model: "MegaDetector 5A",
      classification_model: "Europe",
    },
  });

  const createJobMutation = useMutation({
    mutationFn: (data: JobCreate) => jobsApi.create(data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["jobs"] });
      form.reset();
      onOpenChange(false);
    },
  });

  const onSubmit = (values: DeploymentFormValues) => {
    if (!projectId) {
      console.error("Project ID is missing");
      return;
    }

    const payload: DeploymentAnalysisPayload = {
      project_id: projectId,
      folder_path: values.folder_path,
      detection_model: values.detection_model as DetectionModel,
      classification_model: values.classification_model as ClassificationModel,
    };

    const jobCreate: JobCreate = {
      type: "deployment_analysis",
      payload: payload as unknown as Record<string, unknown>,
    };

    createJobMutation.mutate(jobCreate);
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-[500px]">
        <DialogHeader>
          <DialogTitle>Add Deployment to Queue</DialogTitle>
          <DialogDescription>
            Select a folder and configure ML models for camera trap analysis.
            Site will be automatically set to "Unknown Site".
          </DialogDescription>
        </DialogHeader>

        <Form {...form}>
          <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-4">
            {/* Folder Path */}
            <FormField
              control={form.control}
              name="folder_path"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Folder Path</FormLabel>
                  <FormControl>
                    <Input
                      placeholder="/Users/you/camera-traps/site-a"
                      {...field}
                    />
                  </FormControl>
                  <FormDescription>
                    Paste the absolute path to your camera trap folder.
                    <br />
                    <em className="text-xs text-muted-foreground">
                      Note: Native folder picker coming with desktop app
                    </em>
                  </FormDescription>
                  <FormMessage />
                </FormItem>
              )}
            />

            {/* Detection Model */}
            <FormField
              control={form.control}
              name="detection_model"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Detection Model</FormLabel>
                  <Select
                    onValueChange={field.onChange}
                    defaultValue={field.value}
                  >
                    <FormControl>
                      <SelectTrigger>
                        <SelectValue placeholder="Select detection model" />
                      </SelectTrigger>
                    </FormControl>
                    <SelectContent>
                      <SelectItem value="MegaDetector 5A">
                        MegaDetector 5A
                      </SelectItem>
                      <SelectItem value="MegaDetector v1000 Redwood">
                        MegaDetector v1000 Redwood
                      </SelectItem>
                    </SelectContent>
                  </Select>
                  <FormDescription>
                    Model for detecting animals in images
                  </FormDescription>
                  <FormMessage />
                </FormItem>
              )}
            />

            {/* Classification Model */}
            <FormField
              control={form.control}
              name="classification_model"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Classification Model</FormLabel>
                  <Select
                    onValueChange={field.onChange}
                    defaultValue={field.value}
                  >
                    <FormControl>
                      <SelectTrigger>
                        <SelectValue placeholder="Select classification model" />
                      </SelectTrigger>
                    </FormControl>
                    <SelectContent>
                      <SelectItem value="Europe">Europe</SelectItem>
                      <SelectItem value="Africa">Africa</SelectItem>
                    </SelectContent>
                  </Select>
                  <FormDescription>
                    Regional species classifier
                  </FormDescription>
                  <FormMessage />
                </FormItem>
              )}
            />

            <DialogFooter>
              <Button
                type="button"
                variant="outline"
                onClick={() => onOpenChange(false)}
              >
                Cancel
              </Button>
              <Button type="submit" disabled={createJobMutation.isPending}>
                {createJobMutation.isPending ? "Adding..." : "Add to Queue"}
              </Button>
            </DialogFooter>
          </form>
        </Form>
      </DialogContent>
    </Dialog>
  );
}

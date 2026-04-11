/**
 * Edit deployment slideout panel.
 */

import { useState, useEffect } from "react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import * as z from "zod";
import { deploymentsApi } from "../../api/deployments";
import type { DeploymentResponse, DeploymentUpdate } from "../../api/types";
import { Button } from "../ui/button";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetFooter,
  SheetHeader,
  SheetTitle,
} from "../ui/sheet";
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
import { TagsEditor } from "../ui/tags-editor";

const deploymentSchema = z.object({
  start_date: z.string().min(1, "Start date is required"),
  end_date: z
    .string()
    .optional()
    .transform((val) => (val === "" ? undefined : val)),
  camera_model: z
    .string()
    .optional()
    .transform((val) => (val === "" ? undefined : val)),
  camera_serial: z
    .string()
    .optional()
    .transform((val) => (val === "" ? undefined : val)),
  datetime_offset_seconds: z
    .union([z.string(), z.number()])
    .optional()
    .transform((val) => {
      if (val === "" || val === undefined) return undefined;
      const num = typeof val === "string" ? parseInt(val, 10) : val;
      return isNaN(num) ? undefined : num;
    }),
  notes: z
    .string()
    .optional()
    .transform((val) => (val === "" ? undefined : val)),
});

interface EditDeploymentDialogProps {
  deployment: DeploymentResponse;
  projectId: string;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export function EditDeploymentDialog({
  deployment,
  projectId,
  open,
  onOpenChange,
}: EditDeploymentDialogProps) {
  const queryClient = useQueryClient();
  const [tags, setTags] = useState<Record<string, string>>(deployment.tags ?? {});

  const form = useForm<DeploymentUpdate>({
    resolver: zodResolver(deploymentSchema),
    defaultValues: {
      start_date: deployment.start_date,
      end_date: deployment.end_date ?? "",
      camera_model: deployment.camera_model ?? "",
      camera_serial: deployment.camera_serial ?? "",
      datetime_offset_seconds: deployment.datetime_offset_seconds ?? undefined,
      notes: deployment.notes ?? "",
    },
  });

  useEffect(() => {
    form.reset({
      start_date: deployment.start_date,
      end_date: deployment.end_date ?? "",
      camera_model: deployment.camera_model ?? "",
      camera_serial: deployment.camera_serial ?? "",
      datetime_offset_seconds: deployment.datetime_offset_seconds ?? undefined,
      notes: deployment.notes ?? "",
    });
    setTags(deployment.tags ?? {});
  }, [deployment, form]);

  const updateMutation = useMutation({
    mutationFn: (data: DeploymentUpdate) =>
      deploymentsApi.update(deployment.id, { ...data, tags }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["deployments", projectId] });
      queryClient.invalidateQueries({ queryKey: ["deployment-stats", projectId] });
      onOpenChange(false);
    },
    onError: (error: Error) => {
      form.setError("root", { message: error.message });
    },
  });

  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent side="right">
        <SheetHeader>
          <SheetTitle>Edit deployment</SheetTitle>
          <SheetDescription>
            Update deployment metadata
          </SheetDescription>
        </SheetHeader>

        <Form {...form}>
          <form
            onSubmit={form.handleSubmit((data) => updateMutation.mutate(data))}
            className="mt-6 space-y-4"
          >
            <div className="grid grid-cols-2 gap-4">
              <FormField
                control={form.control}
                name="start_date"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Start date</FormLabel>
                    <FormControl>
                      <Input type="date" {...field} />
                    </FormControl>
                    <FormMessage />
                  </FormItem>
                )}
              />

              <FormField
                control={form.control}
                name="end_date"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>End date</FormLabel>
                    <FormControl>
                      <Input type="date" {...field} />
                    </FormControl>
                    <FormMessage />
                  </FormItem>
                )}
              />
            </div>

            <div className="grid grid-cols-2 gap-4">
              <FormField
                control={form.control}
                name="camera_model"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Camera model</FormLabel>
                    <FormControl>
                      <Input placeholder="e.g., Reconyx HP2X" {...field} />
                    </FormControl>
                    <FormMessage />
                  </FormItem>
                )}
              />

              <FormField
                control={form.control}
                name="camera_serial"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Camera serial</FormLabel>
                    <FormControl>
                      <Input placeholder="e.g., H500-12345" {...field} />
                    </FormControl>
                    <FormMessage />
                  </FormItem>
                )}
              />
            </div>

            <FormField
              control={form.control}
              name="datetime_offset_seconds"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Datetime offset (seconds)</FormLabel>
                  <FormControl>
                    <Input
                      type="number"
                      placeholder="0"
                      {...field}
                      value={field.value ?? ""}
                    />
                  </FormControl>
                  <FormDescription>
                    Offset applied to file timestamps during analysis
                  </FormDescription>
                  <FormMessage />
                </FormItem>
              )}
            />

            <FormField
              control={form.control}
              name="notes"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Notes</FormLabel>
                  <FormControl>
                    <Textarea placeholder="Additional notes about this deployment" {...field} />
                  </FormControl>
                  <FormMessage />
                </FormItem>
              )}
            />

            <TagsEditor value={tags} onChange={setTags} />

            {form.formState.errors.root && (
              <p className="text-sm font-medium text-destructive">
                {form.formState.errors.root.message}
              </p>
            )}

            <SheetFooter className="gap-2 sm:gap-0 pt-4">
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
            </SheetFooter>
          </form>
        </Form>
      </SheetContent>
    </Sheet>
  );
}

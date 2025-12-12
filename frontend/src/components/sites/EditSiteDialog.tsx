/**
 * Edit Site Dialog.
 */

import { useEffect } from "react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import * as z from "zod";
import { sitesApi } from "../../api/sites";
import type { SiteUpdate, SiteResponse } from "../../api/types";
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
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
} from "../ui/form";
import { Input } from "../ui/input";

const siteSchema = z.object({
  name: z.string().min(1, "Site name is required"),
  latitude: z
    .union([z.string(), z.number()])
    .optional()
    .transform((val) => {
      if (val === "" || val === undefined) return undefined;
      const num = typeof val === "string" ? parseFloat(val) : val;
      return isNaN(num) ? undefined : num;
    }),
  longitude: z
    .union([z.string(), z.number()])
    .optional()
    .transform((val) => {
      if (val === "" || val === undefined) return undefined;
      const num = typeof val === "string" ? parseFloat(val) : val;
      return isNaN(num) ? undefined : num;
    }),
  elevation_m: z
    .union([z.string(), z.number()])
    .optional()
    .transform((val) => {
      if (val === "" || val === undefined) return undefined;
      const num = typeof val === "string" ? parseFloat(val) : val;
      return isNaN(num) ? undefined : num;
    }),
  habitat_type: z
    .string()
    .optional()
    .transform((val) => (val === "" ? undefined : val)),
  notes: z
    .string()
    .optional()
    .transform((val) => (val === "" ? undefined : val)),
});

interface EditSiteDialogProps {
  site: SiteResponse;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export function EditSiteDialog({
  site,
  open,
  onOpenChange,
}: EditSiteDialogProps) {
  const queryClient = useQueryClient();

  const form = useForm<SiteUpdate>({
    resolver: zodResolver(siteSchema),
    defaultValues: {
      name: site.name,
      latitude: site.latitude ?? undefined,
      longitude: site.longitude ?? undefined,
      elevation_m: site.elevation_m ?? undefined,
      habitat_type: site.habitat_type ?? "",
      notes: site.notes ?? "",
    },
  });

  // Reset form when site changes
  useEffect(() => {
    form.reset({
      name: site.name,
      latitude: site.latitude ?? undefined,
      longitude: site.longitude ?? undefined,
      elevation_m: site.elevation_m ?? undefined,
      habitat_type: site.habitat_type ?? "",
      notes: site.notes ?? "",
    });
  }, [site, form]);

  const updateMutation = useMutation({
    mutationFn: (data: SiteUpdate) => sitesApi.update(site.id, data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["sites", site.project_id] });
      queryClient.invalidateQueries({ queryKey: ["sites", site.id] });
      onOpenChange(false);
    },
    onError: (error: Error) => {
      form.setError("root", { message: error.message });
    },
  });

  const deleteMutation = useMutation({
    mutationFn: () => sitesApi.delete(site.id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["sites", site.project_id] });
      onOpenChange(false);
    },
    onError: (error: Error) => {
      form.setError("root", { message: error.message });
    },
  });

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Edit Site</DialogTitle>
          <DialogDescription>
            Update site details or delete the site
          </DialogDescription>
        </DialogHeader>

        <Form {...form}>
          <form
            onSubmit={form.handleSubmit((data) => updateMutation.mutate(data))}
            className="space-y-4"
          >
            <FormField
              control={form.control}
              name="name"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Site Name</FormLabel>
                  <FormControl>
                    <Input placeholder="e.g., North Ridge" {...field} />
                  </FormControl>
                  <FormMessage />
                </FormItem>
              )}
            />

            <div className="grid grid-cols-2 gap-4">
              <FormField
                control={form.control}
                name="latitude"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Latitude</FormLabel>
                    <FormControl>
                      <Input type="number" step="any" placeholder="44.4280" {...field} />
                    </FormControl>
                    <FormMessage />
                  </FormItem>
                )}
              />

              <FormField
                control={form.control}
                name="longitude"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Longitude</FormLabel>
                    <FormControl>
                      <Input type="number" step="any" placeholder="-110.5885" {...field} />
                    </FormControl>
                    <FormMessage />
                  </FormItem>
                )}
              />
            </div>

            <FormField
              control={form.control}
              name="elevation_m"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Elevation (meters)</FormLabel>
                  <FormControl>
                    <Input type="number" placeholder="2000" {...field} />
                  </FormControl>
                  <FormMessage />
                </FormItem>
              )}
            />

            <FormField
              control={form.control}
              name="habitat_type"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Habitat Type</FormLabel>
                  <FormControl>
                    <Input placeholder="e.g., Forest, Grassland" {...field} />
                  </FormControl>
                  <FormMessage />
                </FormItem>
              )}
            />

            {form.formState.errors.root && (
              <p className="text-sm font-medium text-destructive">
                {form.formState.errors.root.message}
              </p>
            )}

            <DialogFooter className="gap-2 sm:gap-0">
              <Button
                type="button"
                variant="destructive"
                onClick={() => {
                  if (confirm("Are you sure you want to delete this site? This action cannot be undone.")) {
                    deleteMutation.mutate();
                  }
                }}
                disabled={deleteMutation.isPending}
                className="sm:mr-auto"
              >
                {deleteMutation.isPending ? "Deleting..." : "Delete"}
              </Button>
              <Button
                type="button"
                variant="outline"
                onClick={() => onOpenChange(false)}
              >
                Cancel
              </Button>
              <Button type="submit" disabled={updateMutation.isPending}>
                {updateMutation.isPending ? "Saving..." : "Save Changes"}
              </Button>
            </DialogFooter>
          </form>
        </Form>
      </DialogContent>
    </Dialog>
  );
}

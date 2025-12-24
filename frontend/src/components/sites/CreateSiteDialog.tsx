/**
 * Create Site Dialog.
 */

import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import * as z from "zod";
import { sitesApi } from "../../api/sites";
import type { SiteCreate } from "../../api/types";
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

interface CreateSiteDialogProps {
  projectId: string;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export function CreateSiteDialog({
  projectId,
  open,
  onOpenChange,
}: CreateSiteDialogProps) {
  const queryClient = useQueryClient();

  const form = useForm<Omit<SiteCreate, "project_id">>({
    resolver: zodResolver(siteSchema),
    defaultValues: {
      name: "",
      latitude: undefined,
      longitude: undefined,
      elevation_m: undefined,
      habitat_type: "",
      notes: "",
    },
  });

  const createMutation = useMutation({
    mutationFn: (data: Omit<SiteCreate, "project_id">) =>
      sitesApi.create({ ...data, project_id: projectId }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["sites", projectId] });
      form.reset();
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
          <DialogTitle>Create new site</DialogTitle>
          <DialogDescription>
            Add a new site to this project
          </DialogDescription>
        </DialogHeader>

        <Form {...form}>
          <form
            onSubmit={form.handleSubmit((data) => createMutation.mutate(data))}
            className="space-y-4"
          >
            <FormField
              control={form.control}
              name="name"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Site name</FormLabel>
                  <FormControl>
                    <Input placeholder="e.g., North Ridge" {...field} />
                  </FormControl>
                  <FormDescription>
                    A descriptive name for this camera trap location
                  </FormDescription>
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
                    <FormDescription>
                      Decimal degrees (e.g., 44.4280)
                    </FormDescription>
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
                    <FormDescription>
                      Decimal degrees (e.g., -110.5885)
                    </FormDescription>
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
                  <FormDescription>
                    Height above sea level in meters
                  </FormDescription>
                  <FormMessage />
                </FormItem>
              )}
            />

            <FormField
              control={form.control}
              name="habitat_type"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Habitat type</FormLabel>
                  <FormControl>
                    <Input placeholder="e.g., Forest, Grassland" {...field} />
                  </FormControl>
                  <FormDescription>
                    Main vegetation or ecosystem type
                  </FormDescription>
                  <FormMessage />
                </FormItem>
              )}
            />

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
                {createMutation.isPending ? "Creating..." : "Create Site"}
              </Button>
            </DialogFooter>
          </form>
        </Form>
      </DialogContent>
    </Dialog>
  );
}

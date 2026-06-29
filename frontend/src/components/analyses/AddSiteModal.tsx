/**
 * Site Modal Component (create and edit)
 *
 * Modal dialog for creating or editing a camera trap site.
 * - Form with site name, coordinates, elevation, habitat type, notes
 * - Interactive map for location selection
 * - Auto-fills lat/lon from map clicks
 * - In edit mode: prefills all fields from existing site
 */

import { useState, useEffect } from "react";
import { useForm, type Resolver } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { z } from "zod";
import { RefreshCw } from "lucide-react";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { Callout } from "@/components/ui/callout";
import { sitesApi } from "@/api/sites";
import type { SiteResponse } from "@/api/types";
import { TagsEditor } from "@/components/ui/tags-editor";
import { invalidateProjectData } from "@/lib/invalidate-project";
import { SiteMap } from "./SiteMap";

// Validation schema
const siteSchema = z.object({
  name: z.string().min(1, "Site name is required"),
  latitude: z.number().min(-90).max(90),
  longitude: z.number().min(-180).max(180),
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

type SiteFormData = z.infer<typeof siteSchema>;

interface AddSiteModalProps {
  projectId: string;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onSiteCreated?: (siteId: string) => void;
  initialLocation?: { lat: number; lon: number };
  /** When provided, modal operates in edit mode with prefilled values. */
  site?: SiteResponse;
}

export function AddSiteModal({
  projectId,
  open,
  onOpenChange,
  onSiteCreated,
  initialLocation,
  site,
}: AddSiteModalProps) {
  const queryClient = useQueryClient();
  const isEditMode = !!site;

  const [selectedLocation, setSelectedLocation] = useState<{ lat: number; lon: number } | null>(
    null
  );
  const [mapOffline, setMapOffline] = useState(false);
  const [showMap, setShowMap] = useState(true);
  const [tags, setTags] = useState<Record<string, string>>({});

  // Form setup
  const {
    register,
    handleSubmit,
    setValue,
    watch,
    reset,
    formState: { errors },
  } = useForm<SiteFormData>({
    resolver: zodResolver(siteSchema) as Resolver<SiteFormData>,
    defaultValues: {
      name: "",
      latitude: 0,
      longitude: 0,
      elevation_m: undefined,
      habitat_type: "",
      notes: "",
    },
  });

  const latitude = watch("latitude");
  const longitude = watch("longitude");

  // Prefill form when modal opens
  useEffect(() => {
    if (!open) return;

    if (isEditMode) {
      // Edit mode: prefill from existing site
      setValue("name", site.name);
      setValue("latitude", site.latitude ?? 0);
      setValue("longitude", site.longitude ?? 0);
      setValue("elevation_m", site.elevation_m ?? undefined);
      setValue("habitat_type", site.habitat_type ?? "");
      setValue("notes", site.notes ?? "");
      setTags(site.tags ?? {});
      if (site.latitude != null && site.longitude != null) {
        setSelectedLocation({ lat: site.latitude, lon: site.longitude });
      }
    } else if (initialLocation) {
      // Create mode with GPS pre-fill
      setValue("latitude", initialLocation.lat);
      setValue("longitude", initialLocation.lon);
      setSelectedLocation(initialLocation);
    }
  }, [open, isEditMode, site, initialLocation, setValue]);

  // Update selected location when form values change (only for valid coordinates)
  useEffect(() => {
    if (typeof latitude === "number" && !isNaN(latitude) &&
        typeof longitude === "number" && !isNaN(longitude) &&
        latitude >= -90 && latitude <= 90 &&
        longitude >= -180 && longitude <= 180) {
      setSelectedLocation({ lat: latitude, lon: longitude });
    }
  }, [latitude, longitude]);

  // Create mutation
  const createSite = useMutation({
    mutationFn: (data: SiteFormData) =>
      sitesApi.create({
        project_id: projectId,
        name: data.name,
        latitude: data.latitude,
        longitude: data.longitude,
        elevation_m: data.elevation_m ?? null,
        habitat_type: data.habitat_type ?? null,
        notes: data.notes ?? null,
        tags,
      }),
    onSuccess: (newSite) => {
      // Site coordinates feed into the Map insights page, Dashboard
      // sun bands, Activity overlap sun bands, deployment popups, and
      // export pipelines. Use the blanket invalidator so every view
      // refreshes after a create / edit, not just the sites table.
      invalidateProjectData(queryClient, projectId);
      onSiteCreated?.(newSite.id);
      onOpenChange(false);
      reset();
      setSelectedLocation(null);
    },
    onError: (error) => {
      console.error("Failed to create site:", error);
    },
  });

  // Update mutation
  const updateSite = useMutation({
    mutationFn: (data: SiteFormData) =>
      sitesApi.update(site!.id, {
        name: data.name,
        latitude: data.latitude,
        longitude: data.longitude,
        elevation_m: data.elevation_m ?? null,
        habitat_type: data.habitat_type ?? null,
        notes: data.notes ?? null,
        tags,
      }),
    onSuccess: () => {
      invalidateProjectData(queryClient, projectId);
      onOpenChange(false);
      reset();
      setSelectedLocation(null);
    },
    onError: (error) => {
      console.error("Failed to update site:", error);
    },
  });

  const mutation = isEditMode ? updateSite : createSite;

  // Handle map location selection
  const handleLocationSelect = (lat: number, lon: number) => {
    setValue("latitude", lat);
    setValue("longitude", lon);
    setSelectedLocation({ lat, lon });
  };

  // Handle map error (offline)
  const handleMapError = () => {
    setMapOffline(true);
    setShowMap(false);
  };

  // Retry loading map
  const handleRetryMap = () => {
    setMapOffline(false);
    setShowMap(true);
  };

  // Handle form submission
  const onSubmit = (data: SiteFormData) => {
    mutation.mutate(data);
  };

  // Reset form when modal closes
  useEffect(() => {
    if (!open) {
      reset();
      setSelectedLocation(null);
      setTags({});
      setMapOffline(false);
      setShowMap(true);
    }
  }, [open, reset]);

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-2xl max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle>
            {isEditMode ? "Edit site" : "Add new site"}
          </DialogTitle>
          <DialogDescription>
            {isEditMode
              ? "Update this site's details. Click on the map to change the location."
              : "Create a new camera trap site for this project. Click on the map to set the location."}
          </DialogDescription>
        </DialogHeader>

        <form onSubmit={handleSubmit(onSubmit)} className="space-y-4">
          {/* Site name */}
          <div className="space-y-2">
            <Label htmlFor="name">
              Site name
            </Label>
            <Input
              id="name"
              {...register("name")}
              placeholder="e.g., Forest ridge north"
              className={errors.name ? "border-red-500" : ""}
            />
            {errors.name && <p className="text-sm text-red-600">{errors.name.message}</p>}
          </div>

          {/* Offline notice */}
          {mapOffline && (
            <Callout
              variant="warning"
              action={
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  onClick={handleRetryMap}
                >
                  <RefreshCw className="h-4 w-4 mr-2" />
                  Retry map
                </Button>
              }
            >
              Map unavailable offline. Enter coordinates manually.
            </Callout>
          )}

          {/* Map */}
          {showMap && (
            <div className="space-y-2">
              <Label>
                Location
              </Label>
              <SiteMap
                projectId={projectId}
                selectedLocation={selectedLocation}
                onLocationSelect={handleLocationSelect}
                onMapError={handleMapError}
                excludeSiteId={site?.id}
              />
            </div>
          )}

          {/* Coordinates */}
          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label htmlFor="latitude">Latitude</Label>
              <Input
                id="latitude"
                {...register("latitude", { valueAsNumber: true })}
                type="number"
                step="any"
                placeholder="e.g., 44.4280"
              />
              {errors.latitude && (
                <p className="text-sm text-red-600">{errors.latitude.message}</p>
              )}
            </div>

            <div className="space-y-2">
              <Label htmlFor="longitude">Longitude</Label>
              <Input
                id="longitude"
                {...register("longitude", { valueAsNumber: true })}
                type="number"
                step="any"
                placeholder="e.g., -110.5885"
              />
              {errors.longitude && (
                <p className="text-sm text-red-600">{errors.longitude.message}</p>
              )}
            </div>
          </div>

          {/* Extra fields */}
          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label htmlFor="elevation_m">Elevation in meters</Label>
              <Input
                id="elevation_m"
                {...register("elevation_m")}
                type="number"
                step="any"
                placeholder="e.g., 2000"
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="habitat_type">Habitat type</Label>
              <Input
                id="habitat_type"
                {...register("habitat_type")}
                placeholder="e.g., Forest, Grassland"
              />
            </div>
          </div>

          <div className="space-y-2">
            <Label htmlFor="notes">Notes</Label>
            <Textarea
              id="notes"
              {...register("notes")}
              maxLength={1000}
              placeholder="e.g., Honey badgers keep stealing the SD cards"
            />
          </div>

          {/* Tags */}
          <TagsEditor
            value={tags}
            onChange={setTags}
            keyPlaceholder="e.g., access"
            valuePlaceholder="e.g., 4x4 only"
          />

          {/* Error message */}
          {mutation.isError && (
            <div className="text-sm text-red-600">
              {isEditMode ? "Failed to update site." : "Failed to create site."}{" "}
              {mutation.error instanceof Error
                ? mutation.error.message
                : String(mutation.error)}
            </div>
          )}

          <DialogFooter>
            <Button type="button" variant="outline" onClick={() => onOpenChange(false)}>
              Cancel
            </Button>
            <Button type="submit" disabled={mutation.isPending || !selectedLocation}>
              {mutation.isPending
                ? (isEditMode ? "Saving..." : "Creating...")
                : (isEditMode ? "Save changes" : "Create site")}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
}

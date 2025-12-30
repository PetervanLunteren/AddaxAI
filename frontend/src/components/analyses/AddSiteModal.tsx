/**
 * Add Site Modal Component
 *
 * Modal dialog for creating a new camera trap site.
 * - Form with site name and coordinates
 * - Interactive map for location selection
 * - Auto-fills lat/lon from map clicks
 * - Validates uniqueness and required fields
 */

import { useState, useEffect } from "react";
import { useForm } from "react-hook-form";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { z } from "zod";
import { MapPin } from "lucide-react";
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
import { Label } from "@/components/ui/label";
import { sitesApi } from "@/api/sites";
import { SiteMap } from "./SiteMap";

// Validation schema
const siteSchema = z.object({
  name: z.string().min(1, "Site name is required"),
  latitude: z.number().min(-90).max(90),
  longitude: z.number().min(-180).max(180),
});

type SiteFormData = z.infer<typeof siteSchema>;

interface AddSiteModalProps {
  projectId: string;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onSiteCreated?: (siteId: string) => void;
}

export function AddSiteModal({
  projectId,
  open,
  onOpenChange,
  onSiteCreated,
}: AddSiteModalProps) {
  const queryClient = useQueryClient();
  const [selectedLocation, setSelectedLocation] = useState<{ lat: number; lon: number } | null>(
    null
  );

  // Form setup
  const {
    register,
    handleSubmit,
    setValue,
    watch,
    reset,
    formState: { errors },
  } = useForm<SiteFormData>({
    defaultValues: {
      name: "",
      latitude: 0,
      longitude: 0,
    },
  });

  const latitude = watch("latitude");
  const longitude = watch("longitude");

  // Update selected location when form values change
  useEffect(() => {
    if (latitude && longitude) {
      setSelectedLocation({ lat: latitude, lon: longitude });
    }
  }, [latitude, longitude]);

  // Create site mutation
  const createSite = useMutation({
    mutationFn: (data: SiteFormData) =>
      sitesApi.create({
        project_id: projectId,
        name: data.name,
        latitude: data.latitude,
        longitude: data.longitude,
      }),
    onSuccess: (newSite) => {
      // Refresh sites list
      queryClient.invalidateQueries({ queryKey: ["sites", projectId] });

      // Call callback with new site ID
      onSiteCreated?.(newSite.id);

      // Close modal and reset form
      onOpenChange(false);
      reset();
      setSelectedLocation(null);
    },
  });

  // Handle map location selection
  const handleLocationSelect = (lat: number, lon: number) => {
    setValue("latitude", lat);
    setValue("longitude", lon);
    setSelectedLocation({ lat, lon });
  };

  // Handle form submission
  const onSubmit = (data: SiteFormData) => {
    createSite.mutate(data);
  };

  // Reset form when modal closes
  useEffect(() => {
    if (!open) {
      reset();
      setSelectedLocation(null);
    }
  }, [open, reset]);

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-2xl max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <MapPin className="h-5 w-5" />
            Add new site
          </DialogTitle>
          <DialogDescription>
            Create a new camera trap site for this project. Click on the map to set the location.
          </DialogDescription>
        </DialogHeader>

        <form onSubmit={handleSubmit(onSubmit)} className="space-y-4">
          {/* Site name */}
          <div className="space-y-2">
            <Label htmlFor="name">
              Site name
              <span className="text-red-600 ml-1">*</span>
            </Label>
            <Input
              id="name"
              {...register("name")}
              placeholder="e.g., Forest Ridge North"
              className={errors.name ? "border-red-500" : ""}
            />
            {errors.name && <p className="text-sm text-red-600">{errors.name.message}</p>}
          </div>

          {/* Map */}
          <div className="space-y-2">
            <Label>
              Location
              <span className="text-red-600 ml-1">*</span>
            </Label>
            <SiteMap
              projectId={projectId}
              selectedLocation={selectedLocation}
              onLocationSelect={handleLocationSelect}
            />
          </div>

          {/* Coordinates (readonly) */}
          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label htmlFor="latitude">Latitude</Label>
              <Input
                id="latitude"
                {...register("latitude", { valueAsNumber: true })}
                type="number"
                step="any"
                readOnly
                className="bg-gray-50"
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
                readOnly
                className="bg-gray-50"
              />
              {errors.longitude && (
                <p className="text-sm text-red-600">{errors.longitude.message}</p>
              )}
            </div>
          </div>

          {/* Error message */}
          {createSite.isError && (
            <div className="text-sm text-red-600">
              Failed to create site. {createSite.error instanceof Error && createSite.error.message}
            </div>
          )}

          <DialogFooter>
            <Button type="button" variant="outline" onClick={() => onOpenChange(false)}>
              Cancel
            </Button>
            <Button type="submit" disabled={createSite.isPending || !selectedLocation}>
              {createSite.isPending ? "Creating..." : "Create site"}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
}

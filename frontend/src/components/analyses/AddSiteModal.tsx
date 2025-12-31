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
import { zodResolver } from "@hookform/resolvers/zod";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { z } from "zod";
import { MapPin, WifiOff, RefreshCw } from "lucide-react";
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
import { Alert, AlertDescription } from "@/components/ui/alert";
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
  initialLocation?: { lat: number; lon: number };
}

export function AddSiteModal({
  projectId,
  open,
  onOpenChange,
  onSiteCreated,
  initialLocation,
}: AddSiteModalProps) {
  const queryClient = useQueryClient();
  const [selectedLocation, setSelectedLocation] = useState<{ lat: number; lon: number } | null>(
    null
  );
  const [mapOffline, setMapOffline] = useState(false);
  const [showMap, setShowMap] = useState(true);

  // Form setup
  const {
    register,
    handleSubmit,
    setValue,
    watch,
    reset,
    formState: { errors },
  } = useForm<SiteFormData>({
    resolver: zodResolver(siteSchema),
    defaultValues: {
      name: "",
      latitude: 0,
      longitude: 0,
    },
  });

  const latitude = watch("latitude");
  const longitude = watch("longitude");

  // Set initial location when modal opens with GPS from deployment
  useEffect(() => {
    if (open && initialLocation) {
      setValue("latitude", initialLocation.lat);
      setValue("longitude", initialLocation.lon);
      setSelectedLocation(initialLocation);
    }
  }, [open, initialLocation, setValue]);

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
    onError: (error) => {
      console.error("Failed to create site:", error);
    },
  });

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
    createSite.mutate(data);
  };

  // Reset form when modal closes
  useEffect(() => {
    if (!open) {
      reset();
      setSelectedLocation(null);
      setMapOffline(false);
      setShowMap(true);
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
            </Label>
            <Input
              id="name"
              {...register("name")}
              placeholder="e.g., Forest Ridge North"
              className={errors.name ? "border-red-500" : ""}
            />
            {errors.name && <p className="text-sm text-red-600">{errors.name.message}</p>}
          </div>

          {/* Offline notice */}
          {mapOffline && (
            <Alert>
              <WifiOff className="h-4 w-4" />
              <AlertDescription className="flex items-center justify-between">
                <span>Map unavailable offline. Enter coordinates manually.</span>
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  onClick={handleRetryMap}
                >
                  <RefreshCw className="h-4 w-4 mr-2" />
                  Retry map
                </Button>
              </AlertDescription>
            </Alert>
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
              />
            </div>
          )}

          {/* Coordinates (editable when offline, readonly when map works) */}
          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label htmlFor="latitude">Latitude</Label>
              <Input
                id="latitude"
                {...register("latitude", { valueAsNumber: true })}
                type="number"
                step="any"
                readOnly={!mapOffline}
                className={mapOffline ? "" : "bg-gray-50"}
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
                readOnly={!mapOffline}
                className={mapOffline ? "" : "bg-gray-50"}
              />
              {errors.longitude && (
                <p className="text-sm text-red-600">{errors.longitude.message}</p>
              )}
            </div>
          </div>

          {/* Error message */}
          {createSite.isError && (
            <div className="text-sm text-red-600">
              Failed to create site.{" "}
              {createSite.error instanceof Error
                ? createSite.error.message
                : typeof createSite.error === 'object' && createSite.error !== null
                  ? JSON.stringify(createSite.error)
                  : String(createSite.error)}
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

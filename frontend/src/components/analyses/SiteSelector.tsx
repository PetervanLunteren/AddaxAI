/**
 * Site Selector Component
 *
 * Simplified version matching Create Project modal style.
 * - Clean select dropdown with info tooltip
 * - Button to add new site
 * - Inline validation
 */

import { useEffect } from "react";
import { useQuery } from "@tanstack/react-query";
import { MapPin, Plus, Info } from "lucide-react";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Button } from "@/components/ui/button";
import { sitesApi } from "@/api/sites";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";

interface SiteSelectorProps {
  projectId: string;
  value: string | null;
  onChange: (id: string) => void;
  onAddNew: () => void;
  deploymentGps?: { latitude: number; longitude: number } | null;
}

export function SiteSelector({
  projectId,
  value,
  onChange,
  onAddNew,
  deploymentGps,
}: SiteSelectorProps) {
  const { data: sites, isLoading } = useQuery({
    queryKey: ["sites", projectId],
    queryFn: () => sitesApi.list(projectId),
  });

  const selectedSite = sites?.find((s) => s.id === value);

  // Auto-select closest site when deployment GPS is available
  useEffect(() => {
    if (!deploymentGps || !sites || sites.length === 0 || value) return;

    // Find site with GPS that is closest to deployment
    const sitesWithGps = sites.filter(
      (site) => site.latitude != null && site.longitude != null
    );

    if (sitesWithGps.length === 0) return;

    // Calculate distances and find closest
    const sitesWithDistances = sitesWithGps.map((site) => ({
      site,
      distance: calculateDistance(
        deploymentGps.latitude,
        deploymentGps.longitude,
        site.latitude!,
        site.longitude!
      ),
    }));

    const closest = sitesWithDistances.reduce((prev, current) =>
      current.distance < prev.distance ? current : prev
    );

    // Auto-select only if within 100m (0.1km)
    // Camera traps at the same site shouldn't move more than 100m between deployments
    if (closest.distance <= 0.1) {
      onChange(closest.site.id);
    }
  }, [deploymentGps, sites, value, onChange]);

  // Calculate distance using Haversine formula
  const calculateDistance = (
    lat1: number,
    lon1: number,
    lat2: number,
    lon2: number
  ): number => {
    const R = 6371; // Earth's radius in km
    const dLat = ((lat2 - lat1) * Math.PI) / 180;
    const dLon = ((lon2 - lon1) * Math.PI) / 180;
    const a =
      Math.sin(dLat / 2) * Math.sin(dLat / 2) +
      Math.cos((lat1 * Math.PI) / 180) *
        Math.cos((lat2 * Math.PI) / 180) *
        Math.sin(dLon / 2) *
        Math.sin(dLon / 2);
    const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
    return R * c; // Distance in km
  };

  const formatDistance = (km: number): string => {
    if (km < 1) {
      return `${Math.round(km * 1000)}m`;
    }
    return `${km.toFixed(1)}km`;
  };

  return (
    <TooltipProvider>
      <div className="space-y-2">
        {/* Label with info tooltip */}
        <label className="flex items-center gap-1.5 text-sm font-medium">
          Camera site
          <Tooltip>
            <TooltipTrigger asChild>
              <Info className="h-3.5 w-3.5 text-muted-foreground cursor-help" />
            </TooltipTrigger>
            <TooltipContent>
              <p className="max-w-xs">
                The camera trap location where these images were captured. You can add a new site if needed.
              </p>
            </TooltipContent>
          </Tooltip>
        </label>

        {/* Select + Add button */}
        <div className="flex gap-2">
          <Select value={value ?? ""} onValueChange={onChange} disabled={isLoading}>
            <SelectTrigger className="flex-1">
              <SelectValue placeholder={isLoading ? "Loading sites..." : "Select a site"} />
            </SelectTrigger>
            <SelectContent>
              {sites && sites.length > 0 ? (
                sites.map((site) => {
                  // Calculate distance if both deployment GPS and site GPS are available
                  let distanceText = null;
                  if (
                    deploymentGps &&
                    site.latitude != null &&
                    site.longitude != null
                  ) {
                    const distance = calculateDistance(
                      deploymentGps.latitude,
                      deploymentGps.longitude,
                      site.latitude,
                      site.longitude
                    );
                    distanceText = formatDistance(distance);
                  }

                  return (
                    <SelectItem key={site.id} value={site.id}>
                      <div className="flex items-center gap-2">
                        <MapPin className="h-4 w-4 text-gray-400" />
                        <span>{site.name}</span>
                        {distanceText && (
                          <span className="text-xs text-gray-500">
                            ({distanceText})
                          </span>
                        )}
                      </div>
                    </SelectItem>
                  );
                })
              ) : (
                <div className="p-2 text-sm text-gray-500 text-center">
                  No sites found - click + to add one
                </div>
              )}
            </SelectContent>
          </Select>

          <Button
            type="button"
            variant="outline"
            size="icon"
            onClick={onAddNew}
            title="Add new site"
            className="shrink-0"
          >
            <Plus className="h-4 w-4" />
          </Button>
        </div>
      </div>
    </TooltipProvider>
  );
}

/**
 * Site Selector Component
 *
 * Simplified version matching Create Project modal style.
 * - Clean select dropdown
 * - Button to add new site
 * - Inline validation
 */

import { useEffect } from "react";
import { useQuery } from "@tanstack/react-query";
import { MapPin, Plus } from "lucide-react";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Button } from "@/components/ui/button";
import { FieldHeader } from "@/components/ui/field-header";
import { sitesApi } from "@/api/sites";

// Reserved option value that maps to `null` in the parent's state.
// shadcn's Select refuses empty-string item values, so we pick an
// unambiguous sentinel and translate it at the component boundary.
const NO_SITE_OPTION = "__no_site__";

interface SiteSelectorProps {
  projectId: string;
  value: string | null;
  onChange: (id: string | null) => void;
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
    <div className="space-y-2">
      {/* Shared with EditDeploymentDialog, so the caption has to read
          sensibly in both. That rules out a "you can set it later" note,
          which is odd wording inside the very dialog for setting it later.
          The marker carries it: optional means you may leave it blank, and
          NoSiteBanner nudges anyone who does. */}
      <FieldHeader
        label={
          <label className="text-sm font-medium">
            Camera site
            <span className="ml-1 font-normal text-muted-foreground">optional</span>
          </label>
        }
        caption="Puts this camera on the map and in site comparisons."
      />

      {/* Select + Add button */}
      <div className="flex gap-2">
        <Select
          value={value ?? ""}
          onValueChange={(v) => onChange(v === NO_SITE_OPTION ? null : v)}
          disabled={isLoading}
        >
          <SelectTrigger className="flex-1">
            <SelectValue
              placeholder={isLoading ? "Loading sites..." : "Optionally select a site"}
            />
          </SelectTrigger>
          <SelectContent>
            {/* "(no site)" is only a meaningful choice alongside real
                sites. With zero sites the deployment already defaults to
                no site, so a lone "(no site)" option is a confusing no-op;
                show the add-a-site hint instead (below). */}
            {sites && sites.length > 0 && (
              <SelectItem key={NO_SITE_OPTION} value={NO_SITE_OPTION}>
                <div className="flex items-center gap-2">
                  <span className="text-muted-foreground">(no site)</span>
                </div>
              </SelectItem>
            )}
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
                No sites yet, click + to add one
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
  );
}

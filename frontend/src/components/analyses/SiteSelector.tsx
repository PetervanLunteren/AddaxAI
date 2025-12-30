/**
 * Site Selector Component
 *
 * Dropdown to select an existing site, with option to create new site.
 * Opens AddSiteModal when "+ Add new site" is selected.
 */

import { useState } from "react";
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
import { sitesApi } from "@/api/sites";

interface SiteSelectorProps {
  projectId: string;
  value: string | null;
  onChange: (id: string) => void;
  onAddNew: () => void;
  error?: string;
}

export function SiteSelector({
  projectId,
  value,
  onChange,
  onAddNew,
  error,
}: SiteSelectorProps) {
  const { data: sites, isLoading } = useQuery({
    queryKey: ["sites", projectId],
    queryFn: () => sitesApi.list(projectId),
  });

  const selectedSite = sites?.find((s) => s.id === value);

  return (
    <div className="space-y-2">
      <label className="text-sm font-medium">
        Camera site
        <span className="text-red-600 ml-1">*</span>
      </label>

      <div className="flex gap-2">
        <Select value={value || undefined} onValueChange={onChange} disabled={isLoading}>
          <SelectTrigger className={`flex-1 ${error ? "border-red-500" : ""}`}>
            <SelectValue placeholder={isLoading ? "Loading sites..." : "Select a site"}>
              {selectedSite && (
                <div className="flex items-center gap-2">
                  <MapPin className="h-4 w-4 text-gray-400" />
                  <span>{selectedSite.name}</span>
                </div>
              )}
            </SelectValue>
          </SelectTrigger>
          <SelectContent>
            {sites && sites.length > 0 ? (
              sites.map((site) => (
                <SelectItem key={site.id} value={site.id}>
                  <div className="flex items-center gap-2">
                    <MapPin className="h-4 w-4 text-gray-400" />
                    <span>{site.name}</span>
                    {site.latitude && site.longitude && (
                      <span className="text-xs text-gray-500">
                        ({site.latitude.toFixed(4)}, {site.longitude.toFixed(4)})
                      </span>
                    )}
                  </div>
                </SelectItem>
              ))
            ) : (
              <div className="p-2 text-sm text-gray-500 text-center">
                No sites found for this project
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
        >
          <Plus className="h-4 w-4" />
        </Button>
      </div>

      {error && <p className="text-sm text-red-600">{error}</p>}

      <p className="text-xs text-gray-500">
        Select the camera trap location, or add a new site
      </p>
    </div>
  );
}

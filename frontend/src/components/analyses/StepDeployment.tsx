/**
 * Step 2: Deployment - Site/Camera location selection with modern styling.
 */

import { useState } from "react";
import { Label } from "@/components/ui/label";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Callout } from "@/components/ui/callout";
import { useQuery } from "@tanstack/react-query";
import { useWizard } from "@/components/wizard/WizardContext";
import { MapPin, Plus, Check } from "lucide-react";

interface StepDeploymentProps {
  projectId: string;
  siteId: string | null;
  onSiteChange: (siteId: string | null) => void;
}

export function StepDeployment({
  projectId,
  siteId,
  onSiteChange,
}: StepDeploymentProps) {
  const { setCanGoNext } = useWizard();
  const [modalOpen, setModalOpen] = useState(false);
  const [newSiteName, setNewSiteName] = useState("");
  const [newSiteNotes, setNewSiteNotes] = useState("");
  const [latitude, setLatitude] = useState("");
  const [longitude, setLongitude] = useState("");

  // Fetch sites for this project
  const { data: sites, isLoading, refetch } = useQuery({
    queryKey: ["sites", projectId],
    queryFn: async () => {
      const response = await fetch(`/api/sites?project_id=${projectId}`);
      if (!response.ok) throw new Error("Failed to fetch sites");
      return response.json();
    },
  });

  const handleSiteChange = (value: string) => {
    if (value === "new") {
      setModalOpen(true);
      setCanGoNext(false);
    } else {
      onSiteChange(value);
      setCanGoNext(true);
    }
  };

  const handleCreateSite = async () => {
    if (!newSiteName.trim()) return;

    try {
      const payload = {
        project_id: projectId,
        name: newSiteName,
        notes: newSiteNotes || null,
        latitude: latitude ? parseFloat(latitude) : null,
        longitude: longitude ? parseFloat(longitude) : null,
        elevation_m: null,
        habitat_type: null,
      };

      console.log("Creating site with payload:", payload);

      const response = await fetch("/api/sites", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      console.log("Response status:", response.status);

      if (!response.ok) {
        const errorText = await response.text();
        console.error("Error response:", errorText);
        try {
          const error = JSON.parse(errorText);
          alert(`Failed to create site: ${error.detail || errorText}`);
        } catch {
          alert(`Failed to create site: ${errorText}`);
        }
        return;
      }

      const newSite = await response.json();
      console.log("Created site:", newSite);

      // Reset form
      setNewSiteName("");
      setNewSiteNotes("");
      setLatitude("");
      setLongitude("");
      setModalOpen(false);

      // Refresh sites list and select the new site
      await refetch();
      onSiteChange(newSite.id);
      setCanGoNext(true);
    } catch (error) {
      console.error("Failed to create site:", error);
      alert(`Failed to create site. Error: ${error}`);
    }
  };

  const selectedSite = sites?.find((s: any) => s.id === siteId);

  return (
    <div className="space-y-6">
      <div>
        <Label htmlFor="siteId" className="text-lg font-semibold text-gray-900 flex items-center gap-2">
          <MapPin className="w-5 h-5 text-blue-600" />
          Camera Location
        </Label>
        <p className="text-sm text-gray-600 mt-2">
          Select the site where this camera was deployed. Sites represent physical locations in your project.
        </p>
      </div>

      <div className="space-y-4">
        <Select value={siteId || undefined} onValueChange={handleSiteChange} disabled={isLoading}>
          <SelectTrigger className="w-full h-12 text-base">
            <SelectValue placeholder={isLoading ? "Loading sites..." : "Select a site"} />
          </SelectTrigger>
          <SelectContent>
            {sites?.map((site: any) => (
              <SelectItem key={site.id} value={site.id} className="text-base py-3">
                <div className="flex items-center gap-2">
                  <MapPin className="w-4 h-4 text-gray-500" />
                  {site.name}
                </div>
              </SelectItem>
            ))}
            <SelectItem value="new" className="font-medium text-blue-600 py-3">
              <div className="flex items-center gap-2">
                <Plus className="w-4 h-4" />
                Add new site
              </div>
            </SelectItem>
          </SelectContent>
        </Select>

        {selectedSite && (
          <Callout variant="success" title="Site selected" className="animate-in fade-in slide-in-from-top-2 duration-300">
            <p>{selectedSite.name}</p>
            {selectedSite.notes && (
              <p className="text-xs mt-1 opacity-80">{selectedSite.notes}</p>
            )}
          </Callout>
        )}
      </div>

      {/* Modal Dialog for "Add New Site" */}
      <Dialog open={modalOpen} onOpenChange={setModalOpen}>
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle>Add New Site</DialogTitle>
            <DialogDescription>
              Create a new camera trap site for your project
            </DialogDescription>
          </DialogHeader>

          <div className="space-y-4 py-4">
            <div className="space-y-2">
              <Label htmlFor="siteName">Site Name *</Label>
              <Input
                id="siteName"
                placeholder="e.g., North Ridge Trail"
                value={newSiteName}
                onChange={(e) => setNewSiteName(e.target.value)}
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="notes">Notes (optional)</Label>
              <Input
                id="notes"
                placeholder="Dense forest area..."
                value={newSiteNotes}
                onChange={(e) => setNewSiteNotes(e.target.value)}
              />
            </div>

            <div className="space-y-2">
              <Label>Location</Label>
              <div className="h-32 bg-gray-100 rounded-lg flex items-center justify-center border-2 border-dashed">
                <p className="text-sm text-gray-500">🗺️ Map integration coming soon</p>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label htmlFor="lat">Latitude</Label>
                <Input
                  id="lat"
                  type="number"
                  placeholder="51.5074"
                  value={latitude}
                  onChange={(e) => setLatitude(e.target.value)}
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="lon">Longitude</Label>
                <Input
                  id="lon"
                  type="number"
                  placeholder="-0.1278"
                  value={longitude}
                  onChange={(e) => setLongitude(e.target.value)}
                />
              </div>
            </div>
          </div>

          <DialogFooter className="sm:justify-between">
            <Button variant="outline" onClick={() => setModalOpen(false)} className="w-full sm:w-auto">
              Cancel
            </Button>
            <Button onClick={handleCreateSite} disabled={!newSiteName.trim()} className="w-full sm:w-auto">
              Save
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}

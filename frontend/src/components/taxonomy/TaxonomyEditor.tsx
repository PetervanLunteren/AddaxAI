/**
 * Taxonomy Editor Component.
 *
 * Following DEVELOPERS.md principles:
 * - Type hints everywhere
 * - Simple, clear implementation
 * - MVP: Flat list with checkboxes (tree component deferred)
 *
 * Full-screen modal for selecting species classes to monitor.
 */

import { useEffect, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { modelsApi } from "../../api/models";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "../ui/dialog";
import { Button } from "../ui/button";
import { Input } from "../ui/input";
import { Checkbox } from "../ui/checkbox";
import { ScrollArea } from "../ui/scroll-area";

interface TaxonomyEditorProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  modelId: string | null;
  selectedClasses: string[];
  onSave: (selectedClasses: string[]) => void;
}

export function TaxonomyEditor({
  open,
  onOpenChange,
  modelId,
  selectedClasses,
  onSave,
}: TaxonomyEditorProps) {
  const [searchQuery, setSearchQuery] = useState("");
  const [localSelected, setLocalSelected] = useState<Set<string>>(
    new Set(selectedClasses)
  );

  // Fetch taxonomy when model changes
  const { data: taxonomy, isLoading } = useQuery({
    queryKey: ["taxonomy", modelId],
    queryFn: () => modelsApi.getTaxonomy(modelId!),
    enabled: open && !!modelId,
  });

  // Update local state when selectedClasses prop changes
  useEffect(() => {
    setLocalSelected(new Set(selectedClasses));
  }, [selectedClasses]);

  const allClasses = taxonomy?.all_classes || [];
  const filteredClasses = allClasses.filter((cls) =>
    cls.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const handleToggle = (className: string) => {
    const newSelected = new Set(localSelected);
    if (newSelected.has(className)) {
      newSelected.delete(className);
    } else {
      newSelected.add(className);
    }
    setLocalSelected(newSelected);
  };

  const handleSelectAll = () => {
    setLocalSelected(new Set(filteredClasses));
  };

  const handleDeselectAll = () => {
    setLocalSelected(new Set());
  };

  const handleInvert = () => {
    const newSelected = new Set<string>();
    allClasses.forEach((cls) => {
      if (!localSelected.has(cls)) {
        newSelected.add(cls);
      }
    });
    setLocalSelected(newSelected);
  };

  const handleSave = () => {
    onSave(Array.from(localSelected));
    onOpenChange(false);
  };

  const handleCancel = () => {
    // Reset to original selection
    setLocalSelected(new Set(selectedClasses));
    onOpenChange(false);
  };

  if (!modelId) {
    return (
      <Dialog open={open} onOpenChange={onOpenChange}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Configure Taxonomy</DialogTitle>
            <DialogDescription>
              Please select a classification model first.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button onClick={() => onOpenChange(false)}>Close</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    );
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-4xl h-[80vh] flex flex-col">
        <DialogHeader>
          <DialogTitle>Configure Species Taxonomy</DialogTitle>
          <DialogDescription>
            Select which species classes you want to monitor for this project.
            {localSelected.size > 0 && (
              <span className="ml-2 font-medium">
                ({localSelected.size} selected)
              </span>
            )}
          </DialogDescription>
        </DialogHeader>

        <div className="flex-1 flex flex-col gap-4 min-h-0">
          {/* Search */}
          <Input
            placeholder="Search species..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
          />

          {/* Bulk actions */}
          <div className="flex gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={handleSelectAll}
              disabled={isLoading}
            >
              Select All {searchQuery && `(${filteredClasses.length})`}
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={handleDeselectAll}
              disabled={isLoading}
            >
              Deselect All
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={handleInvert}
              disabled={isLoading}
            >
              Invert Selection
            </Button>
          </div>

          {/* Species list */}
          <ScrollArea className="flex-1 border rounded-md">
            {isLoading ? (
              <div className="p-4 text-center text-muted-foreground">
                Loading taxonomy...
              </div>
            ) : filteredClasses.length === 0 ? (
              <div className="p-4 text-center text-muted-foreground">
                {searchQuery ? "No species found" : "No species available"}
              </div>
            ) : (
              <div className="p-4 space-y-2">
                {filteredClasses.map((className) => (
                  <div
                    key={className}
                    className="flex items-center space-x-2 p-2 hover:bg-accent rounded-md cursor-pointer"
                    onClick={() => handleToggle(className)}
                  >
                    <Checkbox
                      checked={localSelected.has(className)}
                      onCheckedChange={() => handleToggle(className)}
                    />
                    <label className="flex-1 cursor-pointer capitalize">
                      {className}
                    </label>
                  </div>
                ))}
              </div>
            )}
          </ScrollArea>
        </div>

        <DialogFooter>
          <Button variant="outline" onClick={handleCancel}>
            Cancel
          </Button>
          <Button onClick={handleSave} disabled={isLoading}>
            Save ({localSelected.size} species)
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

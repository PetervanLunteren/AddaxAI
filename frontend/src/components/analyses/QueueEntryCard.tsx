/**
 * Queue Entry Card - compact, modern card for kanban board.
 */

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { FolderOpen, Trash2, ChevronDown, ChevronUp, Brain, MapPin } from "lucide-react";
import { basename } from "@/lib/path-utils";

interface QueueEntry {
  id: string;
  folder_path: string;
  site_id: string | null;
  detection_model_id: string | null;
  classification_model_id: string | null;
  species_list: Record<string, any> | null;
  status: string;
  created_at_utc: string;
}

interface QueueEntryCardProps {
  entry: QueueEntry;
  onRemove: (id: string) => void;
}

export function QueueEntryCard({ entry, onRemove }: QueueEntryCardProps) {
  const [isExpanded, setIsExpanded] = useState(false);

  const folderName = basename(entry.folder_path) || entry.folder_path;

  const statusStyles: Record<string, string> = {
    pending: "bg-card border-border",
    processing: "bg-blue-50 border-blue-300 shadow-md",
    completed: "bg-green-50 border-green-300",
    failed: "bg-red-50 border-red-300",
  };

  return (
    <div
      className={`
        rounded-lg border-2 transition-all duration-200 overflow-hidden
        ${statusStyles[entry.status] || statusStyles.pending}
        ${isExpanded ? "shadow-md" : "shadow-sm hover:shadow-md"}
      `}
    >
      {/* Header */}
      <div className="p-3">
        <div className="flex items-start gap-2">
          <FolderOpen className="w-4 h-4 text-muted-foreground mt-0.5 shrink-0" />
          <div className="flex-1 min-w-0">
            <p className="font-medium text-sm text-foreground truncate">{folderName}</p>
            <p className="text-xs text-muted-foreground mt-0.5">
              {new Date(entry.created_at_utc).toLocaleDateString()}
            </p>
          </div>
          <div className="flex gap-1 shrink-0">
            <Button
              type="button"
              variant="ghost"
              size="sm"
              onClick={() => setIsExpanded(!isExpanded)}
              className="h-7 w-7 p-0"
            >
              {isExpanded ? (
                <ChevronUp className="w-3.5 h-3.5" />
              ) : (
                <ChevronDown className="w-3.5 h-3.5" />
              )}
            </Button>
            <Button
              type="button"
              variant="ghost"
              size="sm"
              onClick={() => onRemove(entry.id)}
              className="h-7 w-7 p-0 text-red-600 hover:text-red-700 hover:bg-red-50"
            >
              <Trash2 className="w-3.5 h-3.5" />
            </Button>
          </div>
        </div>
      </div>

      {/* Expanded Details */}
      {isExpanded && (
        <div className="px-3 pb-3 pt-1 border-t border-border bg-card/50 space-y-2 text-xs">
          <div className="flex items-start gap-2">
            <MapPin className="w-3.5 h-3.5 text-muted-foreground mt-0.5 shrink-0" />
            <div>
              <span className="font-medium text-muted-foreground">Site:</span>{" "}
              <span className="text-muted-foreground">{entry.site_id || "Not specified"}</span>
            </div>
          </div>
          <div className="flex items-start gap-2">
            <Brain className="w-3.5 h-3.5 text-muted-foreground mt-0.5 shrink-0" />
            <div>
              <span className="font-medium text-muted-foreground">Detection:</span>{" "}
              <span className="text-muted-foreground">
                {entry.detection_model_id || "None"}
              </span>
            </div>
          </div>
          <div className="flex items-start gap-2">
            <Brain className="w-3.5 h-3.5 text-muted-foreground mt-0.5 shrink-0" />
            <div>
              <span className="font-medium text-muted-foreground">Classification:</span>{" "}
              <span className="text-muted-foreground">
                {entry.classification_model_id || "None"}
              </span>
            </div>
          </div>
          <div className="pt-1 border-t border-border">
            <p className="text-muted-foreground break-all font-mono">{entry.folder_path}</p>
          </div>
        </div>
      )}
    </div>
  );
}

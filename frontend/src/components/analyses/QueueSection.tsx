/**
 * Queue Section - Kanban-style board for deployment queue.
 *
 * Shows deployments in columns: Pending → Processing → Completed
 */

import { Button } from "@/components/ui/button";
import { Callout } from "@/components/ui/callout";
import { QueueEntryCard } from "./QueueEntryCard";
import { Play, Clock, Loader2, CheckCircle2, XCircle } from "lucide-react";

interface QueueEntry {
  id: string;
  folder_path: string;
  site_id: string | null;
  detection_model_id: string | null;
  classification_model_id: string | null;
  species_list: Record<string, any> | null;
  status: string;
  created_at: string;
}

interface QueueSectionProps {
  entries: QueueEntry[];
  isLoading: boolean;
  onRemove: (id: string) => void;
  onProcess: () => void;
  isProcessing: boolean;
}

export function QueueSection({
  entries,
  isLoading,
  onRemove,
  onProcess,
  isProcessing,
}: QueueSectionProps) {
  const pendingEntries = entries.filter((e) => e.status === "pending");
  const processingEntries = entries.filter((e) => e.status === "processing");
  const completedEntries = entries.filter((e) => e.status === "completed");
  const failedEntries = entries.filter((e) => e.status === "failed");

  if (isLoading) {
    return (
      <div className="text-center py-12">
        <Loader2 className="w-8 h-8 text-gray-400 animate-spin mx-auto mb-3" />
        <p className="text-gray-500">Loading queue...</p>
      </div>
    );
  }

  if (entries.length === 0) {
    return (
      <Callout variant="info" title="No deployments in queue">
        <p>Add deployments using the wizard above to start analyzing camera trap images.</p>
        <Button variant="outline" size="sm" disabled className="mt-3">
          <Play className="w-3 h-3 mr-2" />
          Process Queue
        </Button>
      </Callout>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-xl font-bold text-gray-900">Analysis Queue</h3>
          <p className="text-sm text-gray-600 mt-1">
            {pendingEntries.length} pending • {processingEntries.length} processing • {completedEntries.length} completed
          </p>
        </div>
        {pendingEntries.length > 0 && !isProcessing && (
          <Button
            onClick={onProcess}
            size="lg"
            className="bg-gradient-to-r from-blue-600 to-blue-700 hover:from-blue-700 hover:to-blue-800 shadow-md"
          >
            <Play className="w-4 h-4 mr-2" />
            Process Queue
          </Button>
        )}
        {isProcessing && (
          <Button
            disabled
            size="lg"
            className="bg-gradient-to-r from-blue-600 to-blue-700"
          >
            <Loader2 className="w-4 h-4 mr-2 animate-spin" />
            Processing...
          </Button>
        )}
      </div>

      {/* Kanban Board */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {/* Pending Column */}
        <div className="space-y-3">
          <div className="flex items-center gap-2 px-4 py-2 bg-gray-100 rounded-t-lg">
            <Clock className="w-4 h-4 text-gray-600" />
            <h4 className="font-semibold text-gray-900">
              Pending ({pendingEntries.length})
            </h4>
          </div>
          <div className="space-y-3 min-h-[200px]">
            {pendingEntries.map((entry) => (
              <QueueEntryCard key={entry.id} entry={entry} onRemove={onRemove} />
            ))}
            {pendingEntries.length === 0 && (
              <div className="p-6 text-center text-sm text-gray-500 border-2 border-dashed border-gray-200 rounded-lg">
                No pending deployments
              </div>
            )}
          </div>
        </div>

        {/* Processing Column */}
        <div className="space-y-3">
          <div className="flex items-center gap-2 px-4 py-2 bg-blue-100 rounded-t-lg">
            <Loader2 className="w-4 h-4 text-blue-600 animate-spin" />
            <h4 className="font-semibold text-blue-900">
              Processing ({processingEntries.length})
            </h4>
          </div>
          <div className="space-y-3 min-h-[200px]">
            {processingEntries.map((entry) => (
              <QueueEntryCard key={entry.id} entry={entry} onRemove={onRemove} />
            ))}
            {processingEntries.length === 0 && (
              <div className="p-6 text-center text-sm text-gray-500 border-2 border-dashed border-gray-200 rounded-lg">
                No active processing
              </div>
            )}
          </div>
        </div>

        {/* Completed Column */}
        <div className="space-y-3">
          <div className="flex items-center gap-2 px-4 py-2 bg-green-100 rounded-t-lg">
            <CheckCircle2 className="w-4 h-4 text-green-600" />
            <h4 className="font-semibold text-green-900">
              Completed ({completedEntries.length + failedEntries.length})
            </h4>
          </div>
          <div className="space-y-3 min-h-[200px]">
            {completedEntries.map((entry) => (
              <QueueEntryCard key={entry.id} entry={entry} onRemove={onRemove} />
            ))}
            {failedEntries.map((entry) => (
              <QueueEntryCard key={entry.id} entry={entry} onRemove={onRemove} />
            ))}
            {completedEntries.length === 0 && failedEntries.length === 0 && (
              <div className="p-6 text-center text-sm text-gray-500 border-2 border-dashed border-gray-200 rounded-lg">
                No completed analyses
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

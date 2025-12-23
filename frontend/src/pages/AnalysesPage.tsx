/**
 * Analyses Page - New Analysis Wizard & Queue Management
 *
 * 4-step wizard for adding deployments to queue:
 * 1. Data (folder selection)
 * 2. Deployment (site selection)
 * 3. Model (detection + classification)
 * 4. Species (expected species)
 *
 * Plus queue display with process button.
 */

import { useParams } from "react-router-dom";
import { DeploymentWizard } from "../components/analyses/DeploymentWizard";
import { QueueSection } from "../components/analyses/QueueSection";
import {
  useDeploymentQueue,
  useRemoveFromQueue,
  useProcessQueue,
} from "../hooks/useDeploymentQueue";

export function AnalysesPage() {
  const { projectId } = useParams<{ projectId: string }>();

  // Fetch deployment queue
  const { data: queueEntries, isLoading: queueLoading } = useDeploymentQueue(
    projectId || ""
  );

  // Mutations
  const removeFromQueue = useRemoveFromQueue();
  const processQueue = useProcessQueue();

  if (!projectId) {
    return (
      <div className="p-8">
        <p className="text-red-600">Project ID missing</p>
      </div>
    );
  }

  const handleRemove = async (id: string) => {
    if (confirm("Remove this entry from the queue?")) {
      await removeFromQueue.mutateAsync(id);
    }
  };

  const handleProcess = async () => {
    if (confirm("Start processing all pending deployments in the queue?")) {
      await processQueue.mutateAsync({ project_id: projectId });
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="border-b bg-white">
        <div className="mx-auto max-w-7xl px-4 py-4 sm:px-6 lg:px-8">
          <div>
            <h1 className="text-2xl font-bold tracking-tight">New Analysis</h1>
            <p className="text-sm text-muted-foreground">
              Configure and add deployments to the analysis queue
            </p>
          </div>
        </div>
      </header>

      {/* Main Content - Full Page Wizard */}
      <main className="mx-auto max-w-5xl px-4 py-8 sm:px-6 lg:px-8 space-y-16">
        {/* Add to Queue Section */}
        <section>
          <div className="mb-6">
            <h2 className="text-xl font-semibold text-gray-900">Add to Queue</h2>
            <div className="mt-2 h-px bg-gradient-to-r from-gray-300 to-transparent" />
          </div>
          <DeploymentWizard projectId={projectId} />
        </section>

        {/* Process Queue Section */}
        <section>
          <div className="mb-6">
            <h2 className="text-xl font-semibold text-gray-900">Process Queue</h2>
            <div className="mt-2 h-px bg-gradient-to-r from-gray-300 to-transparent" />
          </div>
          <QueueSection
            entries={queueEntries || []}
            isLoading={queueLoading}
            onRemove={handleRemove}
            onProcess={handleProcess}
            isProcessing={processQueue.isPending}
          />
        </section>
      </main>
    </div>
  );
}

/**
 * Analyses Page - New Two-Card Layout
 *
 * Simple, clean interface for managing deployment analysis queue:
 * - Card 1 (left): Add deployment form
 * - Card 2 (right): Queue display and run button
 *
 * Replaces previous wizard-based approach with simpler form.
 */

import { useParams } from "react-router-dom";
import { AddDeploymentCard } from "../components/analyses/AddDeploymentCard";
import { QueueCard } from "../components/analyses/QueueCard";

export function AnalysesPage() {
  const { projectId } = useParams<{ projectId: string }>();

  if (!projectId) {
    return (
      <div className="p-8">
        <p className="text-red-600">Project ID missing</p>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="border-b bg-white">
        <div className="mx-auto max-w-7xl px-4 py-4 sm:px-6 lg:px-8">
          <div>
            <h1 className="text-2xl font-bold tracking-tight">New analysis</h1>
            <p className="text-sm text-muted-foreground">
              Add deployments to the queue and process them with your configured models
            </p>
          </div>
        </div>
      </header>

      {/* Main Content - Two-Card Layout */}
      <main className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Card 1: Add Deployment */}
          <div>
            <AddDeploymentCard projectId={projectId} />
          </div>

          {/* Card 2: Queue Display */}
          <div>
            <QueueCard projectId={projectId} />
          </div>
        </div>
      </main>
    </div>
  );
}

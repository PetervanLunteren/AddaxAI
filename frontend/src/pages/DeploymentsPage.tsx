/**
 * Deployments Page - ML Model Deployment Management
 *
 * Allows users to create and manage ML model deployments to analyze images.
 */

import { useState } from "react";
import { useParams } from "react-router-dom";
import { Plus, Play, Pause, Trash2, Clock } from "lucide-react";
import { Button } from "../components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "../components/ui/card";

// Mock deployment data - will be replaced with real API calls
interface Deployment {
  id: string;
  name: string;
  model: string;
  status: "running" | "completed" | "paused" | "failed";
  progress: number;
  created_at: string;
  images_processed: number;
  total_images: number;
}

const mockDeployments: Deployment[] = [
  {
    id: "1",
    name: "Wildlife Detection - Summer 2024",
    model: "MegaDetector v5",
    status: "running",
    progress: 65,
    created_at: "2024-12-10T10:30:00Z",
    images_processed: 650,
    total_images: 1000,
  },
  {
    id: "2",
    name: "Species Classification",
    model: "Wildlife Classifier v2",
    status: "completed",
    progress: 100,
    created_at: "2024-12-08T14:20:00Z",
    images_processed: 500,
    total_images: 500,
  },
];

export function DeploymentsPage() {
  const { projectId } = useParams<{ projectId: string }>();
  const [createDialogOpen, setCreateDialogOpen] = useState(false);
  const [deployments] = useState<Deployment[]>(mockDeployments);

  const getStatusColor = (status: Deployment["status"]) => {
    switch (status) {
      case "running":
        return "text-blue-600 bg-blue-50";
      case "completed":
        return "text-green-600 bg-green-50";
      case "paused":
        return "text-yellow-600 bg-yellow-50";
      case "failed":
        return "text-red-600 bg-red-50";
    }
  };

  const getStatusIcon = (status: Deployment["status"]) => {
    switch (status) {
      case "running":
        return <Play className="h-3 w-3" />;
      case "completed":
        return <Clock className="h-3 w-3" />;
      case "paused":
        return <Pause className="h-3 w-3" />;
      case "failed":
        return <Trash2 className="h-3 w-3" />;
    }
  };

  if (!projectId) {
    return <div>Project ID missing</div>;
  }

  return (
    <div className="min-h-screen">
      {/* Header */}
      <header className="border-b bg-white/80 backdrop-blur-sm">
        <div className="mx-auto max-w-7xl px-4 py-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold tracking-tight">Deployments</h1>
              <p className="text-sm text-muted-foreground">
                Run ML models to analyze your camera trap images
              </p>
            </div>
            <Button onClick={() => setCreateDialogOpen(true)}>
              <Plus className="h-4 w-4" />
              New Deployment
            </Button>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8">
        {deployments.length > 0 ? (
          <div className="space-y-4">
            {deployments.map((deployment) => (
              <Card key={deployment.id} className="transition-shadow hover:shadow-lg">
                <CardHeader>
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <div className="flex items-center gap-3">
                        <CardTitle>{deployment.name}</CardTitle>
                        <span
                          className={`inline-flex items-center gap-1 rounded-full px-2 py-1 text-xs font-medium ${getStatusColor(
                            deployment.status
                          )}`}
                        >
                          {getStatusIcon(deployment.status)}
                          {deployment.status.charAt(0).toUpperCase() +
                            deployment.status.slice(1)}
                        </span>
                      </div>
                      <CardDescription className="mt-1">
                        Model: {deployment.model}
                      </CardDescription>
                    </div>
                    <div className="flex gap-2">
                      {deployment.status === "running" && (
                        <Button variant="outline" size="sm">
                          <Pause className="h-4 w-4" />
                        </Button>
                      )}
                      {deployment.status === "paused" && (
                        <Button variant="outline" size="sm">
                          <Play className="h-4 w-4" />
                        </Button>
                      )}
                      <Button variant="outline" size="sm">
                        <Trash2 className="h-4 w-4" />
                      </Button>
                    </div>
                  </div>
                </CardHeader>
                <CardContent className="space-y-4">
                  {/* Progress Bar */}
                  <div className="space-y-2">
                    <div className="flex justify-between text-sm">
                      <span className="text-muted-foreground">Progress</span>
                      <span className="font-medium">{deployment.progress}%</span>
                    </div>
                    <div className="h-2 w-full overflow-hidden rounded-full bg-muted">
                      <div
                        className="h-full bg-primary transition-all duration-500"
                        style={{ width: `${deployment.progress}%` }}
                      />
                    </div>
                    <p className="text-xs text-muted-foreground">
                      {deployment.images_processed.toLocaleString()} /{" "}
                      {deployment.total_images.toLocaleString()} images processed
                    </p>
                  </div>

                  {/* Metadata */}
                  <div className="flex items-center justify-between border-t pt-4 text-xs text-muted-foreground">
                    <span>
                      Started: {new Date(deployment.created_at).toLocaleDateString()}{" "}
                      {new Date(deployment.created_at).toLocaleTimeString()}
                    </span>
                    {deployment.status === "running" && (
                      <span className="text-blue-600">
                        Estimated time remaining: ~15 minutes
                      </span>
                    )}
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        ) : (
          <Card>
            <CardContent className="flex flex-col items-center justify-center py-12">
              <p className="mb-4 text-muted-foreground">
                No deployments yet. Create your first deployment to start analyzing images.
              </p>
              <Button onClick={() => setCreateDialogOpen(true)}>
                <Plus className="h-4 w-4" />
                Create Deployment
              </Button>
            </CardContent>
          </Card>
        )}
      </main>

      {/* TODO: Add CreateDeploymentDialog */}
      {createDialogOpen && (
        <div>CreateDeploymentDialog - Coming soon</div>
      )}
    </div>
  );
}

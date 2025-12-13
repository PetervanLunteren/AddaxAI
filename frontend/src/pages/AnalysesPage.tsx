/**
 * Analyses Page - ML Analysis Management
 *
 * Allows users to create and manage ML analyses to detect animals in camera trap images.
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

// Mock analysis data - will be replaced with real API calls
interface Analysis {
  id: string;
  name: string;
  model: string;
  status: "running" | "completed" | "paused" | "failed";
  progress: number;
  created_at: string;
  images_processed: number;
  total_images: number;
}

const mockAnalyses: Analysis[] = [
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

export function AnalysesPage() {
  const { projectId } = useParams<{ projectId: string }>();
  const [createDialogOpen, setCreateDialogOpen] = useState(false);
  const [analyses] = useState<Analysis[]>(mockAnalyses);

  const getStatusColor = (status: Analysis["status"]) => {
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

  const getStatusIcon = (status: Analysis["status"]) => {
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
              <h1 className="text-2xl font-bold tracking-tight">New Analysis</h1>
              <p className="text-sm text-muted-foreground">
                Run AI models to detect and classify wildlife in your camera trap images
              </p>
            </div>
            <Button onClick={() => setCreateDialogOpen(true)}>
              <Plus className="h-4 w-4" />
              Create Analysis
            </Button>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8">
        {analyses.length > 0 ? (
          <div className="space-y-4">
            {analyses.map((analysis) => (
              <Card key={analysis.id} className="transition-shadow hover:shadow-lg">
                <CardHeader>
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <div className="flex items-center gap-3">
                        <CardTitle>{analysis.name}</CardTitle>
                        <span
                          className={`inline-flex items-center gap-1 rounded-full px-2 py-1 text-xs font-medium ${getStatusColor(
                            analysis.status
                          )}`}
                        >
                          {getStatusIcon(analysis.status)}
                          {analysis.status.charAt(0).toUpperCase() +
                            analysis.status.slice(1)}
                        </span>
                      </div>
                      <CardDescription className="mt-1">
                        Model: {analysis.model}
                      </CardDescription>
                    </div>
                    <div className="flex gap-2">
                      {analysis.status === "running" && (
                        <Button variant="outline" size="sm">
                          <Pause className="h-4 w-4" />
                        </Button>
                      )}
                      {analysis.status === "paused" && (
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
                      <span className="font-medium">{analysis.progress}%</span>
                    </div>
                    <div className="h-2 w-full overflow-hidden rounded-full bg-muted">
                      <div
                        className="h-full bg-primary transition-all duration-500"
                        style={{ width: `${analysis.progress}%` }}
                      />
                    </div>
                    <p className="text-xs text-muted-foreground">
                      {analysis.images_processed.toLocaleString()} /{" "}
                      {analysis.total_images.toLocaleString()} images processed
                    </p>
                  </div>

                  {/* Metadata */}
                  <div className="flex items-center justify-between border-t pt-4 text-xs text-muted-foreground">
                    <span>
                      Started: {new Date(analysis.created_at).toLocaleDateString()}{" "}
                      {new Date(analysis.created_at).toLocaleTimeString()}
                    </span>
                    {analysis.status === "running" && (
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
                No analyses yet. Create your first analysis to start detecting wildlife.
              </p>
              <Button onClick={() => setCreateDialogOpen(true)}>
                <Plus className="h-4 w-4" />
                Create Analysis
              </Button>
            </CardContent>
          </Card>
        )}
      </main>

      {/* TODO: Add CreateAnalysisDialog */}
      {createDialogOpen && (
        <div>CreateAnalysisDialog - Coming soon</div>
      )}
    </div>
  );
}

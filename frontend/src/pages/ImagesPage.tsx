/**
 * Images browser page - displays files with detections
 */

import { useQuery } from "@tanstack/react-query";
import { useParams } from "react-router-dom";
import { useState } from "react";
import { filesApi } from "../api/files";
import { projectsApi } from "../api/projects";
import { Card, CardContent, CardHeader, CardTitle } from "../components/ui/card";
import { Badge } from "../components/ui/badge";
import { Dialog, DialogContent } from "../components/ui/dialog";
import type { FileResponse, FileWithDetections, DetectionResponse } from "../api/types";

// Category colors
const getCategoryColor = (category: string) => {
  switch (category) {
    case "animal":
      return "rgb(34, 197, 94)"; // green
    case "person":
      return "rgb(239, 68, 68)"; // red
    case "vehicle":
      return "rgb(59, 130, 246)"; // blue
    default:
      return "rgb(156, 163, 175)"; // gray
  }
};

// Observation type badge variant
const getObservationBadge = (type: string): { label: string; className: string } => {
  switch (type) {
    case "animal":
      return { label: "Animal", className: "bg-green-100 text-green-800 border-green-200" };
    case "human":
      return { label: "Human", className: "bg-red-100 text-red-800 border-red-200" };
    case "vehicle":
      return { label: "Vehicle", className: "bg-blue-100 text-blue-800 border-blue-200" };
    case "blank":
      return { label: "Blank", className: "bg-gray-100 text-gray-600 border-gray-200" };
    case "unknown":
      return { label: "Unknown", className: "bg-yellow-100 text-yellow-800 border-yellow-200" };
    default:
      return { label: "Unclassified", className: "bg-gray-50 text-gray-500 border-gray-200" };
  }
};

// Format detection label for bounding box overlay
function getDetectionLabel(detection: DetectionResponse): string {
  const categoryLabel = detection.category.charAt(0).toUpperCase() + detection.category.slice(1);
  const confPct = `${(detection.confidence * 100).toFixed(0)}%`;

  if (detection.species && detection.species_confidence != null) {
    const speciesLabel = detection.species.charAt(0).toUpperCase() + detection.species.slice(1);
    const speciesConfPct = `${(detection.species_confidence * 100).toFixed(0)}%`;
    return `${speciesLabel} ${speciesConfPct} · ${categoryLabel} ${confPct}`;
  }

  return `${categoryLabel} ${confPct}`;
}

export default function ImagesPage() {
  const { projectId } = useParams<{ projectId: string }>();
  const [selectedFileId, setSelectedFileId] = useState<string | null>(null);

  // Fetch project for detection threshold
  const { data: project } = useQuery({
    queryKey: ["project", projectId],
    queryFn: () => projectsApi.get(projectId!),
    enabled: !!projectId,
  });

  // Fetch species stats for bar chart
  const { data: speciesStats } = useQuery({
    queryKey: ["species-stats", projectId],
    queryFn: () => projectsApi.getSpeciesStats(projectId!),
    enabled: !!projectId,
  });

  // Fetch files
  const { data: files, isLoading } = useQuery({
    queryKey: ["files", projectId],
    queryFn: () =>
      filesApi.list({
        project_id: projectId,
        limit: 100,
      }),
  });

  // Fetch selected file with detections
  const { data: selectedFile } = useQuery({
    queryKey: ["file", selectedFileId],
    queryFn: () => filesApi.get(selectedFileId!),
    enabled: !!selectedFileId,
  });

  return (
    <div className="p-8 bg-gradient-to-br from-slate-50 to-slate-100 min-h-screen">
      <div className="mx-auto max-w-7xl space-y-6">
        {/* Header */}
        <div>
          <h1 className="text-2xl font-bold tracking-tight">Images</h1>
          <p className="text-sm text-muted-foreground mt-1">
            Browse camera trap images and detections
          </p>
        </div>

        {/* Species bar chart */}
        {speciesStats && speciesStats.length > 0 && (
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground">
                Top Species
              </CardTitle>
            </CardHeader>
            <CardContent className="pb-4">
              <div className="space-y-1.5">
                {speciesStats.map(({ species, count }) => {
                  const maxCount = speciesStats[0].count;
                  const pct = (count / maxCount) * 100;
                  return (
                    <div key={species} className="flex items-center gap-3 text-sm">
                      <span className="w-28 truncate text-right capitalize text-muted-foreground">
                        {species}
                      </span>
                      <div className="flex-1 h-5 bg-muted rounded overflow-hidden">
                        <div
                          className="h-full bg-green-500 rounded"
                          style={{ width: `${pct}%` }}
                        />
                      </div>
                      <span className="w-10 text-right text-muted-foreground tabular-nums">
                        {count}
                      </span>
                    </div>
                  );
                })}
              </div>
            </CardContent>
          </Card>
        )}

        {/* Content card */}
        <Card>
          <CardContent className="p-6">
            {isLoading ? (
              <div className="flex items-center justify-center h-64">
                <div className="text-muted-foreground">Loading images...</div>
              </div>
            ) : !files || files.length === 0 ? (
              <div className="text-center text-muted-foreground py-12">
                <p className="text-lg">No images found</p>
                <p className="text-sm mt-2">
                  Run detection on a deployment to see images here
                </p>
              </div>
            ) : (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
                {files.map((file) => (
                  <ImageCard
                    key={file.id}
                    file={file}
                    onClick={() => setSelectedFileId(file.id)}
                  />
                ))}
              </div>
            )}
          </CardContent>
        </Card>

        {/* Image viewer modal with detections */}
        <Dialog open={!!selectedFileId} onOpenChange={() => setSelectedFileId(null)}>
          <DialogContent className="max-w-6xl max-h-[90vh] overflow-auto">
            {selectedFile && (
              <ImageViewer file={selectedFile} detectionThreshold={project?.detection_threshold ?? 0} />
            )}
          </DialogContent>
        </Dialog>
      </div>
    </div>
  );
}

function ImageCard({ file, onClick }: { file: FileResponse; onClick: () => void }) {
  const timestamp = new Date(file.timestamp).toLocaleString();
  const imageUrl = `http://localhost:8000/api/files/${file.id}/image`;
  const badge = getObservationBadge(file.observation_type);

  return (
    <Card className="overflow-hidden hover:shadow-lg transition-shadow cursor-pointer" onClick={onClick}>
      <div className="aspect-video bg-muted relative">
        <img
          src={imageUrl}
          alt="Camera trap"
          className="w-full h-full object-cover"
          onError={(e) => {
            // Fallback if image fails to load
            (e.target as HTMLImageElement).style.display = "none";
          }}
        />
        {/* Observation type badge */}
        <Badge
          variant="outline"
          className={`absolute top-2 right-2 text-xs ${badge.className}`}
        >
          {badge.label}
        </Badge>
      </div>
      <CardHeader className="p-4">
        <CardTitle className="text-sm truncate" title={file.file_path}>
          {file.file_path.split("/").pop()}
        </CardTitle>
      </CardHeader>
      <CardContent className="p-4 pt-0 space-y-1 text-xs text-muted-foreground">
        <div>{timestamp}</div>
        {file.width_px && file.height_px && (
          <div>
            {file.width_px} x {file.height_px}
          </div>
        )}
        {file.size_bytes && (
          <div>{(file.size_bytes / 1024 / 1024).toFixed(2)} MB</div>
        )}
      </CardContent>
    </Card>
  );
}

function ImageViewer({ file, detectionThreshold }: { file: FileWithDetections; detectionThreshold: number }) {
  const imageUrl = `http://localhost:8000/api/files/${file.id}/image`;
  const timestamp = new Date(file.timestamp).toLocaleString();
  const filteredDetections = file.detections.filter(
    (d) => d.confidence >= detectionThreshold
  );

  return (
    <div className="space-y-4">
      <div className="space-y-2">
        <h2 className="text-xl font-bold">{file.file_path.split("/").pop()}</h2>
        <p className="text-sm text-muted-foreground">{timestamp}</p>
      </div>

      <div className="relative inline-block">
        <img
          src={imageUrl}
          alt="Camera trap"
          className="w-full h-auto"
          id={`image-${file.id}`}
        />

        {/* SVG overlay for bounding boxes */}
        <svg
          className="absolute top-0 left-0 w-full h-full pointer-events-none"
          viewBox={`0 0 ${file.width_px || 1} ${file.height_px || 1}`}
          preserveAspectRatio="none"
        >
          {filteredDetections.map((detection, idx) => {
            const x = detection.bbox_x * (file.width_px || 1);
            const y = detection.bbox_y * (file.height_px || 1);
            const width = detection.bbox_width * (file.width_px || 1);
            const height = detection.bbox_height * (file.height_px || 1);
            const color = getCategoryColor(detection.category);
            const label = getDetectionLabel(detection);

            return (
              <g key={idx}>
                {/* Bounding box rectangle */}
                <rect
                  x={x}
                  y={y}
                  width={width}
                  height={height}
                  fill="none"
                  stroke={color}
                  strokeWidth="3"
                />
                {/* Label background */}
                <rect
                  x={x}
                  y={y - 20}
                  width={Math.max(width, 80)}
                  height="20"
                  fill={color}
                  fillOpacity="0.8"
                />
                {/* Label text */}
                <text
                  x={x + 4}
                  y={y - 6}
                  fill="white"
                  fontSize="14"
                  fontWeight="bold"
                >
                  {label}
                </text>
              </g>
            );
          })}
        </svg>
      </div>

      {/* Detection list */}
      <div className="space-y-2">
        <h3 className="font-semibold">
          Detections ({filteredDetections.length})
        </h3>
        <div className="space-y-1">
          {filteredDetections.map((detection, idx) => (
            <div
              key={idx}
              className="flex items-center justify-between text-sm p-2 rounded border"
              style={{ borderColor: getCategoryColor(detection.category) }}
            >
              <div className="flex items-center gap-2">
                <div
                  className="w-3 h-3 rounded-full"
                  style={{ backgroundColor: getCategoryColor(detection.category) }}
                />
                {detection.species ? (
                  <>
                    <span className="font-medium capitalize">{detection.species}</span>
                    <span className="text-muted-foreground capitalize">
                      ({detection.category})
                    </span>
                  </>
                ) : (
                  <span className="font-medium capitalize">{detection.category}</span>
                )}
              </div>
              <div className="flex gap-2 text-muted-foreground">
                {detection.species_confidence != null && (
                  <span>{(detection.species_confidence * 100).toFixed(1)}%</span>
                )}
                <span>{(detection.confidence * 100).toFixed(1)}%</span>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

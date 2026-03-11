/**
 * Images browser page - displays files with detections
 */

import { useQuery } from "@tanstack/react-query";
import { useParams } from "react-router-dom";
import { useState } from "react";
import { filesApi } from "../api/files";
import { API_BASE_URL } from "../lib/api-client";
import { projectsApi } from "../api/projects";
import { Card, CardContent, CardHeader, CardTitle } from "../components/ui/card";
import { Badge } from "../components/ui/badge";
import { Dialog, DialogContent } from "../components/ui/dialog";
import type { FileResponse, FileWithDetections } from "../api/types";
import { getCategoryColor, getDetectionLabel, getObservationBadge } from "../lib/detection-utils";

export default function ImagesPage() {
  const { projectId } = useParams<{ projectId: string }>();
  const [selectedFileId, setSelectedFileId] = useState<string | null>(null);

  // Fetch project for detection threshold
  const { data: project } = useQuery({
    queryKey: ["project", projectId],
    queryFn: () => projectsApi.get(projectId!),
    enabled: !!projectId,
  });

  // Fetch label stats for bar chart
  const { data: labelStats } = useQuery({
    queryKey: ["label-stats", projectId],
    queryFn: () => projectsApi.getLabelStats(projectId!),
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

        {/* Label bar chart */}
        {labelStats && labelStats.length > 0 && (
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground">
                Top Labels
              </CardTitle>
            </CardHeader>
            <CardContent className="pb-4">
              <div className="space-y-1.5">
                {labelStats.map(({ label, count }) => {
                  const maxCount = labelStats[0].count;
                  const pct = (count / maxCount) * 100;
                  return (
                    <div key={label} className="flex items-center gap-3 text-sm">
                      <span className="w-28 truncate text-right capitalize text-muted-foreground">
                        {label}
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
  const imageUrl = `${API_BASE_URL}/api/files/${file.id}/image`;
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
          style={badge.style}
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
  const imageUrl = `${API_BASE_URL}/api/files/${file.id}/image`;
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
                {detection.label ? (
                  <>
                    <span className="font-medium capitalize">{detection.label}</span>
                    <span className="text-muted-foreground capitalize">
                      ({detection.category})
                    </span>
                  </>
                ) : (
                  <span className="font-medium capitalize">{detection.category}</span>
                )}
              </div>
              <div className="flex gap-2 text-muted-foreground">
                {detection.label_confidence != null && (
                  <span>{(detection.label_confidence * 100).toFixed(1)}%</span>
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

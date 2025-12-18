/**
 * Images browser page - displays files with detections
 */

import { useQuery } from "@tanstack/react-query";
import { useParams } from "react-router-dom";
import { useState } from "react";
import { filesApi } from "../api/files";
import { Card, CardContent, CardHeader, CardTitle } from "../components/ui/card";
import { Dialog, DialogContent } from "../components/ui/dialog";
import type { FileResponse, FileWithDetections, DetectionResponse } from "../api/types";

export default function ImagesPage() {
  const { projectId } = useParams<{ projectId: string }>();
  const [selectedFileId, setSelectedFileId] = useState<string | null>(null);

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
    <div className="p-8 max-w-7xl mx-auto">
      <div className="mb-6">
        <h1 className="text-3xl font-bold">Images</h1>
        <p className="text-muted-foreground mt-2">
          Browse camera trap images and detections
        </p>
      </div>

      {isLoading ? (
        <div className="flex items-center justify-center h-64">
          <div className="text-muted-foreground">Loading images...</div>
        </div>
      ) : !files || files.length === 0 ? (
        <Card>
          <CardContent className="py-12">
            <div className="text-center text-muted-foreground">
              <p className="text-lg">No images found</p>
              <p className="text-sm mt-2">
                Run detection on a deployment to see images here
              </p>
            </div>
          </CardContent>
        </Card>
      ) : (
        <>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
            {files.map((file) => (
              <ImageCard
                key={file.id}
                file={file}
                onClick={() => setSelectedFileId(file.id)}
              />
            ))}
          </div>

          {/* Image viewer modal with detections */}
          <Dialog open={!!selectedFileId} onOpenChange={() => setSelectedFileId(null)}>
            <DialogContent className="max-w-6xl max-h-[90vh] overflow-auto">
              {selectedFile && (
                <ImageViewer file={selectedFile} />
              )}
            </DialogContent>
          </Dialog>
        </>
      )}
    </div>
  );
}

function ImageCard({ file, onClick }: { file: FileResponse; onClick: () => void }) {
  const timestamp = new Date(file.timestamp).toLocaleString();
  const imageUrl = `http://localhost:8000/api/files/${file.id}/image`;

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
            {file.width_px} × {file.height_px}
          </div>
        )}
        {file.size_bytes && (
          <div>{(file.size_bytes / 1024 / 1024).toFixed(2)} MB</div>
        )}
      </CardContent>
    </Card>
  );
}

function ImageViewer({ file }: { file: FileWithDetections }) {
  const imageUrl = `http://localhost:8000/api/files/${file.id}/image`;
  const timestamp = new Date(file.timestamp).toLocaleString();

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
          {file.detections.map((detection, idx) => {
            const x = detection.bbox_x * (file.width_px || 1);
            const y = detection.bbox_y * (file.height_px || 1);
            const width = detection.bbox_width * (file.width_px || 1);
            const height = detection.bbox_height * (file.height_px || 1);
            const color = getCategoryColor(detection.category);

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
                  {detection.category} {(detection.confidence * 100).toFixed(0)}%
                </text>
              </g>
            );
          })}
        </svg>
      </div>

      {/* Detection list */}
      <div className="space-y-2">
        <h3 className="font-semibold">
          Detections ({file.detections.length})
        </h3>
        <div className="space-y-1">
          {file.detections.map((detection, idx) => (
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
                <span className="font-medium capitalize">{detection.category}</span>
                {detection.species && (
                  <span className="text-muted-foreground">
                    - {detection.species}
                  </span>
                )}
              </div>
              <div className="flex gap-2 text-muted-foreground">
                <span>{(detection.confidence * 100).toFixed(1)}%</span>
                {detection.species_confidence && (
                  <span>({(detection.species_confidence * 100).toFixed(1)}%)</span>
                )}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

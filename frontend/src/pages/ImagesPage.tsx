/**
 * Images browser page - displays files with detections
 */

import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { filesApi } from "../api/files";
import { Card, CardContent, CardHeader, CardTitle } from "../components/ui/card";
import type { FileResponse } from "../api/types";

export default function ImagesPage() {
  const [selectedProjectId, setSelectedProjectId] = useState<string | undefined>();

  // Fetch files
  const { data: files, isLoading } = useQuery({
    queryKey: ["files", selectedProjectId],
    queryFn: () =>
      filesApi.list({
        project_id: selectedProjectId,
        limit: 100,
      }),
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
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
          {files.map((file) => (
            <ImageCard key={file.id} file={file} />
          ))}
        </div>
      )}
    </div>
  );
}

function ImageCard({ file }: { file: FileResponse }) {
  const timestamp = new Date(file.timestamp).toLocaleString();

  return (
    <Card className="overflow-hidden hover:shadow-lg transition-shadow">
      <div className="aspect-video bg-muted relative">
        <img
          src={`file://${file.file_path}`}
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

/**
 * Drag-and-drop image picker for project thumbnails.
 *
 * Accepts JPEG/PNG up to 5 MB. Shows preview of selected or existing image.
 */

import { useCallback, useRef, useState } from "react";
import { Upload, X } from "lucide-react";
import { Button } from "../ui/button";

const MAX_SIZE = 5 * 1024 * 1024; // 5 MB
const ACCEPTED_TYPES = ["image/jpeg", "image/png"];

interface ImageDropZoneProps {
  value: File | null;
  existingUrl: string | null;
  onChange: (file: File | null) => void;
  onRemove?: () => void;
}

export function ImageDropZone({
  value,
  existingUrl,
  onChange,
  onRemove,
}: ImageDropZoneProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragOver, setDragOver] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const validate = useCallback((file: File): boolean => {
    if (!ACCEPTED_TYPES.includes(file.type)) {
      setError("Only JPEG and PNG images are accepted");
      return false;
    }
    if (file.size > MAX_SIZE) {
      setError("Image must be smaller than 5 MB");
      return false;
    }
    setError(null);
    return true;
  }, []);

  const handleFile = useCallback(
    (file: File) => {
      if (validate(file)) {
        onChange(file);
      }
    },
    [validate, onChange]
  );

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragOver(false);
      const file = e.dataTransfer.files[0];
      if (file) handleFile(file);
    },
    [handleFile]
  );

  const handleInputChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) handleFile(file);
    },
    [handleFile]
  );

  const previewUrl = value ? URL.createObjectURL(value) : existingUrl;
  const hasImage = !!previewUrl;

  return (
    <div className="space-y-1.5">
      <label className="text-sm font-medium">Project image</label>
      <div
        className={`relative rounded-md border-2 border-dashed overflow-hidden cursor-pointer transition-colors ${!hasImage ? "h-32" : ""} ${
          dragOver
            ? "border-primary bg-primary/5"
            : hasImage
              ? "border-transparent"
              : "border-muted-foreground/25 hover:border-muted-foreground/40"
        }`}
        onDragOver={(e) => {
          e.preventDefault();
          setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={handleDrop}
        onClick={() => inputRef.current?.click()}
      >
        <input
          ref={inputRef}
          type="file"
          accept="image/jpeg,image/png"
          className="hidden"
          onChange={handleInputChange}
        />

        {hasImage ? (
          <img
            src={previewUrl!}
            alt="Project image preview"
            className="w-full object-contain"
          />
        ) : (
          <div className="flex h-full flex-col items-center justify-center gap-2 text-muted-foreground">
            <Upload className="h-6 w-6" />
            <span className="text-xs">
              Drag and drop an image, or click to browse
            </span>
            <span className="text-xs text-muted-foreground/60">
              JPEG or PNG, max 5 MB
            </span>
          </div>
        )}

        {/* Remove button */}
        {hasImage && (onRemove || value) && (
          <Button
            type="button"
            variant="secondary"
            size="icon"
            className="absolute top-2 right-2 h-7 w-7 rounded-full opacity-80 hover:opacity-100"
            onClick={(e) => {
              e.stopPropagation();
              setError(null);
              if (value) {
                onChange(null);
              } else if (onRemove) {
                onRemove();
              }
            }}
          >
            <X className="h-3.5 w-3.5" />
          </Button>
        )}
      </div>

      {error && <p className="text-xs text-destructive">{error}</p>}
    </div>
  );
}

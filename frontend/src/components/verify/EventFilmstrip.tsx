/**
 * Horizontal filmstrip for navigating files within an event.
 *
 * Shows small thumbnails with selection highlight and verified status.
 */

import { useEffect, useRef } from "react";
import { Check } from "lucide-react";
import { cn } from "../../lib/utils";
import type { FileWithDetections } from "../../api/types";

interface EventFilmstripProps {
  files: FileWithDetections[];
  selectedIndex: number;
  onSelectIndex: (index: number) => void;
}

export function EventFilmstrip({
  files,
  selectedIndex,
  onSelectIndex,
}: EventFilmstripProps) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const selectedRef = useRef<HTMLButtonElement>(null);

  // Scroll selected thumbnail into view
  useEffect(() => {
    selectedRef.current?.scrollIntoView({
      behavior: "smooth",
      block: "nearest",
      inline: "center",
    });
  }, [selectedIndex]);

  return (
    <div className="border-t bg-white shrink-0">
      <div
        ref={scrollRef}
        className="flex items-center gap-1.5 px-4 py-2 overflow-x-auto"
      >
        {files.map((file, index) => {
          const thumbnailUrl = `http://localhost:8000/api/files/${file.id}/image`;
          const isSelected = index === selectedIndex;

          return (
            <button
              key={file.id}
              ref={isSelected ? selectedRef : undefined}
              onClick={() => onSelectIndex(index)}
              className={cn(
                "relative shrink-0 w-24 h-16 rounded overflow-hidden border-2 transition-all",
                isSelected
                  ? "border-blue-500 ring-2 ring-blue-200"
                  : "border-transparent hover:border-gray-300"
              )}
            >
              <img
                src={thumbnailUrl}
                alt={`File ${index + 1}`}
                className="w-full h-full object-cover"
                onError={(e) => {
                  (e.target as HTMLImageElement).style.display = "none";
                }}
              />
              {/* Verified checkmark */}
              {file.verified && (
                <div className="absolute top-0.5 right-0.5 bg-green-500 rounded-full p-0.5">
                  <Check className="h-2.5 w-2.5 text-white" />
                </div>
              )}
            </button>
          );
        })}
      </div>
      <div className="text-center text-xs text-muted-foreground pb-1">
        Image {selectedIndex + 1} of {files.length}
      </div>
    </div>
  );
}

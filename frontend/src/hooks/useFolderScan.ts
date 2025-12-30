/**
 * Hook for scanning folders and validating file counts
 *
 * Calls backend API to preview folder contents before adding to queue.
 * Auto-debounces to avoid spamming API during user input.
 */

import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api-client";

interface FolderScanResult {
  image_count: number;
  video_count: number;
  total_count: number;
  gps_location: { latitude: number; longitude: number } | null;
  suggested_site_id: string | null;
  sample_files: string[];
}

/**
 * Scan a folder for camera trap images
 *
 * @param folderPath - Absolute path to folder, or null if not selected
 * @returns File count, loading state, and error if any
 */
export function useFolderScan(folderPath: string | null) {
  return useQuery<FolderScanResult>({
    queryKey: ["folder-scan", folderPath],
    queryFn: async () => {
      if (!folderPath) {
        return {
          image_count: 0,
          video_count: 0,
          total_count: 0,
          gps_location: null,
          suggested_site_id: null,
          sample_files: [],
        };
      }

      const response = await api.get<FolderScanResult>(
        `/api/deployments/preview-folder?path=${encodeURIComponent(folderPath)}`
      );

      return response;
    },
    enabled: !!folderPath, // Only run when folder is selected
    staleTime: 5 * 60 * 1000, // Cache for 5 minutes
    retry: 1, // Only retry once on failure
  });
}

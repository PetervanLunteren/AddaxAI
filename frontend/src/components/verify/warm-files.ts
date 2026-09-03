/**
 * Pull a set of files into the caches a grid paints from: the file
 * detail (boxes, dimensions) into the query cache, and the thumbnail
 * into the browser's HTTP cache (the endpoint serves
 * `Cache-Control: max-age=86400, immutable`, so a warmed thumbnail is
 * free on every later request).
 *
 * One helper for both paginated grids (Files tiles, Counts collages),
 * used to prefetch the NEXT page while the current one is being
 * worked, so clicking Next paints instantly instead of starting cold.
 * The same idea as the viewer's warm-next-file, one level up.
 */

import type { QueryClient } from "@tanstack/react-query";

import { filesApi } from "../../api/files";
import { API_BASE_URL } from "../../lib/api-client";

export function warmFiles(queryClient: QueryClient, fileIds: string[]): void {
  for (const id of fileIds) {
    queryClient.prefetchQuery({
      queryKey: ["file", id],
      queryFn: () => filesApi.get(id),
    });
    // Must match FrameThumbnail's URL exactly; any difference is a
    // different cache entry and warms nothing.
    new Image().src = `${API_BASE_URL}/api/files/${id}/image?size=thumb`;
  }
}

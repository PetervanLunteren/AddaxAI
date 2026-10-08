/**
 * Random photos of animals the AI is sure about.
 *
 * Two shapes from one component: the Overview's single "Gracious random
 * animal" hero, and the Explore tab's wall of six for the chosen taxon.
 * The rule behind the pick lives in one place, the backend
 * (`get_animal_photos`): a confident box, a confident name or a human
 * verdict, a crop that can be drawn. Among those it is random on purpose:
 * confidence saturates near 100%, so "best" would only mean "most
 * recent", and a fresh face on every visit is the point.
 *
 * The query is never cached, so opening the dashboard always rolls again;
 * the refresh button picks again on demand.
 */

import { useQuery } from "@tanstack/react-query";
import { RefreshCw } from "lucide-react";

import { statisticsApi, type DashboardScope } from "../../api/statistics";
import { cn } from "../../lib/utils";
import { Button } from "../ui/button";
import { Card, CardContent } from "../ui/card";
import { DashboardCardHeader } from "./DashboardCardHeader";
import { AnimalPhotoTile } from "./AnimalPhotoTile";

const WALL_SIZE = 6;

interface AnimalPhotosCardProps {
  projectId: string;
  variant: "hero" | "wall";
  scope?: DashboardScope;
  className?: string;
}

export function AnimalPhotosCard({
  projectId,
  variant,
  scope,
  className,
}: AnimalPhotosCardProps) {
  const hero = variant === "hero";
  const limit = hero ? 1 : WALL_SIZE;
  const { data: photos, isLoading, isFetching, refetch } = useQuery({
    // The hero's title promises an animal, so it asks for wildlife; the
    // wall shows whatever Explore has selected.
    queryKey: ["statistics", "animal-photos", projectId, limit, scope, hero],
    queryFn: () => statisticsApi.getAnimalPhotos(projectId, limit, scope, hero),
    staleTime: 0,
    gcTime: 0,
    refetchOnWindowFocus: false,
  });

  // A project with nothing confident yet has no hero; the tiles beside
  // it take the room. The wall stays, because on Explore an empty card
  // says something about the selection.
  if (hero && !isLoading && (photos ?? []).length === 0) return null;

  return (
    <Card className={cn("flex flex-col", className)}>
      <DashboardCardHeader
        title={hero ? "Gracious random animal" : "Random selection"}
        caption={
          hero
            ? "A new one every visit. Click to check its label."
            : "Picked at random from confident detections"
        }
        actions={
          <Button
            variant="ghost"
            size="icon"
            className="h-8 w-8"
            aria-label="Show other photos"
            title="Show others"
            disabled={isFetching}
            onClick={() => void refetch()}
          >
            <RefreshCw className={cn("h-4 w-4", isFetching && "animate-spin")} />
          </Button>
        }
      />
      <CardContent className="flex flex-1 flex-col">
        {isLoading ? (
          <div
            className={cn(
              "animate-pulse rounded-md bg-muted",
              hero ? "aspect-[16/10]" : "aspect-[3/2]",
            )}
          />
        ) : hero ? (
          <AnimalPhotoTile
            projectId={projectId}
            photo={photos![0]}
            size="large"
          />
        ) : photos && photos.length > 0 ? (
          <div className="grid grid-cols-3 gap-2">
            {photos.map((photo) => (
              <AnimalPhotoTile
                key={photo.detection_id}
                projectId={projectId}
                photo={photo}
                size="small"
              />
            ))}
          </div>
        ) : (
          <p className="py-8 text-center text-sm text-muted-foreground">
            No confident animal photos in this selection
          </p>
        )}
      </CardContent>
    </Card>
  );
}

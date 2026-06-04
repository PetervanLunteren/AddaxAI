/**
 * Site info sheet.
 *
 * Read-only side drawer triggered from the Sites table's rows. Shows a
 * small map marker, site metadata, and stats aggregated across every
 * deployment at the site. Buttons: Edit (opens the Add / edit site
 * modal), Show (opens Google Maps in the system browser), Delete
 * (opens the delete confirmation).
 */

import { useQuery } from "@tanstack/react-query";
import { ExternalLink, Pencil, Trash2 } from "lucide-react";

import { sitesApi, type SiteInfo } from "../../api/sites";
import { formatCameraDateTime } from "../../lib/datetime";
import { resolveSpeciesName } from "../../lib/species-name-mode";
import { Button } from "../ui/button";
import {
  IdWithCopy,
  NotSet,
  Row,
  Section,
  formatBytes,
} from "../ui/info-sheet-parts";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from "../ui/sheet";
import { Separator } from "../ui/separator";
import { TagPills } from "../ui/tag-pills";
import { SiteLocationMap } from "./SiteLocationMap";

interface SiteInfoSheetProps {
  siteId: string | null;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  /** Opens the Edit dialog for this site. Parent closes the sheet. */
  onEdit?: () => void;
  /** Opens the Delete confirmation for this site. Parent closes the sheet. */
  onDelete?: () => void;
}

export function SiteInfoSheet({
  siteId,
  open,
  onOpenChange,
  onEdit,
  onDelete,
}: SiteInfoSheetProps) {
  const { data, isLoading, isError } = useQuery({
    queryKey: ["sites", siteId, "info"],
    queryFn: () => sitesApi.getInfo(siteId!),
    enabled: open && !!siteId,
  });

  const googleMapsUrl = data
    ? `https://www.google.com/maps/search/?api=1&query=${data.latitude},${data.longitude}`
    : undefined;

  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent className="w-full overflow-y-auto sm:max-w-xl">
        <SheetHeader>
          <SheetTitle>Site info</SheetTitle>
          <SheetDescription>
            Investigation snapshot for this site.
          </SheetDescription>
        </SheetHeader>

        <div className="mt-4 grid grid-cols-3 gap-2">
          <Button
            type="button"
            variant="outline"
            onClick={onEdit}
            disabled={!onEdit}
          >
            <Pencil className="mr-2 h-4 w-4" />
            Edit
          </Button>
          <Button
            type="button"
            variant="outline"
            onClick={() =>
              googleMapsUrl &&
              window.open(googleMapsUrl, "_blank", "noopener,noreferrer")
            }
            disabled={!googleMapsUrl}
            title="Open in Google Maps"
          >
            <ExternalLink className="mr-2 h-4 w-4" />
            Show
          </Button>
          <Button
            type="button"
            variant="outline"
            className="text-destructive hover:text-destructive"
            onClick={onDelete}
            disabled={!onDelete}
          >
            <Trash2 className="mr-2 h-4 w-4" />
            Delete
          </Button>
        </div>

        {isLoading && (
          <p className="mt-6 text-sm text-muted-foreground">Loading...</p>
        )}
        {isError && (
          <p className="mt-6 text-sm text-destructive">
            Could not load site info.
          </p>
        )}
        {data && <InfoBody info={data} />}
      </SheetContent>
    </Sheet>
  );
}

function InfoBody({ info }: { info: SiteInfo }) {
  const firstLast =
    info.first_captured_at_local && info.last_captured_at_local
      ? `${formatCameraDateTime(info.first_captured_at_local)} to ${formatCameraDateTime(info.last_captured_at_local)}`
      : null;
  const hasTags = Object.keys(info.tags).length > 0;

  return (
    <div className="mt-6 space-y-6 text-sm">
      <SiteLocationMap latitude={info.latitude} longitude={info.longitude} />

      <Section title="Reference">
        <Row label="Site ID" value={<IdWithCopy value={info.site_id} />} />
      </Section>

      <Separator />

      <Section title="Metadata">
        <Row
          label="Coordinates"
          value={`${info.latitude.toFixed(4)}, ${info.longitude.toFixed(4)}`}
        />
        <Row
          label="Elevation"
          value={
            info.elevation_m === null ? (
              <NotSet />
            ) : (
              `${info.elevation_m.toLocaleString()} m`
            )
          }
        />
        <Row label="Habitat" value={info.habitat_type ?? <NotSet />} />
        <Row label="Notes" value={info.notes ?? <NotSet />} />
        <Row
          label="Tags"
          value={hasTags ? <TagPills tags={info.tags} /> : <NotSet />}
        />
      </Section>

      <Separator />

      <Section title="Deployments">
        <Row
          label="Count"
          value={info.deployment_count.toLocaleString()}
        />
        <Row label="First / last file" value={firstLast ?? <NotSet />} />
        <Row
          label="Trap nights"
          value={
            info.trap_nights === null ? (
              <NotSet />
            ) : (
              info.trap_nights.toLocaleString()
            )
          }
        />
      </Section>

      <Separator />

      <Section title="Files">
        <Row
          label="Count"
          value={
            <>
              {info.files.total.toLocaleString()} total
              {" ("}
              {info.files.images.toLocaleString()} images,{" "}
              {info.files.videos.toLocaleString()} videos)
            </>
          }
        />
        <Row label="Size on disk" value={formatBytes(info.total_size_bytes)} />
        <Row
          label="Verified"
          value={
            <span>
              {info.verification.verified.toLocaleString()} /{" "}
              {info.verification.total.toLocaleString()}
              {info.verification.total > 0 && (
                <span className="ml-2 text-muted-foreground">
                  (
                  {Math.round(
                    (info.verification.verified / info.verification.total) *
                      100,
                  )}
                  %)
                </span>
              )}
            </span>
          }
        />
      </Section>

      <Separator />

      <Section title="Observations">
        <Row label="Events" value={info.event_count.toLocaleString()} />
        <Row
          label="Observations (MaxN)"
          value={info.observation_count.toLocaleString()}
        />
        <Row
          label="Rate per 100 trap nights"
          value={
            info.observation_rate_per_100_trap_nights === null ? (
              <NotSet />
            ) : (
              info.observation_rate_per_100_trap_nights.toFixed(2)
            )
          }
        />
      </Section>

      <Separator />

      <Section title="Observation categories">
        <Row
          label="Animal"
          value={info.detection_categories.animal.toLocaleString()}
        />
        <Row
          label="Person"
          value={info.detection_categories.person.toLocaleString()}
        />
        <Row
          label="Vehicle"
          value={info.detection_categories.vehicle.toLocaleString()}
        />
        <Row
          label="Empty"
          value={info.detection_categories.empty.toLocaleString()}
        />
      </Section>

      <Separator />

      <Section title="Top species">
        {info.top_species.length === 0 ? (
          <p className="text-muted-foreground">No classified observations.</p>
        ) : (
          <ol className="space-y-1">
            {info.top_species.map((s, i) => (
              <li
                key={s.label}
                className="flex items-baseline justify-between gap-4"
              >
                <span className="truncate">
                  <span className="tabular-nums text-muted-foreground">
                    {i + 1}.
                  </span>{" "}
                  {resolveSpeciesName(s)}
                </span>
                <span className="tabular-nums">
                  {s.count.toLocaleString()}
                </span>
              </li>
            ))}
          </ol>
        )}
      </Section>
    </div>
  );
}

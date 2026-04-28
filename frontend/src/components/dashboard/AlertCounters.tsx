/**
 * Detection category counters: Animals, People, Vehicles, Empties.
 *
 * Single card with 4 stacked rows, each showing an icon + count.
 */

import { useQuery } from "@tanstack/react-query";
import { PawPrint, User, Car, ImageOff } from "lucide-react";
import { Card, CardHeader, CardTitle, CardContent } from "../ui/card";
import { DashboardAboutPopover } from "./DashboardAboutPopover";
import { statisticsApi } from "../../api/statistics";
import { formatCompact } from "../../lib/utils";

interface AlertCountersProps {
  projectId: string;
  siteIds?: string;
  dateFrom?: string;
  dateTo?: string;
}

interface CounterConfig {
  label: string;
  icon: React.ElementType;
  color: string;
  key: "animal_count" | "person_count" | "vehicle_count" | "empty_count";
}

const COUNTERS: CounterConfig[] = [
  { label: "Animals", icon: PawPrint, color: "#0f6064", key: "animal_count" },
  { label: "People", icon: User, color: "#ff8945", key: "person_count" },
  { label: "Vehicles", icon: Car, color: "#71b7ba", key: "vehicle_count" },
  { label: "Empties", icon: ImageOff, color: "#882000", key: "empty_count" },
];

export const AlertCounters: React.FC<AlertCountersProps> = ({
  projectId,
  siteIds,
  dateFrom,
  dateTo,
}) => {
  const { data, isLoading } = useQuery({
    queryKey: ["statistics", "categories", projectId, siteIds, dateFrom, dateTo],
    queryFn: () => statisticsApi.getDetectionCategories(projectId, siteIds, dateFrom, dateTo),
  });

  return (
    <Card>
      <CardHeader className="pb-2">
        <div className="flex items-center gap-1.5">
          <CardTitle className="text-lg">Observation categories</CardTitle>
          <DashboardAboutPopover
            what={
              <p>
                Captures grouped by what is in them. Each capture (still
                photo or extracted video frame) lands in exactly one
                tile: Animals, People, Vehicles, or Empties. The four
                counts add up to the total captures in the filtered
                view.
              </p>
            }
            how={
              <p>
                Counts use the file's observation type, which encodes
                the priority rule animal &gt; human &gt; vehicle &gt;
                blank. A photo with both an animal and a person is
                counted under Animals only. Numbers are abbreviated
                with k for thousands and M for millions.
              </p>
            }
          />
        </div>
        <p className="text-sm text-muted-foreground">
          Captures grouped by what is in them
        </p>
      </CardHeader>
      <CardContent>
        {isLoading ? (
          <div className="flex items-center justify-center h-32">
            <p className="text-muted-foreground">Loading...</p>
          </div>
        ) : (
          <div className="flex flex-col gap-3">
            {COUNTERS.map(({ label, icon: Icon, color, key }) => (
              <div
                key={key}
                className="flex items-center gap-3 p-3 rounded-lg bg-muted/50"
              >
                <div
                  className="p-2 rounded-full"
                  style={{ backgroundColor: `${color}20` }}
                >
                  <Icon className="h-5 w-5" style={{ color }} />
                </div>
                <div className="flex-1">
                  <p className="text-sm text-muted-foreground">{label}</p>
                </div>
                <p className="text-xl font-bold">
                  {formatCompact(data?.[key] ?? 0)}
                </p>
              </div>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
};

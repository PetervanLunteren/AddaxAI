/**
 * Summary metrics cards for the Insights → Deployment timeline page.
 *
 * Mirrors the dashboard summary-card style (title + big value on the
 * left, tinted icon box on the right) so the two views feel coherent.
 */

import {
  CalendarDays,
  Camera,
  FolderOpen,
  Hourglass,
  MapPin,
} from "lucide-react";
import type { LucideIcon } from "lucide-react";

import type { TimelineMetrics } from "../../api/timeline";
import { Card, CardContent } from "../ui/card";

interface DeploymentTimelineMetricsProps {
  metrics: TimelineMetrics | undefined;
  loading: boolean;
}

const CARD_COLOR = "#0f6064";

function compact(n: number): string {
  if (n >= 1_000_000) return `${+(n / 1_000_000).toFixed(1)}M`;
  if (n >= 10_000) return `${+(n / 1_000).toFixed(1)}K`;
  return n.toLocaleString();
}

function formatMedian(value: number | null): string {
  if (value === null) return "–";
  const rounded = Math.round(value * 10) / 10;
  return `${rounded.toLocaleString()} d`;
}

interface SummaryCard {
  title: string;
  fullTitle?: string;
  value: string;
  icon: LucideIcon;
}

export function DeploymentTimelineMetrics({
  metrics,
  loading,
}: DeploymentTimelineMetricsProps) {
  const cards: SummaryCard[] = [
    {
      title: "Sites",
      value: loading && !metrics ? "..." : compact(metrics?.site_count ?? 0),
      icon: MapPin,
    },
    {
      title: "Deployments",
      value: loading && !metrics ? "..." : compact(metrics?.deployment_count ?? 0),
      icon: FolderOpen,
    },
    {
      title: "Trap nights",
      value: loading && !metrics ? "..." : compact(metrics?.total_trap_nights ?? 0),
      icon: CalendarDays,
    },
    {
      title: "Median length",
      fullTitle: "Median deployment length",
      value:
        loading && !metrics
          ? "..."
          : formatMedian(metrics?.median_deployment_length_days ?? null),
      icon: Hourglass,
    },
    {
      title: "Max cameras",
      fullTitle: "Max concurrent cameras",
      value:
        loading && !metrics
          ? "..."
          : compact(metrics?.max_concurrent_cameras ?? 0),
      icon: Camera,
    },
  ];

  return (
    <div className="grid gap-4 grid-cols-2 md:grid-cols-3 xl:grid-cols-5">
      {cards.map((card) => (
        <Card key={card.title}>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p
                  className="text-sm font-medium text-muted-foreground"
                  title={card.fullTitle}
                >
                  {card.title}
                </p>
                <p className="text-2xl font-bold mt-1">{card.value}</p>
              </div>
              <div
                className="p-3 rounded-lg"
                style={{ backgroundColor: `${CARD_COLOR}20` }}
              >
                <card.icon className="h-6 w-6" style={{ color: CARD_COLOR }} />
              </div>
            </div>
          </CardContent>
        </Card>
      ))}
    </div>
  );
}

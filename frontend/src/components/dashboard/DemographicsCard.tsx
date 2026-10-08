/**
 * Sex, age and behaviour as three stacked bars in one card, ported from
 * AddaxAI Connect.
 *
 * People record these per event on the Counts page and the AI never does,
 * so on most projects every bar reads Unknown. The bars render anyway:
 * an empty bar is a fact about what has been recorded, not a task anyone
 * failed, and the card keeps its shape as data arrives. The caption says
 * where the fields are filled in, which is how people find out they can.
 */

import { useQuery } from "@tanstack/react-query";

import { statisticsApi, type DashboardScope } from "../../api/statistics";
import { toSegments } from "../../lib/demographics";
import { formatShare } from "../../lib/share";
import { OBSERVATION_ATTRIBUTES } from "../../lib/observation-attributes";
import { Card, CardContent } from "../ui/card";
import { DashboardCardHeader } from "./DashboardCardHeader";

interface DemographicsCardProps {
  projectId: string;
  scope: DashboardScope;
}

export function DemographicsCard({ projectId, scope }: DemographicsCardProps) {
  const { data, isLoading } = useQuery({
    queryKey: ["statistics", "demographics", projectId, scope],
    queryFn: () => statisticsApi.getDemographics(projectId, scope),
  });

  return (
    <Card>
      <DashboardCardHeader
        title="Sex, age and behaviour"
        caption={
          data && data.observations > 0
            ? `From ${data.observations.toLocaleString()} observations. Set sex, age and behaviour per event on the Counts page.`
            : "Set sex, age and behaviour per event on the Counts page."
        }
        info={
          <>
            <p>
              Observations split by what was recorded on each count row.
              The AI does not fill these in, so anything not set by a
              person reads Unknown.
            </p>
            <p>
              Models that predict sex or age as part of the species name
              are not counted here; only the fields on the Counts page
              are.
            </p>
          </>
        }
      />
      <CardContent>
        {isLoading ? (
          <p className="text-sm text-muted-foreground">Loading...</p>
        ) : !data || data.observations === 0 ? (
          <p className="text-sm text-muted-foreground">
            No observations in this selection
          </p>
        ) : (
          <div className="space-y-3">
            {OBSERVATION_ATTRIBUTES.map((attr) => (
              <AttributeBar
                key={attr.field}
                label={attr.label}
                segments={toSegments(data[attr.field], attr.options)}
              />
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
}

function AttributeBar({
  label,
  segments,
}: {
  label: string;
  segments: ReturnType<typeof toSegments>;
}) {
  return (
    <div className="space-y-1">
      <div className="flex items-baseline justify-between gap-3 text-sm">
        <span className="font-medium">{label}</span>
        <span className="text-right text-xs text-muted-foreground">
          {segments.map((s) => `${s.label} ${formatShare(s.share)}`).join(" · ")}
        </span>
      </div>
      <div className="flex h-3.5 overflow-hidden rounded bg-muted">
        {segments.map((s) => (
          <div
            key={s.label}
            className={s.color === null ? "bg-muted-foreground/25" : undefined}
            style={{
              width: `${s.share * 100}%`,
              backgroundColor: s.color ?? undefined,
            }}
            title={`${s.label}: ${s.count.toLocaleString()}`}
          />
        ))}
      </div>
    </div>
  );
}

/**
 * The Trend page's table: per species, the mean of the site rates over
 * the whole date range, with its SD and 95% confidence interval.
 *
 * Every site with at least one trap night in the range counts, and a
 * site that never saw the species counts as zero.
 */

import type { RateSummary, SpeciesRate } from "../../api/statistics";
import { resolveSpeciesName } from "../../lib/species-name-mode";
import { formatRate, rateFromPer100, rateLabel } from "../../lib/trap-night-rate";
import { DashboardCardHeader } from "../dashboard/DashboardCardHeader";
import { Card, CardContent } from "../ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "../ui/table";
import { DownloadCsvButton } from "./DownloadCsvButton";

interface SpeciesRateTableProps {
  species: SpeciesRate[];
  isLoading: boolean;
  onDownload: () => void;
}

const MISSING = "–";

const rate = (value: number | null) =>
  value === null ? MISSING : formatRate(rateFromPer100(value));

const interval = ({ ci_low, ci_high }: RateSummary) =>
  ci_low === null || ci_high === null ? MISSING : `${rate(ci_low)} to ${rate(ci_high)}`;

export function SpeciesRateTable({ species, isLoading, onDownload }: SpeciesRateTableProps) {
  return (
    <Card>
      <DashboardCardHeader
        title="By species"
        // Every site with effort counts for every species, so the number
        // of sites is the same on every row: said once, here.
        caption={
          species.length
            ? `Observations ${rateLabel()}, across ${species[0].n_sites} site${species[0].n_sites === 1 ? "" : "s"}, over the whole date range`
            : `Observations ${rateLabel()}, over the whole date range`
        }
        info={
          <>
            <p>
              Per species, the average of the site rates over the whole
              date range, with its standard deviation and 95% confidence
              interval. Every site with at least one trap night counts,
              and a site that never saw the species counts as zero.
            </p>
            <p>
              With one site there is no spread to measure, so SD and the
              interval stay empty.
            </p>
          </>
        }
        actions={<DownloadCsvButton onClick={onDownload} disabled={species.length === 0} />}
      />
      <CardContent>
        {isLoading ? (
          <p className="py-8 text-center text-muted-foreground">Loading...</p>
        ) : species.length === 0 ? (
          <p className="py-8 text-center text-muted-foreground">
            No observations in this selection
          </p>
        ) : (
          <div className="rounded-lg border">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Species</TableHead>
                  <TableHead className="text-right">Mean</TableHead>
                  <TableHead className="text-right">SD</TableHead>
                  <TableHead className="text-right">95% CI</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {species.map((row) => (
                  <TableRow key={`${row.label}-${row.label_taxonomy_id}`}>
                    <TableCell className="font-medium">{resolveSpeciesName(row)}</TableCell>
                    <TableCell className="text-right tabular-nums">{rate(row.mean)}</TableCell>
                    <TableCell className="text-right tabular-nums">{rate(row.sd)}</TableCell>
                    <TableCell className="text-right tabular-nums">{interval(row)}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

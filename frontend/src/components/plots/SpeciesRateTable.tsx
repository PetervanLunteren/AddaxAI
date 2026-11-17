/**
 * The Species rates page's table: per species, the mean of the site
 * rates over the whole date range, with its SD and 95% confidence
 * interval, in the plain box an Insights page puts its one table in.
 *
 * Every site with at least one trap night in the range counts, and a
 * site that never saw the species counts as zero.
 */

import type { RateSummary, SpeciesRate } from "../../api/statistics";
import { resolveSpeciesName } from "../../lib/species-name-mode";
import { formatRate, rateFromPer100, rateLabel } from "../../lib/trap-night-rate";
import {
  Table,
  TableBody,
  TableCaption,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "../ui/table";

interface SpeciesRateTableProps {
  species: SpeciesRate[];
  isLoading: boolean;
}

const MISSING = "–";

const rate = (value: number | null) =>
  value === null ? MISSING : formatRate(rateFromPer100(value));

const interval = ({ ci_low, ci_high }: RateSummary) =>
  ci_low === null || ci_high === null ? MISSING : `${rate(ci_low)} to ${rate(ci_high)}`;

function Message({ children }: { children: React.ReactNode }) {
  return (
    <div className="rounded-lg border bg-card p-16 text-center text-muted-foreground">
      {children}
    </div>
  );
}

export function SpeciesRateTable({ species, isLoading }: SpeciesRateTableProps) {
  if (isLoading) return <Message>Loading...</Message>;
  if (species.length === 0) return <Message>No observations in this selection</Message>;

  // Every site with effort counts for every species, so the number of
  // sites is the same on every row: said once, in the caption.
  const sites = species[0].n_sites;
  return (
    <div className="rounded-lg border bg-card">
      <Table>
        <TableCaption className="mb-4">
          Observations {rateLabel()}, across {sites} site{sites === 1 ? "" : "s"}
        </TableCaption>
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
  );
}

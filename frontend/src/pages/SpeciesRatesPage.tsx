/**
 * Species rates page: the observation rate per species, with its
 * uncertainty.
 *
 * The site is the sampling unit. Each site gets its own rate per
 * species, and the table gives the mean of those rates over the whole
 * date range with its SD and 95% confidence interval (backend:
 * `get_species_rates` in `crud/rate_trend.py`). The same statistic over
 * time is the Trend page.
 *
 * The frame, the filters and the download are `RatePageLayout`.
 */

import { useQuery } from "@tanstack/react-query";
import { useParams } from "react-router-dom";

import { statisticsApi } from "../api/statistics";
import { PlotExplainer } from "../components/plots/PlotExplainer";
import { RatePageLayout } from "../components/plots/RatePageLayout";
import { SpeciesRateTable } from "../components/plots/SpeciesRateTable";
import { SOLLMANN_2013, SWEETAPPLE_HICKLING_2024 } from "../lib/rate-references";
import { DATA_CSV } from "../components/plots/DownloadMenu";
import { saveCsv, speciesRatesCsv } from "../lib/insights-csv";
import { useExploreFilters } from "./dashboard/useExploreFilters";

export function SpeciesRatesPage() {
  const { projectId } = useParams<{ projectId: string }>();
  const id = projectId as string;
  const explore = useExploreFilters(id);
  const { scope } = explore;

  const { data, isLoading, error } = useQuery({
    queryKey: ["statistics", "species-rates", id, scope],
    queryFn: () => statisticsApi.getSpeciesRates(id, scope),
    enabled: explore.ready && !explore.noSiteMatch,
  });

  return (
    <RatePageLayout
      projectId={id}
      title="Species rates"
      subtitle="Observation rate per species, with its uncertainty"
      explore={explore}
      dates={data && { first: data.first_date, last: data.last_date }}
      error={error}
      downloads={
        data?.species.length
          ? [
              {
                label: DATA_CSV,
                save: (fileScope, dates) =>
                  saveCsv("species-rates", speciesRatesCsv(data.species, fileScope), dates),
              },
            ]
          : []
      }
      explainer={
        <PlotExplainer
          plotKey="species-rates"
          what={
            <p>
              One row per species, for the sites and dates chosen above. Each
              site gets its own observation rate for the species. Mean is the
              average of those site rates, SD how much the sites differ, and the
              95% confidence interval how sure the average is. The highest mean
              comes first. Compare a species with itself, between periods or
              sites, rather than with other species: a rate also depends on how
              easy a species is to detect, so a higher rate does not mean more
              animals (Sollmann et al. 2013). To compare two periods, such as
              before and after a control operation, set the date range to one,
              download the table, then set it to the other.
            </p>
          }
          how={
            <p>
              Per site, rate = observations / trap_nights, shown in the unit
              chosen under View &gt; Rate unit. Every site with at least one
              trap night in the date range counts, and a site that never saw the
              species is a rate of zero. Then mean and standard deviation across
              the n sites, and 95% confidence interval = mean ± t × SD / √n,
              with t from Student's distribution for n − 1 degrees of freedom,
              the method of New Zealand's camera trap catch index (Sweetapple &
              Hickling 2024). The lower bound stops at zero. With one site there is no spread to
              measure, so SD and the interval stay empty. Deployments without a
              site are left out. An observation without a species label is
              listed under its detector category: animal, person or vehicle.
            </p>
          }
          references={[SOLLMANN_2013, SWEETAPPLE_HICKLING_2024]}
        />
      }
    >
      <SpeciesRateTable species={data?.species ?? []} isLoading={isLoading} />
    </RatePageLayout>
  );
}

export default SpeciesRatesPage;

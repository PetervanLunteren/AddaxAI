/**
 * Literature behind the Trend and Species rates pages, for their "About
 * this view" references. Each page lists only what its own text cites.
 * DOIs checked against Crossref (2026-11-17).
 */

import type { PlotReference } from "../components/plots/PlotExplainer";

/** The rate as an index of density: it tracked duiker density closely. */
export const ROVERO_MARSHALL_2009: PlotReference = {
  citation:
    "Rovero, F., & Marshall, A. R. (2009). Camera trapping photographic " +
    "rate as an index of density in forest ungulates. Journal of Applied " +
    "Ecology, 46(5), 1011–1017.",
  url: "https://doi.org/10.1111/j.1365-2664.2009.01705.x",
};

/** Why rates of different species cannot be compared as abundance. */
export const SOLLMANN_2013: PlotReference = {
  citation:
    "Sollmann, R., Mohamed, A., Samejima, H., & Wilting, A. (2013). Risky " +
    "business or simple solution: relative abundance indices from " +
    "camera-trapping. Biological Conservation, 159, 405–412.",
  url: "https://doi.org/10.1016/j.biocon.2012.12.025",
};

/** The method both pages compute: mean per-camera index with a t interval. */
export const SWEETAPPLE_HICKLING_2024: PlotReference = {
  citation:
    "Sweetapple, P., & Hickling, G. (2024). Animal pests: camera trap " +
    "catch indices for deer and goats, version 1.0 (DOCCM-7605340). " +
    "Department of Conservation, New Zealand.",
  url: "https://www.doc.govt.nz/globalassets/documents/science-and-technical/inventory-monitoring/camera-trap-catch-indices-for-deer-goats.pdf",
};

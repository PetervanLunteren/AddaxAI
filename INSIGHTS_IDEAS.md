# AddaxAI insights backlog for camera-trap research workflows

## Executive summary

The strongest next additions for AddaxAI are the ones that sit between raw annotation and full statistical modelling: they should be effort-aware, transparent about assumptions, and directly useful for deciding whether a survey is publishable, complete enough, or ready for downstream modelling. Across reviews of camera-trap practice, the most common ecological questions remain presence, relative abundance or detection rate, occupancy, density, richness, and activity, while large-scale syntheses still flag study design, simulation, and power analysis as underused. Put differently: most practitioners still need better exploratory and survey-diagnostics tooling before they need another glossy dashboard. citeturn37view0turn31search2turn49search1turn3search14

The highest-leverage additions, in priority order, are these:

1. **Species accumulation and completeness panel**. This is the fastest way to tell whether a project is still inventorying new species or has plateaued. It answers an immediate field question and is widely used in camera-trap community studies. citeturn49search1turn49search3turn21search2  
2. **Rarefied richness comparison across sites, habitats, and periods**. Raw richness is badly confounded by effort. Rarefaction or coverage standardisation turns a misleading chart into a defensible comparison. citeturn49search7turn21search2turn24search5  
3. **Effort-standardised species summary table**. A sortable table with sites occupied, events per 100 trap-nights, first and last detection, and MaxN summaries would probably become the most-used monthly view in a real workflow. Many papers report exactly these species-level summaries, but most platforms still make users build them externally. citeturn37view0turn49search2turn10search6  
4. **Naive site-use view with explicit warnings**. Researchers constantly want a quick “where did this species turn up?” view before they fit occupancy models. The crucial design requirement is honest labelling: site-use or naive occupancy, not “occupancy”. citeturn16search5turn48search0turn47search11  
5. **Effort-standardised seasonal trend view**. Month or week trends with an effort offset let users spot phenology, disturbance responses, and survey artefacts without pretending that raw counts are abundance. citeturn6view0turn49search2turn49search9  
6. **Detection-history heatmap builder**. This is the bridge to real occupancy modelling. Scientists routinely leave platforms for R because the platform never helps them inspect occasion structure, effort gaps, or species-specific detection histories before export. citeturn21search8turn48search0turn43search11turn4academia11  
7. **Community composition and diversity views**. A careful set of rank-abundance, Hill diversity, composition heatmap, and ordination views would fill genuine white space between simple dashboards and full modelling. camtrapR, camtraptor, and a few specialist tools cover parts of this, but mainstream platforms mostly do not. citeturn42search0turn42search2turn45search6turn9search1

The realistic boundary is important. Scientists still expect to do serious occupancy model specification, dynamic or multi-species occupancy, spatial occupancy, density estimation for unmarked species, spatial capture-recapture, and bespoke co-occurrence work in R or Python, where they can control covariates, priors, diagnostics, and model selection. That expectation is reinforced by the current tool ecosystem itself: entity["organization","Agouti","camera trap platform consortium"] explicitly points users from Camtrap-DP export to camtrapdp, camtraptor, and camtrapR; current entity["organization","TRAPPER","camera trap platform"] documentation emphasises Camtrap-DP export; entity["organization","WildTrax","environmental sensor platform"] exposes R workflows through wildrtrax; and entity["organization","Camelot","camera trap software project"] still presents itself as a first step that “plays nicely” with PRESENCE and camtrapR. citeturn9search1turn12view1turn24search0turn13search5

The main caveat for AddaxAI is therefore not computational, but epistemic. Camera-trap platforms as a class often over-reach when they present raw detection rates as abundance, call quick site summaries “occupancy”, or hide the sensitivity of ecological outputs to survey design, effort imbalance, placement bias, and AI labelling error. The most defensible desktop strategy is to become excellent at effort-aware exploratory analysis, publication readiness, and model-ready exports, while leaving high-assumption inferential models to specialist workflows unless the implementation can expose its assumptions and diagnostics honestly. citeturn3search17turn16search9turn47search2turn48search0turn48academia32

## Platform audit matrix

The table below focuses on the current, publicly documented state of each platform or package that could be verified in this review. `⚠️` means partial support, export-first support, or support that exists mainly through an attached R workflow rather than a first-class in-app insight.

| Platform / package | Survey effort / report | Rate maps / trends | Activity / overlap | Richness / diversity / rarefaction | Occupancy / co-occurrence | Individual ID / SCR | Export / standards |
|---|---|---|---|---|---|---|---|
| Wildlife Insights citeturn6view0 | ✅ trap-day-based summaries and GLM-weighted monthly detection rates | ✅ map view + monthly detection rates | ✅ single-species activity + two-species overlap | ❌ no public evidence of accumulation, rarefaction, or diversity profiles | ❌ no public occupancy tools | ❌ | ⚠️ CSV / PNG export, but not positioned as model-ready occupancy tooling |
| Agouti citeturn7search0turn9search1 | ⚠️ strong management and sequence workflow, limited public evidence of built-in ecological reporting | ❌ | ❌ | ❌ | ❌ in core platform docs | ❌ | ✅ Camtrap-DP export and explicit hand-off to camtrapdp, camtraptor, camtrapR |
| TRAPPER / TRAPPER AI citeturn10search4turn12view1 | ⚠️ GIS mapping and project management, but limited public evidence of ecology reporting | ⚠️ map/search interfaces rather than analytical rate products | ❌ | ❌ | ❌ in public docs | ❌ | ✅ Camtrap-DP export, including event-level aggregation options |
| Camelot citeturn14view0turn13search16 | ✅ reports, independence logic, nocturnal percentage, abundance index | ⚠️ report outputs, not a rich analytical suite | ❌ no documented overlap module | ❌ no documented accumulation or rarefaction | ❌ | ❌ | ✅ export-first design, explicit integration with PRESENCE and camtrapR |
| eMammal citeturn17view0turn18view0 | ⚠️ archive and browse/download workflows | ❌ | ❌ | ❌ | ❌ | ❌ | ⚠️ strong archive/discovery role, weak public evidence of in-app insight suite |
| TrapTagger citeturn20view0 | ⚠️ built-in tools, but public detail is high-level | ✅ maps, heatmaps, graphing | ✅ activity patterns | ✅ “diversity indices” stated publicly | ✅ occupancy stated publicly | ✅ individual identification and SCR stated publicly | ⚠️ rich exports, but public methods detail is thin |
| WildTrax citeturn25view0turn24search0turn22search0 | ✅ strong management, QA, and report/discovery framing | ⚠️ data discovery and reports, limited explicit ecology graphics on public pages | ❌ in core UI docs | ❌ in core UI docs | ⚠️ R workflow via wildrtrax, not an obvious first-class camera UI feature | ❌ | ⚠️ strong export and interoperability story |
| Wild.ID legacy desktop / WildID web citeturn27search14turn26search2 | ⚠️ management, search, edit, export | ❌ | ❌ | ❌ | ❌ | ❌ legacy Wild.ID; ⚠️ tags/individual annotations in newer WildID | ⚠️ export for downstream analysis rather than a broad insight suite |
| camtrapR citeturn10search11turn45search6turn44search18 | ✅ survey report and camera operation tools | ✅ species maps | ✅ single- and two-species activity | ✅ species accumulation and richness maps | ✅ detection histories, single-species and community occupancy | ⚠️ SCR preparation rather than full GUI | ✅ imports Wildlife Insights and Camtrap-DP data |
| camtraptor citeturn42search0turn42search2 | ✅ effort summaries | ✅ maps of deployments and relative abundance / effort | ⚠️ depends on downstream tools more than first-class package identity | ⚠️ overview of species and relative abundance, but not a full diversity suite | ⚠️ detection history helper functions | ⚠️ density helper functions | ✅ Camtrap-DP-native and Darwin Core export |
| overlap citeturn43search18turn43search12 | ❌ | ❌ | ✅ canonical overlap coefficients and bootstrap CIs | ❌ | ❌ | ❌ | ❌ analysis layer only |
| activity citeturn43search13turn43search5turn7search9 | ❌ | ❌ | ✅ canonical activity distributions and activity level | ❌ | ❌ | ❌ | ❌ analysis layer only |
| unmarked citeturn43search11 | ❌ | ❌ | ❌ | ❌ | ✅ standard occupancy and related models | ❌ | ❌ analysis layer only |
| spaceNtime citeturn41search0turn41search9 | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ⚠️ specialist density / abundance methods for unmarked species |
| spOccupancy citeturn4academia11turn43search14 | ❌ | ⚠️ via prediction outputs | ❌ | ⚠️ through modelled richness / occurrence, not exploratory charts | ✅ single-, multi-species, integrated, and spatial occupancy | ❌ | ❌ analysis layer only |
| camtrapdp citeturn42search15turn42search20turn42search19 | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ read, filter, transform Camtrap-DP, including Darwin Core output |

Two practical conclusions follow from that audit. First, mainstream platforms disproportionately cover upload, tagging, activity, and basic detection-rate graphics, while accumulation, completeness, diversity standardisation, temporal turnover, and occupancy-ready detection-history inspection are still patchy or export-first. Second, the packages that ecologists already rely on most for serious inference are still R packages, not all-in-one dashboards. That is the white space AddaxAI can plausibly fill without over-claiming. citeturn29search0turn9search21turn42search0turn10search11

Current authoritative public package pages for **activityGGplot** and for a maintained R package explicitly named **dtt** were not independently verifiable during this pass. I therefore treat activityGGplot as a small plotting helper and “dtt” as the broader detection-time-to-event method family rather than as audited first-class platform rows.

## Prioritised proposal list

### P0

1. **Species accumulation and completeness panel**  
**What the user sees:** a line chart showing cumulative species richness against deployments, sites, or trap-nights, with a confidence ribbon and a companion completeness estimate so users can see whether the survey is still finding new taxa.  
**Scientific purpose:** inventory sufficiency and design feedback. Camera-trap studies routinely use accumulation curves to judge whether the array and duration captured site richness, and Kays et al. showed richness stabilises faster than occupancy or detection-rate precision. citeturn49search1turn49search3turn21search2  
**Computational feasibility:** aggregate `EventObservation` by selected taxon rank using `Detection.label` joined to `LabelTaxonomy`; define sampling unit as deployment, site, site-month, or fixed trap-night block from `Deployment` plus the existing trap-night primitive; accumulate unique taxa per unit.  
**Peer-platform coverage:** strong in camtrapR and community-ecology workflows, largely absent from mainstream platforms. AddaxAI’s differentiator is that it can compute this locally and directly on event-ready data. citeturn45search6turn10search11turn6view0  
**Standards / conventions:** use sample-based accumulation, default `random` permutations, show 95% ribbon, and pair the curve with incidence-based completeness such as Chao2 or first-order jackknife. Follow Camtrap-DP deployment and observation semantics so exported results are reproducible. citeturn42search9turn4search6turn24search5  
**Edge cases / caveats:** highly clustered or trail-biased arrays can plateau for the wrong reason; completeness of commonly detected species is not completeness of the full community.  
**Effort rating:** 2. **Value rating:** 5. **Priority tier:** P0.

2. **Rarefied richness comparison across sites, habitats, and periods**  
**What the user sees:** side-by-side richness comparisons where every bar or point is standardised to the same sample coverage or effort, so a lightly sampled site is not being compared unfairly with a heavily sampled one.  
**Scientific purpose:** fair comparison of communities across habitats, sites, or seasons. Camera-trap studies that compare richness across placements or vegetation types are vulnerable to effort imbalance unless richness is standardised. citeturn49search7turn21search2turn45search17  
**Computational feasibility:** use deployment- or site-based incidence matrices built from `EventObservation`, grouping by `Site.habitat_type`, `Site.tags`, date bins from `Event.event_start_local`, and effort from trap-nights.  
**Peer-platform coverage:** weak in general platforms; stronger in R via iNEXT or vegan-style workflows. AddaxAI can make this a first-class, publishable comparison rather than an export chore. citeturn9search1turn42search0  
**Standards / conventions:** default to incidence-based rarefaction/extrapolation standardised by sample coverage, not raw sample count; show observed richness, coverage-standardised richness, and uncertainty. Align taxon roll-up with `LabelTaxonomy`.  
**Edge cases / caveats:** free-text habitat categories can be messy; if habitat labels are inconsistent, the app should force explicit grouping review before plotting.  
**Effort rating:** 3. **Value rating:** 5. **Priority tier:** P0.

3. **Effort-standardised species summary table**  
**What the user sees:** a sortable table for the selected filters with rows for taxa and columns such as events per 100 trap-nights, sites occupied, deployments occupied, total independent events, median MaxN, first detection, last detection, and diel class.  
**Scientific purpose:** species-level reporting is one of the most common outputs in camera-trap papers, but users usually assemble it manually. The table becomes the operational backbone for field decisions, manuscripts, and publication checks. citeturn37view0turn49search2turn10search6  
**Computational feasibility:** taxa from `Detection.label` plus `LabelTaxonomy`; effort from trap-nights; occupancy counts from distinct `site_id` or deployment IDs with at least one `EventObservation`; MaxN from `EventObservation.max_n`; temporal summaries from `Event.event_start_local`.  
**Peer-platform coverage:** partial everywhere, but usually split across exports, reports, and dashboards. AddaxAI can differentiate by making the table truly effort-aware and publication-ready. citeturn6view0turn14view0turn25view0  
**Standards / conventions:** default rate denominator = 100 trap-nights; show taxonomic roll-up controls; provide CSV exactly keyed to Camtrap-DP observation and deployment identifiers for reproducibility. citeturn42search9turn24search5turn42search19  
**Edge cases / caveats:** this is not abundance; label the main rate column as “event rate” or “detections per 100 trap-nights”, never abundance.  
**Effort rating:** 2. **Value rating:** 5. **Priority tier:** P0.

4. **Site-use map and table with explicit naive-occupancy labelling**  
**What the user sees:** a map and companion table showing the proportion of active sites or deployments where a species was detected at least once, optionally with Wilson intervals and minimum-effort filters.  
**Scientific purpose:** a rapid, honest spatial summary before model fitting. Occupancy is central in camera-trap ecology, but the app should reserve the word “occupancy” for actual models and call this quick view “naive occupancy” or “site-use”. citeturn16search5turn48search0turn43search11  
**Computational feasibility:** binary detection by site or deployment from `EventObservation`; denominator from active deployments or sites with at least a configurable trap-night minimum; spatial layer from `Site.latitude` and `Site.longitude`.  
**Peer-platform coverage:** some platforms show detections on maps, but few foreground the distinction between detected-use and modelled occupancy. That honesty is a design advantage. citeturn6view0turn10search6turn42search0  
**Standards / conventions:** default minimum effort threshold = 10 trap-nights; let users choose site- and deployment-level aggregation; apply sensitive-location rounding in export paths where needed. citeturn18view0turn42search19turn24search5  
**Edge cases / caveats:** low-detectability species will be underestimated; trail placement and habitat openness can alter apparent site-use strongly.  
**Effort rating:** 2. **Value rating:** 5. **Priority tier:** P0.

5. **Effort-standardised weekly and monthly trend view**  
**What the user sees:** a line or dot-and-whisker chart of events per 100 trap-nights over weeks or months, with filters for species, taxon roll-up, site subsets, habitat, and human-pressure strata.  
**Scientific purpose:** seasonality, phenology, disturbance response, and survey artefact detection. Wildlife Insights already exposes a similar monthly detection-rate graphic, which is a good sign that this is part of the practical minimum. citeturn6view0turn49search2turn9search2  
**Computational feasibility:** count independent events from `Event` and `EventObservation`; use trap-night or active-day effort from the existing primitive; time bins from `Event.event_start_local`; optionally stratify by `Site.habitat_type` or human/vehicle event rate.  
**Peer-platform coverage:** Wildlife Insights has this at species level; most others do not document equivalent trend analysis prominently. AddaxAI can extend it with local-first filtering and habitat or disturbance faceting. citeturn6view0turn25view0  
**Standards / conventions:** fit Poisson or negative-binomial rate models with log-effort offset for smoothed estimates; default to event counts rather than raw file counts; display raw data and uncertainty together.  
**Edge cases / caveats:** users will overread noise if sample sizes are low. Include a warning when monthly support is below a defensible threshold.  
**Effort rating:** 3. **Value rating:** 5. **Priority tier:** P0.

6. **Effort-standardised richness map**  
**What the user sees:** a map of observed or coverage-standardised richness by site or deployment, with optional small multiples by season or habitat.  
**Scientific purpose:** richness is still one of the main camera-trap reporting targets, but raw richness maps are effort traps; a standardised richness map is far more defensible than a bare unique-species count. citeturn16search14turn49search7turn42search0  
**Computational feasibility:** distinct taxa from `EventObservation` aggregated by site or deployment; denominator from trap-nights or coverage standardisation; coordinates from `Site`.  
**Peer-platform coverage:** camtrapR exposes richness maps; mainstream platforms rarely do this carefully. This is clear white space. citeturn45search6turn10search6turn6view0  
**Standards / conventions:** default to observed richness only above a minimum effort threshold; offer coverage-standardised richness as the preferred mode; use generalized coordinates or fuzzing when preparing public outputs. citeturn18view0turn24search5turn42search19  
**Edge cases / caveats:** users may still read richness as “biodiversity quality” even when placement, local habitat, and taxonomic detectability differ.  
**Effort rating:** 2. **Value rating:** 4. **Priority tier:** P0.

### P1

7. **Stratified activity explorer**  
**What the user sees:** the existing activity and overlap plots, but faceted by season, month, habitat, site group, or disturbance class, so users can compare how a species shifts its schedule between contexts.  
**Scientific purpose:** activity shifts are one of the most common behavioural outputs from camera traps, particularly in human-wildlife interaction and predator-prey work. citeturn47search10turn49search2turn45search15  
**Computational feasibility:** reuse `File.captured_at_local` or `Event.event_start_local`; group by selected strata from `Site.habitat_type`, tags, dates, or human/vehicle rate classes.  
**Peer-platform coverage:** Wildlife Insights covers one- and two-species activity, but not this kind of rich faceting in public docs. AddaxAI’s advantage is exploratory flexibility. citeturn6view0  
**Standards / conventions:** keep current circular KDE approach for continuity, but add explicit support for clock-time and solar-time facets and report sample size per stratum.  
**Edge cases / caveats:** activity estimates can be sensitive to data filtering. Since independence filtering may bias activity inference, the UI should make the chosen timestamp mode explicit. citeturn47search2turn47search7  
**Effort rating:** 3. **Value rating:** 4. **Priority tier:** P1.

8. **Activity level estimate**  
**What the user sees:** a scalar estimate of proportion of the day active, with bootstrap uncertainty and optional comparison across strata.  
**Scientific purpose:** Rowcliffe’s activity-level estimator is increasingly relevant, including as an ingredient in camera-based density methods, and it gives users more than a pretty diel curve. citeturn7search9turn43search13turn43search5  
**Computational feasibility:** derive times from `File.captured_at_local` or `Event.event_start_local`; calculate activity level per taxon and per selected subset.  
**Peer-platform coverage:** usually available only through the activity package or specialist analytical workflows, not in mainstream platforms. citeturn43search13turn43search5  
**Standards / conventions:** use `activity::fitact` defaults, bootstrap the activity estimate, and expose solar-time transformation.  
**Edge cases / caveats:** the Rowcliffe estimator assumes peak activity equals full availability, which may fail for some predators or high-latitude conditions. The warning belongs beside the metric, not hidden in docs. citeturn7search9  
**Effort rating:** 2. **Value rating:** 4. **Priority tier:** P1.

9. **Human and vehicle disturbance profile**  
**What the user sees:** maps and trends for human and vehicle event rates, plus optional overlap plots between focal taxa and anthropogenic activity.  
**Scientific purpose:** bycatch on humans and vehicles is scientifically useful, not just a privacy problem. Large coordinated surveys explicitly use temporal outputs to study human-wildlife interactions. citeturn49search2turn35search15  
**Computational feasibility:** derive human and vehicle events from `File.observation_type` and `Detection.category`; compute event rates and activity curves from `Event` timestamps and trap-nights; compare against selected taxa.  
**Peer-platform coverage:** some systems filter or hide human imagery, but few foreground human-pressure as an ecological covariate in the insight layer. citeturn22search0turn18view0  
**Standards / conventions:** privacy defaults should honour the user’s sensitive-data policies; in exported public views, blur or suppress imagery while retaining derived counts where allowed.  
**Edge cases / caveats:** human event rate is only a proxy for disturbance, and it is confounded by access, roads, and camera placement.  
**Effort rating:** 3. **Value rating:** 4. **Priority tier:** P1.

10. **Rank-abundance curve**  
**What the user sees:** a Whittaker-style plot ranking taxa by effort-standardised event rate or incidence across the selected subset.  
**Scientific purpose:** a compact community summary that reveals dominance, evenness, and the long tail of rarely detected taxa. Camera-trap community studies regularly compare composition and relative commonness across habitats or placements. citeturn9search2turn49search7  
**Computational feasibility:** taxa from `EventObservation`; rank by events per 100 trap-nights, site-use, or coverage-standardised incidence; allow taxonomic roll-up.  
**Peer-platform coverage:** essentially absent from mainstream camera-trap platforms; more likely to be recreated manually in R. That makes it a good white-space candidate. citeturn42search0turn10search11  
**Standards / conventions:** default ordering by event rate with log-scale option; show observed species count and effort denominator; make metric label explicit so no one mistakes it for abundance.  
**Edge cases / caveats:** for very sparse projects this becomes unstable and visually noisy; warn when support is low.  
**Effort rating:** 2. **Value rating:** 4. **Priority tier:** P1.

11. **Hill diversity profile**  
**What the user sees:** a curve or panel comparing effective diversity across q = 0, 1, and 2 by habitat, site group, or time period.  
**Scientific purpose:** this is a more informative community comparison than raw richness because it distinguishes richness from evenness and dominance. It is common in biodiversity analysis, but underprovided in camera-trap software. citeturn21search2turn9search2  
**Computational feasibility:** incidence or effort-standardised event-rate matrix from `EventObservation` by chosen grouping; compute Hill numbers on standardised input.  
**Peer-platform coverage:** no strong evidence in the major camera-trap platforms reviewed, though TrapTagger publicly claims diversity indices. citeturn20view0  
**Standards / conventions:** default to q = 0, 1, 2; standardise by sample coverage before comparison; display observed and standardised values side by side.  
**Edge cases / caveats:** results are sensitive to the chosen sampling unit and taxonomic roll-up. The UI should make both explicit.  
**Effort rating:** 3. **Value rating:** 4. **Priority tier:** P1.

12. **Community composition heatmap**  
**What the user sees:** a species-by-site or species-by-habitat matrix coloured by incidence, site-use, or effort-standardised rate, with clustering options.  
**Scientific purpose:** ecologists often want to see which taxa separate habitats or site clusters before fitting a model. This is one of the most practically useful community overviews and is still oddly absent from most platforms. citeturn9search2turn42search0  
**Computational feasibility:** build a matrix from `EventObservation` aggregated by site, habitat, deployment, or month; values can be binary incidence, event rate, or mean MaxN.  
**Peer-platform coverage:** camtraptor and generic R workflows can support the data wrangling, but mainstream platform docs do not foreground this view. citeturn42search0turn42search2  
**Standards / conventions:** default to row-normalised incidence with optional clustering; hide cells below a minimum effort threshold; allow export of the underlying matrix.  
**Edge cases / caveats:** colour scaling can exaggerate sparse differences. Provide a clear legend and allow binary mode.  
**Effort rating:** 2. **Value rating:** 4. **Priority tier:** P1.

13. **Beta-diversity ordination and clustering**  
**What the user sees:** an NMDS or PCoA ordination of sites or deployments, plus an optional dendrogram, coloured by habitat, region, or period.  
**Scientific purpose:** community separation and turnover are standard ecological questions, and camera-trap studies increasingly compare composition across habitat types and placement strategies. citeturn9search2turn49search7  
**Computational feasibility:** derive species-by-site incidence or rate matrix from `EventObservation`; Bray-Curtis for rates or Jaccard/Sørensen for incidence; plot with site metadata from `Site`.  
**Peer-platform coverage:** effectively external-R territory today. That makes an exploratory, non-inferential version attractive inside AddaxAI. citeturn42search0turn10search11  
**Standards / conventions:** default to Jaccard on incidence and Bray-Curtis on standardised rates; show stress or variance explained; offer the matrix export.  
**Edge cases / caveats:** ordinations are exploratory and very sensitive to sparse matrices; do not imply significance unless formal tests are run and reported separately.  
**Effort rating:** 3. **Value rating:** 3. **Priority tier:** P1.

14. **MaxN group-size explorer**  
**What the user sees:** distributions of event-level MaxN for a selected species, broken out by site, habitat, season, or time of day.  
**Scientific purpose:** camera-trap studies often report group size or count summaries for gregarious species, and AddaxAI is unusually well positioned because MaxN already exists in the data model. citeturn21search4turn16search14  
**Computational feasibility:** use `EventObservation.max_n`, grouped by taxon and the selected strata; timestamps from `Event.event_start_local`; spatial link from deployment to site.  
**Peer-platform coverage:** not strongly surfaced in the reviewed platform docs despite being straightforward and useful.  
**Standards / conventions:** default to median, IQR, and count of contributing events; allow histogram and boxplot modes; keep species roll-up explicit.  
**Edge cases / caveats:** MaxN is conservative and setting-dependent; it is informative about groups seen, not a population estimate.  
**Effort rating:** 1. **Value rating:** 4. **Priority tier:** P1.

15. **Detection-history heatmap builder**  
**What the user sees:** a site-by-occasion matrix for a selected species, with detected, non-detected, and insufficient-effort cells, plus occasion-level effort bars and export.  
**Scientific purpose:** this is the missing bridge from platform to occupancy modelling. Scientists need to inspect effort gaps, occasion definition, and sparse sites before they trust a model. citeturn21search8turn48search0turn43search11turn4academia11  
**Computational feasibility:** use `Event.event_start_local` and `Deployment.start_date_local`/`end_date_local` to discretise occasions; aggregate to binary detection from `EventObservation`; store effort from active days in each occasion using the existing operability primitive.  
**Peer-platform coverage:** camtrapR and camtraptor expose detection-history generation, but most full platforms stop at export. AddaxAI could make the intermediate QA step visible. citeturn21search8turn42search2  
**Standards / conventions:** default occasion length = 7 days, configurable; show NA where active effort falls below a threshold; allow binary and count output formats.  
**Edge cases / caveats:** occasion choice affects autocorrelation and parameter estimates. The plot should surface that sensitivity, not hide it. citeturn48search0turn48search2  
**Effort rating:** 3. **Value rating:** 5. **Priority tier:** P1.

### P2

16. **Temporal turnover plot**  
**What the user sees:** similarity between adjacent months, seasons, or years, plotted through time for the full community or a selected taxonomic group.  
**Scientific purpose:** community change through time is often more informative than a stack of monthly richness values, especially for long-running arrays. citeturn49search2turn31search2  
**Computational feasibility:** build monthly or seasonal incidence matrices from `EventObservation` and compute Jaccard or Sørensen similarity between adjacent periods; effort threshold from trap-nights.  
**Peer-platform coverage:** essentially absent from current platforms.  
**Standards / conventions:** default to adjacent-period Sørensen on incidence with minimum period effort filters.  
**Edge cases / caveats:** turnover can reflect seasonality, camera downtime, or changing placement as much as ecological change.  
**Effort rating:** 2. **Value rating:** 3. **Priority tier:** P2.

17. **Pairwise association matrix**  
**What the user sees:** a matrix of pairwise positive or negative association scores between taxa across sites or site-occasions, with significance or uncertainty clearly secondary to effect size.  
**Scientific purpose:** researchers are interested in potential coexistence structure, shared habitat use, and potential disturbance or predation signatures, but often have to build quick exploratory matrices themselves. citeturn49search2turn43search10  
**Computational feasibility:** create species-by-site-occasion detection histories from `EventObservation`, then compute pairwise φ coefficients, odds ratios, or probabilistic co-occurrence departures.  
**Peer-platform coverage:** weak to non-existent in reviewed platforms.  
**Standards / conventions:** default to exploratory, not inferential, output; let users choose site or occasion scale; provide matrix export for downstream modelling.  
**Edge cases / caveats:** this is not evidence of biological interaction. Shared avoidance of roads, cameras, or habitat can produce the same pattern. Put that warning on the plot.  
**Effort rating:** 3. **Value rating:** 3. **Priority tier:** P2.

18. **Time-to-first-detection and cumulative detection curve**  
**What the user sees:** for a focal species, a curve showing how quickly active deployments accumulate first detections, with censored deployments retained.  
**Scientific purpose:** design feedback and detectability intuition. Time-to-event thinking is increasingly used to complement hierarchical models and to estimate abundance for some unmarked-species workflows. citeturn40search2turn41search0turn49search3  
**Computational feasibility:** for each deployment, compute time from `Deployment.start_date_local` to first matching `Event.event_start_local`; censor at deployment end if not detected; optionally group by habitat or camera model.  
**Peer-platform coverage:** almost absent in current platforms, despite clear value for survey planning.  
**Standards / conventions:** default to Kaplan-Meier style cumulative detection plot with median time to detection where estimable; allow export of per-deployment latency table.  
**Edge cases / caveats:** absence of early detection is not absence of the species; strong seasonality can dominate the curve.  
**Effort rating:** 2. **Value rating:** 3. **Priority tier:** P2.

19. **Moon-phase activity explorer**  
**What the user sees:** activity curves or event-rate summaries split by lunar phase or moon illumination bins for nocturnal species.  
**Scientific purpose:** many camera-trap studies ask whether nocturnal taxa shift behaviour with moonlight, and the view is attractive because it requires no new field hardware.  
**Computational feasibility:** derive lunar phase and illumination from `Site.latitude`, `Site.longitude`, and `File.captured_at_local` or `Event.event_start_local`; group events into lunar bins.  
**Peer-platform coverage:** not visible in the reviewed platform docs.  
**Standards / conventions:** default to coarse bins such as new, quarter, full; use solar-time display for nocturnal taxa; keep it explicitly exploratory.  
**Edge cases / caveats:** without cloud cover, canopy openness, and weather, moonlight is only a rough proxy. This is useful for hypothesis generation, not strong inference.  
**Effort rating:** 2. **Value rating:** 2. **Priority tier:** P2.

20. **AI ecological-threshold sensitivity explorer**  
**What the user sees:** a small-multiples view showing how species richness, site-use, event rate, or activity metrics change as the classification confidence threshold is moved.  
**Scientific purpose:** recent work shows that AI training decisions and label noise can affect downstream ecological metrics such as richness, occupancy, and activity. A platform that uses AI should let users inspect ecological robustness, not only confusion matrices. citeturn48academia32turn4academia13  
**Computational feasibility:** re-compute chosen metrics using `Detection.label_confidence`, `Detection.verified`, and `File` / `Event` timestamps under a grid of cut-offs; compare to the verified subset where available.  
**Peer-platform coverage:** the reviewed docs focus on AI processing, not on ecological sensitivity to confidence thresholds. This is high-value white space. citeturn20view0turn5search8turn22search0  
**Standards / conventions:** default threshold grid 0.20 to 0.95 by 0.05; show threshold-specific support and verified-subset performance side by side.  
**Edge cases / caveats:** verified data are often non-randomly sampled; the plot should disclose verification coverage by taxon and stratum.  
**Effort rating:** 3. **Value rating:** 3. **Priority tier:** P2.

### P3

21. **Single-species occupancy model lite**  
**What the user sees:** a guided occupancy analysis for one species at a time, with a deliberately small set of covariates, coefficient plots, fitted site probabilities, and goodness-of-fit diagnostics.  
**Scientific purpose:** occupancy is one of the central analytical targets in camera-trap research, and users do want it in-platform, but only if the app stays honest about autocorrelation, occasion definition, and covariate limits. citeturn16search5turn43search11turn48search0  
**Computational feasibility:** detection histories from proposal 15; state covariates from `Site.elevation_m`, `Site.habitat_type`, coordinates, and selected tags; observation covariates from occasion-level effort and optional human activity.  
**Peer-platform coverage:** TrapTagger publicly claims occupancy; camtrapR, unmarked, and wildrtrax provide the practical current standard through R. citeturn20view0turn10search11turn43search11turn24search0  
**Standards / conventions:** use a small unmarked-style `occu` workflow or equivalent; default to 7-day occasions; expose detection and state formulas explicitly; include an autocorrelation warning informed by Goldstein et al. citeturn48search0turn48search2  
**Edge cases / caveats:** this is where platforms often overreach. If diagnostics, formula transparency, and occasion QA are not excellent, this should remain export-to-R instead.  
**Effort rating:** 4. **Value rating:** 3. **Priority tier:** P3.

22. **Spatial or community occupancy wizard**  
**What the user sees:** an expert-only workflow for spatial single-species occupancy or community occupancy, with prediction maps and coefficient summaries.  
**Scientific purpose:** these are powerful and increasingly expected in professional analysis, but they are specialist tools, not research-grade minimum. spOccupancy exists precisely because users need a flexible Bayesian framework for spatial and multi-species occupancy. citeturn4academia11turn43search14  
**Computational feasibility:** same detection-history machinery as above; coordinates from `Site`; simple covariates from elevation, habitat text, and tags; observation covariates from effort.  
**Peer-platform coverage:** essentially R-only in practice today. That is a warning sign, not just an opportunity.  
**Standards / conventions:** if ever implemented, treat this as a thin guided layer on top of a documented model engine, with model formula, priors, diagnostics, posterior predictive checks, and export of inputs and outputs.  
**Edge cases / caveats:** free-text habitat variables, sparse species, and unmodelled placement bias will undermine results quickly. For a single-user desktop app, this is aspirational rather than immediate.  
**Effort rating:** 5. **Value rating:** 2. **Priority tier:** P3.

## Anti-patterns to avoid

**Raw counts or raw richness without an effort denominator.** A site with more active cameras or more trap-nights will almost always look richer or busier. This is precisely why Wildlife Insights expresses monthly detection rates per 100 trap-days, and why accumulation and rarefaction logic is so important. citeturn6view0turn49search3turn21search2

**Calling detection rate or photographic rate “abundance”.** The methods literature has been warning about this for years. Burton et al. emphasised that photographic rates require strong assumptions about detectability; Parsons et al. explicitly asked whether occupancy or detection rates reflect deer density; Broadley et al. showed that density-dependent space use affects interpretation of detection rates; and Harmange et al. argue that ignoring imperfect detection should not be an option for unmarked-species abundance analysis. citeturn3search17turn16search9turn47search11

**Heatmaps and maps that ignore placement bias.** Trail versus random placement can change relative abundance indices and activity estimates, and habitat-specific placement effects can alter capture rates, composition, and diversity inferences. Any map or table that seems quantitative should remind users that camera placement is part of the signal. citeturn16search14turn7search3

**Using time-to-independence filters for activity analyses without warning.** Peral et al. show that applying independence filters to activity estimation can remove large amounts of data and distort activity and overlap estimates; their review found that nearly 70% of studies using camera traps for activity had applied such filters. AddaxAI’s current event-cluster logic is useful for many summaries, but activity modules should make this limitation explicit. citeturn47search2turn47search3turn47search7

**Using the word occupancy for naive detection summaries.** Occupancy models exist to handle imperfect detection; a map that marks “detected at least once” is not the same thing. Goldstein et al. further show that camera-trap occupancy work must reckon with autocorrelation and occasion choice explicitly. citeturn48search0turn48search2turn43search11

**Pairwise association graphics that imply ecological interaction.** Co-detection and shared site use can be driven by habitat, human access, or camera placement. If you ship a pairwise association matrix, it must be labelled exploratory and paired with an export path to proper modelling. citeturn43search10turn4academia11

**AI outputs that look exact while hiding ecological sensitivity.** Recent work shows that label noise and training-data reduction can alter richness, occupancy, and activity estimates. Confidence thresholds and verification coverage are not implementation trivia; they are part of the ecological result. citeturn48academia32turn4academia13

## Future data capture

The following views would be valuable, but the current AddaxAI schema does not support them without new fields or calibration workflows.

**Density from random encounter, time-to-event, or space-to-event models.**  
**Minimum new fields:** calibrated camera detection zone radius and angle, animal-to-camera distance or digitised positions, camera height and orientation, and for REM-style workflows a defensible species movement-speed input or link to an external trait table.  
**Payoff:** scientifically stronger estimates for unmarked species than raw detection rates, but still assumption-heavy. spaceNtime and camera-trap distance sampling show what a serious implementation requires. citeturn41search0turn40search6turn41search12

**Spatial capture-recapture and robust individual-level reporting.**  
**Minimum new fields:** `individual_id`, flank or viewpoint metadata, individual-match confidence, and ideally sex and age class.  
**Payoff:** defensible density estimation and re-identification workflows for patterned species, plus real individual histories instead of ad hoc notes. TrapTagger’s public materials underscore how important an explicit individual-ID layer becomes once you support it. citeturn20view0

**Demographic and behaviour dashboards.**  
**Minimum new fields:** per-detection or per-event sex, age class, behaviour, and perhaps reproductive status.  
**Payoff:** unlocks recruitment, sex-ratio, and behaviour summaries that many field teams want but cannot reconstruct later from free text.

**Habitat selection and occupancy with serious covariates.**  
**Minimum new fields:** standardised habitat covariates rather than free text, trail or road proximity, bait or lure flag, camera placement protocol, vegetation obstruction, and site-level GIS covariates.  
**Payoff:** far more interpretable occupancy and site-use modelling. Free-text `habitat_type` is useful for first-pass grouping but too weak for rigorous ecological inference. citeturn16search14turn7search3

**Weather and environmental response analyses.**  
**Minimum new fields:** linked rainfall, temperature, and perhaps moonlight-through-cloud or NDVI covariates, either stored or resolvable from external data.  
**Payoff:** real phenology and weather-response analysis instead of attributing every seasonal shift to biology.

**Camera-performance calibration and detection-bias correction.**  
**Minimum new fields:** trigger mode, delay, sensitivity, placement photos, view-zone calibration images, obstruction score, and maintenance logs.  
**Payoff:** stronger comparability across deployments and better correction for camera-specific detectability, which the methods literature treats as a real source of bias. citeturn47search11turn16search14

**Temporal sequence and behaviour-from-video analytics.**  
**Minimum new fields:** explicit trigger burst or sequence IDs, clip-level tracks, and ideally per-clip behaviour annotations.  
**Payoff:** better transition from “files” to ethologically meaningful events, especially for feeding, vigilance, and social behaviour.

## Sources

All web pages below were accessed on 23 April 2026.

**Methods, reviews, and research papers**

- Burton et al., *Wildlife camera trapping: a review and recommendations for linking surveys to ecological processes*. citeturn3search17  
- Sollmann, *A gentle introduction to camera-trap data analysis*. citeturn3search14  
- Blount et al., *Review: COVID-19 highlights the importance of camera traps for wildlife conservation research and management*. citeturn37view0  
- Bruce et al., *Large-scale and long-term wildlife research and monitoring using camera traps: a continental synthesis*. citeturn31search2turn31search6  
- Kays et al., *An empirical evaluation of camera trap study design: how many, how long and when?* citeturn49search1turn49search3turn49search20  
- Tanwar et al., *Camera trap placement for evaluating species richness, abundance, and activity*. citeturn16search14turn49search7  
- Broekhuis et al., *Location, location, location: habitat-specific differences of camera trap placement on species detections and capture rates*. citeturn7search3  
- Rowcliffe et al., *Quantifying levels of animal activity using camera-trap data*. citeturn7search9  
- Ridout and Linkie, *Estimating overlap of daily activity patterns from camera trap data*. citeturn47search10  
- Peral et al., *The inappropriate use of time-to-independence biases estimates of activity patterns of free-ranging mammals derived from camera traps*. citeturn47search2turn47search3  
- Goldstein et al., *Guidelines for estimating occupancy from autocorrelated camera trap detections*. citeturn48search0turn48search2  
- Harmange et al., *Consequences of modelling procedures on detecting environmental effects on species distribution from camera-trap data*. citeturn47search11  
- Pantazis et al., *Deep learning-based ecological analysis of camera trap images is impacted by training data quality and size*. citeturn48academia32turn4academia13  
- Vélez et al., *An evaluation of platforms for processing camera-trap data using artificial intelligence*. citeturn29search0turn29search4  
- Cordier et al., *Camera trap research in Africa: a systematic review*. citeturn29search17  

**Platform and package documentation**

- Wildlife Insights, official analytics documentation. citeturn6view0turn5search0  
- Agouti, official introduction and export documentation. citeturn7search0turn9search1  
- TRAPPER, current documentation and Camtrap-DP export docs. citeturn10search4turn12view1  
- Camelot, official documentation and reports page. citeturn13search1turn14view0turn13search16  
- eMammal, home and browse-data pages. citeturn17view0turn18view0  
- TrapTagger, official product page. citeturn20view0  
- WildTrax, platform site, FAQ, and wildrtrax package site. citeturn25view0turn22search0turn24search0  
- Wild.ID legacy GitHub repository and WildID web user guide. citeturn27search14turn26search2  
- camtrapR package and visualisation/dashboard documentation. citeturn10search11turn45search6turn44search18  
- camtraptor package and reference index. citeturn42search0turn42search2  
- activity package documentation. citeturn43search13turn43search5  
- overlap package documentation. citeturn43search18turn43search12  
- unmarked package documentation. citeturn43search11  
- spOccupancy package and vignettes. citeturn4academia11turn43search14  
- spaceNtime package paper. citeturn41search0turn41search9  
- camtrapdp package documentation. citeturn42search15turn42search20  

**Standards and best-practice pages**

- Camtrap-DP official standard page. citeturn42search9  
- Camtrap-DP standards paper. citeturn4search6turn40search15turn46search8  
- entity["organization","GBIF","global biodiversity infrastructure"] best-practices guide for camera-trap data. citeturn24search5  
- camtraptor documentation for Camtrap-DP to Darwin Core transformation. citeturn42search19  

**Research context and community workflow sources**

- WildCo / Beirne, *An Introduction to Camera Trap Data Analysis in R*, chapters on community composition, occupancy, activity, and density. citeturn21search2turn24search7turn40search8turn41search8  
- WILDLABS discussion on exploratory camera-trap tooling. citeturn9search19turn42search18
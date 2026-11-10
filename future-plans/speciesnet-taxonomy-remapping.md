# SpeciesNet taxonomy remapping (restrict_to_taxa_list)

Plan for a future change. Not scheduled. Captured here so we can pick it up without re-deriving the design. None of this is built. It records what MegaDetector's `restrict_to_taxa_list` actually does, how it differs from our current species selection, and why a naive "just change the save/load file format" is a trap.

## Why

Beta user Dan Morris (a MegaDetector / SpeciesNet maintainer) asked for this. He says it is how he *always* runs SpeciesNet for users now, instead of the standard country geofence, because it produces much cleaner results. It is also the recommended way to "get the most out of vanilla SpeciesNet" without fine-tuning: rather than the geofence (a fixed list of taxa allowed per country / US state), you hand SpeciesNet a small CSV that **remaps its ~3000 raw outputs to your own label set**.

Two things he asked for, which are really one feature:
1. A "load .csv" button that loads the same CSV he already uses for `restrict_to_taxa_list`.
2. Changing our existing species-selection **save/load** format from our JSON to that CSV.

## What restrict_to_taxa_list actually does (verified from source, not the doc summary)

Source: `megadetector/postprocessing/classification_postprocessing.py`, and Morris's skill doc (links below). Corrected after an initial wrong reading:

- **It is a remapping, not an allowlist.** Each CSV row maps a SpeciesNet taxon to an **output label you choose**. Columns: `latin` (required, the SpeciesNet taxon at its most-specific token, e.g. `panthera pardus`, `panthera`, `felidae`, `carnivora`, `mammalia`), `common` (required, the output name you want), optional `original_common` (alternate output names). The `common` value is an arbitrary string.
- **It folds UP by default.** For each prediction it starts at the species and walks up (genus, family, order, class) to the first taxon present in the CSV, then emits that row's output label. `allow_walk_down` is an option, **default False**, that then collapses to a unique allowed child if there is exactly one.
- **Predictions matching no row fall back to `animal`.**
- **Custom labels are the point.** Because `common` is free-form, you can **rename** (`latin=aepyceros` -> `common=impala`) and **regroup** (map `accipitridae`, `falconidae`, `strigiformes` all -> `bird of prey`; or "all other birds" -> `other bird`; or list only the site's species and let the rest fold up).

So it collapses SpeciesNet to exactly the label set the user wants, with sensible fold-up for everything else.

## How this differs from what we have today

Our species selection is **binary include / exclude** on the model's own class names, plus the geofence, plus our roll-up:

- `Project.excluded_classes` (a list of deselected model-class names). Included = all model classes minus excluded.
- Geofence-aware taxonomic roll-up in `backend/app/ml/postprocessing.py` + `taxonomic_rollup.py`: an excluded or out-of-range top-1 rolls **up** to the nearest allowed ancestor; if none, it drops to `animal`.
- Save/load in `frontend/src/components/taxonomy/SpeciesSelectionModal.tsx` (`handleSave` / `applyLoadedFile`) writes JSON `{ model_id, included: [class names] }` and matches by exact string equality against the model's classes.

The **direction is the same** (we already fold up). What we lack is the remapping itself:
- We keep the taxon name as-is. We cannot **rename** a class or **regroup** several taxa into one custom label ("bird of prey", "other bird").
- Selection is one bit per class (in / out). A mapping is one target label per source taxon.

So `restrict_to_taxa_list` is a **custom taxonomy-mapping capability**, not a richer file format for the same include/exclude switch.

## The trap: a format swap alone is half a feature

If we just made save/load read the CSV as an include-list (treat the `latin` column as "allowed", ignore the output labels) and kept our roll-up, Dan gets file interoperability but **loses the relabeling and regrouping that is the entire value**. The output would still be our coarser roll-up taxa, not his curated labels. So the CSV format is only worth doing together with the mapping behavior.

## Design sketch (for when this is scoped properly)

The mechanism is not foreign to us (we already fold up), so the real work is the mapping layer and the custom labels flowing downstream:

- **Data model:** a per-project mapping (source SpeciesNet taxon -> target label), not just `excluded_classes`. Target labels can be custom strings absent from the model taxonomy. `label_taxonomy` already has `is_custom`, `common_name`, `scientific_name`, and the `taxon_*` ranks (`backend/app/models/label_taxonomy.py`), so custom targets have a natural home; the mapping itself is new.
- **Pipeline:** apply the mapping when assigning `Detection.label`, in the same phase as the current rollup (`postprocessing.py`). Two realistic options: call MegaDetector's `restrict_to_taxa_list` directly (it lives in the MD postprocessing package we already ship), or reimplement the up-walk-with-relabel against our taxonomy. Decide how it coexists with geofence rollup (replace it, or offer "geofence" vs "custom mapping" as project modes).
- **Downstream:** custom labels must flow through the label filter tree, dashboards, per-class performance, and exports. Labels not in the model taxonomy need taxonomy rows (rank + names) so the hierarchy and Camtrap-DP export still work. This is the bulk of the surface area.
- **CSV I/O:** parse `latin,common,original_common`, match `latin` against our `label_taxonomy` ranks, build the source-taxon -> target-label map. Save writes the same shape. Matching is by scientific name / rank token, which we have.
- **Authoring is out of scope:** Morris already ships a skill and a web app (`speciesnet-taxonomy-mapper`) that generate these CSVs, so we only need to **consume** them. We do not need to build a mapping-authoring UI (YAGNI); a "load mapping .csv" plus our existing tree editor is enough to start.

## Effort and when to do it

Medium-to-large, and a proper design round, not a tweak. The mapping model + pipeline mode is the core; the long tail is custom labels through filters / stats / exports. Worth doing: it is expert-validated, it is SpeciesNet's own recommended path, and it is a better fit than the geofence for anyone with a known species list. Schedule when custom-label support across the app is worth the plumbing; until then the geofence + include/exclude covers the common case.

## References

- restrict_to_taxa_list docs: https://megadetector.readthedocs.io/en/latest/postprocessing.html#megadetector.postprocessing.classification_postprocessing.restrict_to_taxa_list
- Taxonomy-mapping skill (CSV format): https://github.com/agentmorris/agentmorrispublic/blob/main/skills/speciesnet-taxonomy-mapping/SKILL.md
- Web taxonomy mapper: http://dmorris.net/speciesnet-taxonomy-mapper
- SpeciesNet pro tips: http://lila.science/speciesnet-pro-tips

## Key files (today's behaviour this would extend or replace)

| File | Role |
|------|------|
| `frontend/src/components/taxonomy/SpeciesSelectionModal.tsx` | JSON save/load (`handleSave` / `applyLoadedFile`) to swap for CSV |
| `frontend/src/components/taxonomy/LabelSelectionField.tsx` | Country + edit-species entry point |
| `backend/app/models/project.py` | `excluded_classes` (would gain a mapping alongside it) |
| `backend/app/ml/postprocessing.py`, `taxonomic_rollup.py` | Current geofence-aware roll-up; where a mapping mode would apply |
| `backend/app/models/label_taxonomy.py` | Where custom target labels + ranks would live (`is_custom`) |
| `backend/app/ml/geofence.py` | The geofence this offers an alternative to |

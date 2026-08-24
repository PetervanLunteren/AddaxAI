# Build a taxonomy.csv for a model

Use this when a classification model needs its `taxonomy.csv`, the file that maps
each model class to its taxonomy ranks so the UI can build the hierarchical
filter tree (class > order > family > genus > species). Every cls model needs
one. It is the same GBIF-resolve job each time, with a few traps.

The producer is `backend/scripts/resolve_taxonomy_gbif.py`. It runs at staging
time on a dev machine only; the GBIF calls are never an app dependency.

Output columns (`WEBUI_HEADER`): `model_class,class,order,family,genus,species,variant`.
All values lowercase, empty when unknown, `species` is the epithet only
(`Panthera pardus` gives `pardus`). `variant` is one optional rank below species
for models whose classes sit deeper than a species ("adult", "juvenile", "male");
empty for everything else.

## Steps

1. **Get the exact class names, in model order.** For a checkpoint that stores
   them, read the labels (e.g. `torch.load(...)["labels"]`). Otherwise take them
   from the author's class list. The `model_class` column must match what the
   model actually emits, because `Detection.label` joins to
   `label_taxonomy.name` by lowercase string equality.

2. **Author the input CSV.** One row per class. `model_class` is never matched
   against GBIF (too many collisions: "serval" is also a beetle genus), so give
   the resolver something to resolve:
   - a species class: `scientific_name` as the binomial, e.g.
     `elephant,Loxodonta cyclotis`.
   - a group class (bird of prey, rodent, monkey, hornbill, passeriform): a
     higher taxon name, e.g. `rodent,Rodentia`, `bird of prey,Accipitriformes`.
     GBIF fills the hierarchy down from there.
   - reuse a sibling model's resolved rows for overlapping classes rather than
     re-resolving.
   - a variant class (age, sex, morph of one species): the species binomial in
     `scientific_name` plus the variant in a `variant` column, e.g.
     `red fox adult,Vulpes vulpes,adult`. The variant passes through verbatim;
     GBIF resolves only the species part.
   ```
   model_class,scientific_name
   eastern chipmunk,Tamias striatus
   rodent,Rodentia
   ```

3. **Run the resolver.**
   ```
   cd backend && source venv/bin/activate
   python scripts/resolve_taxonomy_gbif.py <input.csv> ~/AddaxAI/models/cls/<id>/taxonomy.csv
   ```
   It prints a "N rows need review" summary and a final count of rows that got a
   class.

4. **Review, and fix any empty rows.** Empty rows (no class at all) fall under an
   `__other__` node in the tree, so fix them: usually the name was too vague
   (a bare genus or subfamily that GBIF could not place). Give a species-level
   name and re-run. The review flags to eyeball, but usually leave:
   - synonym notes ("GBIF calls X a synonym of Y"): GBIF's preferred name; fine.
   - HIGHERRANK notes: a group correctly resolved to an order/family.

## Gotchas

- **`model_class` is never sent to GBIF.** Always supply `scientific_name` (or a
  GBIF usage key, or hand-authored rank columns).
- **Groups map to higher taxa, not species.** Leave the finer columns empty; the
  resolver does that from an order/family name.
- **Hand-supplied rank columns are treated as overrides.** If you put
  `class/order/...` values in the input, that row is written verbatim and GBIF is
  not consulted. Use this to pin a value GBIF gets wrong.
- **The reptilia rule (`ORDER_AS_CLASS`).** GBIF's backbone has no Reptilia
  class; it puts Squamata / Testudines / Crocodylia / Rhynchocephalia at class
  rank. The script folds those back under `class=reptilia`, order = the original.
  Same idea for amphibian orders. This is deliberate and matches shipped models.
- **Non-label classes get empty taxonomy by design.** `bait`, `blank`, `empty`,
  `false detection`, `none`, `vide` (see `app.ml.label_exclusion`) are never
  loaded to the DB, so an empty row for them is correct.
- **The species column is the epithet only**, not the full binomial. The script
  handles this; do not "fix" it to a binomial.
- **`variant` is never sent to GBIF** and never marks a row hand-written. It is
  a free-form passthrough, lowercase. A model whose CSV uses it needs
  `min_app_version` set to the first release with variant support, or older
  apps show two classes with the same scientific name.
- **Peter reviews before upload.** Taxonomy is an ecologist's call; surface the
  flagged rows and the group mappings you chose.

## Key files

- `backend/scripts/resolve_taxonomy_gbif.py` (the producer, read its docstring)
- `backend/scripts/generate_taxonomy_csv.py` (SpeciesNet-labels variant)
- `backend/app/ml/taxonomy_db.py` (`populate_taxonomy_from_csv`, how the file is
  loaded into `label_taxonomy`)
- `DEVELOPERS.md` section "Label taxonomy and the hierarchical filter tree"

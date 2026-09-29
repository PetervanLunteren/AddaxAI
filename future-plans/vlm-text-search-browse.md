# VLM text search ("Browse") spike

A throwaway experiment, captured so the learnings survive even though the code will not be merged. The work lives on branch `browse-vlm-search` (forked off `labels-observations-rebuild`); it is not going into main. This note records what we built, what we tried, what it cost on a laptop, and what a real version would need.

## Why we tried it

The normal camera-trap pipeline is detector + classifier. We already added a DINOv2 similarity-embedding pass that powers the Labels page (sort crops into visual buckets for faster cleanup). The question: can a vision-language model add **free-text search** over a project's detections, e.g. "an animal lying down", "a grazing animal with its head down", "an animal with horns"? That is a capability the detector + classifier cannot give, because it is about pose, behaviour, and scene, not species identity.

## The one fact that shaped everything

DINOv2 is an image-only encoder. Its vectors cannot be queried with text. Free-text search needs a **joint image-text model** (the CLIP family: SigLIP, SigLIP2, BioCLIP, OpenCLIP) where images and text land in the same vector space, so you embed the query string and rank crops by cosine similarity to it.

We picked **SigLIP2** over BioCLIP. BioCLIP is biology-specialised and great for taxonomic/species text, but we already have species labels; the interesting queries are behaviour and attributes, where a general web-trained model is stronger. OpenCLIP was the lighter fallback but SigLIP2 needed was available through `transformers` directly.

## What is reusable (this was the pleasant surprise)

Roughly 90% of the machinery already existed. The only genuinely new pieces were the VLM itself, a text-encode path, an on-demand build job, and a page.

Reused unchanged:
- `detection_embeddings` is keyed `(detection_id, embedding_model_id)` (`backend/app/models/detection_embedding.py`), so a second model's vectors coexist with DINOv2's with **no schema change**. Vectors are float16 bytes + a precomputed `l2_norm` for cosine.
- `build_embedding_input()` and `save_embeddings_to_db()` in `backend/app/ml/embedding_utils.py` are model-agnostic.
- The standalone-subprocess pattern: heavy ML (torch / faiss / transformers) runs in `env-addaxai-base` as a subprocess that reads SQLite directly and emits NDJSON; the main backend never imports those libs. `backend/app/services/label_service.py` and `backend/app/ml/inference/similarity_script.py` were the template.
- `faiss-cpu` and `torch` were already in the env. FAISS `IndexFlatIP` over L2-normalized vectors gives cosine.
- Frontend `CropGrid`/`CropCard` render a `DetectionSummary[]` by `crop_url`; the insights-page pattern and the route/sidebar/breadcrumb hooks were all in place.

## What we built (the spike)

- `backend/app/ml/inference/siglip_script.py`: standalone subprocess, two modes. `--mode embed` crops -> float16 npz; `--mode search` text query -> FAISS -> NDJSON rows shaped exactly like similarity search (so the frontend reuses the same grid). Imports crop/loader helpers from the sibling `embedding_script.py` / `similarity_script.py` (same dir, same env).
- `backend/app/workers/browse_worker.py`: on-demand "build index" job, mirrors the re-embed worker, skips already-embedded crops.
- `backend/app/services/browse_service.py`, `backend/app/api/routers/browse.py` (build-index / stats / search), `backend/app/api/schemas/browse.py`.
- `frontend/src/pages/BrowsePage.tsx` + `frontend/src/api/browse.ts`: a Browse nav item with a build-index gate, a search box, an "Index N more" top-up button, results in `CropGrid`. Progress is tracked by polling `/browse/stats`, not the WebSocket job channel.
- One env change: `transformers` + `sentencepiece` + `protobuf` added to the three `addaxai-base` `environment.yml` files. SigLIP2 weights auto-download to the HF cache on first use.

## What broke and what we changed (the iteration log)

1. **`transformers` 5.x changed the return type.** `get_image_features` / `get_text_features` now return a `BaseModelOutputWithPooling`, not a tensor. Added a small `_pool()` shim that reads `.pooler_output` if present, else the raw tensor, so the script works across 4.x and 5.x.
2. **The job never started.** `ws_manager.register_start` parks the worker until the frontend opens a WebSocket and sends `{"type":"ready"}` (the deployment-analysis handshake via `useTaskProgress`). Our stats-polling page never did that, so the job sat pending and was dropped after 5 minutes; the spinner span forever. Switched to `asyncio.create_task` to start it immediately.
3. **It would have frozen the backend.** The worker called the blocking SigLIP subprocess directly on the event loop. Offloaded it with `run_in_executor` so stats polling stays responsive.
4. **Build button vanished after the first index.** The gate was tied to `indexed === 0`, so newly-added detections could not be topped up. Re-tied it to `missing > 0` ("Index N more"), and fixed the polling-stop to handle the offline-files case (some crops can never be indexed, so `missing` may never hit 0).
5. **Model swap base -> so400m.** Bumped from `siglip2-base-patch16-224` (768-dim) to `siglip2-so400m-patch14-384` (1152-dim) for better quality. Because the two dims cannot share a FAISS index, this needed a **new `embedding_model_id`** (`SIGLIP2-SO400M`), not a reuse. The old 768-dim rows are then orphaned (harmless, but dead weight).

## What it cost on the laptop (the real deliverable)

Measured on an Apple Silicon Mac (MPS), embedding detection crops:

| model | dim | input | embed throughput | model load (cached) |
|---|---|---|---|---|
| `siglip2-base-patch16-224` | 768 | 224px | ~18 crops/s | ~5 s |
| `siglip2-so400m-patch14-384` | 1152 | 384px | ~2.3 crops/s | ~8 s |

- First-ever model load is much slower because it downloads weights (base ~73 s; so400m is ~1.6 GB).
- Indexing 965 crops: base ~2 min, so400m ~6.5 min. so400m is roughly **8x slower** to index.
- The worker loads the model **once per deployment**, so multi-deployment projects pay the load repeatedly.
- Search latency is dominated by a **per-query model reload** (~8-18 s), because the search subprocess is short-lived.

## What we learned

- **It is feasible and fast enough on a laptop** for small/medium projects, and the "add a model" workflow is genuinely light because the embedding infra generalises. Indexing a few thousand crops is a minute or two with the base model.
- **Crops vs full frames is THE design fork.** We embedded tight detection crops. That works well for animal appearance and pose ("lying down", "with horns", "stripes"), but scene/context/lighting prompts ("at night", "in water", "a herd") are weak because the context is cropped away. Full-frame embedding is the obvious next experiment and probably the bigger win for the open-ended "vibe" queries that motivated this.
- **SigLIP scores are not calibrated.** Absolute cosines sit in a low band (~0.05-0.16) regardless of match quality; only the **ranking** is meaningful. Any UX must be top-k or rank-based, not an absolute-threshold cutoff.
- **Model size is a real tradeoff.** so400m discriminates better (e.g. "a giraffe" returns all giraffes; base mixed in look-alikes) but is ~8x slower and heavier. Pick per project scale.
- **VLMs cannot count.** "exactly three animals" does not work; counting stays with the detector/MaxN path.
- **Per-query model reload is the main UX wart.** Short-lived subprocesses are clean and match the existing pattern, but reloading SigLIP every query is the thing that would feel slow in real use.

## What a real (non-spike) version would need

Roughly in order of leverage:

1. **A resident model**, so search does not reload weights per query. Either a long-lived embedding worker process, or fold it into a persistent service. This is the single biggest UX fix.
2. **One subprocess per job, not per deployment**, to amortise the model load when indexing.
3. **A full-frame embedding option** (likely a per-feature choice, maybe both crop and frame indexes), since that is where the scene/behaviour queries get good.
4. **Model in the catalog/manifest** with the existing HF downloader, instead of `transformers` auto-download, so it works offline and goes through the normal model-add flow.
5. **Filters in the Browse UI.** We sent none; the 20k per-search cap and large projects need site/species/date filters first (the plumbing already accepts `LabelFilters`).
6. **Index lifecycle.** A delete/rebuild endpoint; clean handling of dimension changes on model swap (the orphaned-old-vectors problem); and a decision on whether the index builds automatically after analysis or stays on-demand.
7. **Decide the home and the framing.** It lived as a projects-mode "Browse" page here. Confirm that is where it belongs and how it relates to the Labels similarity sort (both are embedding-driven; they could share more).

## Pointers

- Branch: `browse-vlm-search`.
- Core files: `backend/app/ml/inference/siglip_script.py`, `backend/app/workers/browse_worker.py`, `backend/app/services/browse_service.py`, `backend/app/api/routers/browse.py`, `frontend/src/pages/BrowsePage.tsx`, `frontend/src/api/browse.ts`.
- Related: the DINOv2 similarity infra in `backend/app/ml/inference/similarity_script.py` and `backend/app/ml/embedding_utils.py` is the foundation this reused.
- Models tried: `google/siglip2-base-patch16-224` (768-dim), `google/siglip2-so400m-patch14-384` (1152-dim). Alternatives not tried: BioCLIP (species-specialised), OpenCLIP (lighter).

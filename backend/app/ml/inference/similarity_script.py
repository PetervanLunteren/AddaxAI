"""
Standalone similarity script — runs as subprocess in env-addaxai-base.

Performs greedy nearest-neighbor similarity sort or FAISS nearest-neighbor
search on detection embeddings stored in a SQLite database. No app imports
— uses only stdlib plus numpy and faiss-cpu.

Usage:
    python similarity_script.py \
        --db-path /path/to/addaxai.db \
        --project-id <uuid> \
        --operation sort \
        --params '{"filters": {...}}'

Output: NDJSON event stream on stdout. Each line is one of:
    {"type": "progress", "phase": "load|sort|neighbors", "done": N, "total": M}
    {"type": "result", "detections": [...], "total_detections": N,
     "total_loaded": N, "total_matching": N}
    {"type": "error", "message": "..."}

Exit code: 0 on success (with a "result" line), non-zero on unhandled
errors (an "error" line is still emitted before exit). Streaming via
NDJSON keeps progress reporting on a single channel; the parent service
just relays lines to the HTTP client.

Following CONVENTIONS.md: crash early and loudly, no silent failures.
"""

import argparse
import json
import sqlite3
import sys
from collections import Counter
from collections.abc import Callable
from typing import Any

import numpy as np

# Two independent limits (see the sort load paths):
#   MAX_EMBEDDINGS — the FAISS budget: the most embedding vectors held in
#     RAM and walked. Similarity walks every embedded detection; event sort
#     walks one representative per event. This is the sharp memory/compute
#     limit, so it stays tight.
#   MAX_DETECTIONS — the payload/grid budget: the most detections returned
#     to (and rendered by) the frontend. No FAISS cost, so it can be larger.
# For similarity the two coincide (it loads FROM embeddings); for event /
# time sorts the detection budget is what usually binds.
MAX_EMBEDDINGS = 20_000
MAX_DETECTIONS = 50_000

# How often to emit progress events during long loops. Tuned for ~50ms
# between updates at typical scales: 500 rows per SQL emission, 200
# steps per O(n) loop. Smaller intervals would flood the wire; larger
# would make the bar feel stuck.
PROGRESS_LOAD_EVERY = 500
PROGRESS_LOOP_EVERY = 200

# Minimum share of usable neighbours that must carry the candidate
# label before we surface it as a suggestion. Without this, a plurality
# of 3/10 wins and produces noisy suggestions; 0.6 means a clear
# majority is required. Tunable, start point set from beta feedback.
NEIGHBOR_MAJORITY_FRACTION = 0.6


def _emit_event(event: dict[str, Any]) -> None:
    """Write one NDJSON line to stdout. Flushed so the parent sees it live."""
    sys.stdout.write(json.dumps(event, default=str) + "\n")
    sys.stdout.flush()


def _emit_progress(phase: str, done: int, total: int) -> None:
    _emit_event({"type": "progress", "phase": phase, "done": done, "total": total})

# ── SQL ──────────────────────────────────────────────────────────────────

# Detection + file + event columns shared by both load paths. `d.id AS
# detection_id` equals the embedding table's detection_id via the join,
# so the two queries return the same column set (the embedding path just
# adds the vector). Selecting the same columns lets `_row_to_meta` and
# `_build_query` serve both without branching.
_DETECTION_COLUMNS = """
       d.id AS detection_id,
       d.label, d.label_taxonomy_id, d.label_confidence, d.scientific_name,
       d.common_name,
       d.confidence, d.category,
       d.verified, d.suggestion_dismissed,
       d.classification_method, d.file_id, d.frame_number,
       d.bbox_x, d.bbox_y, d.bbox_width, d.bbox_height,
       f.deployment_id, f.captured_at_local, f.width_px, f.height_px,
       f.flagged AS file_flagged, f.favorited AS file_favorited,
       s.name AS site_name,
       -- Event membership for the "By event" sort and event dividers.
       -- event_files is many-to-many in the schema but one-event-per-file
       -- in practice (clustering deletes+recreates a deployment's events,
       -- assigning each file once). Correlated subqueries keep the result
       -- strictly one row per detection, which the greedy walk and counts
       -- depend on; a LEFT JOIN could multiply rows if that invariant ever
       -- broke. Indexed by idx_event_files_file.
       (SELECT ef.event_id FROM event_files ef WHERE ef.file_id = f.id LIMIT 1)
           AS event_id,
       (SELECT ef.sequence_number FROM event_files ef WHERE ef.file_id = f.id LIMIT 1)
           AS event_sequence,
       (SELECT e.event_start_local FROM event_files ef
        JOIN events e ON e.id = ef.event_id
        WHERE ef.file_id = f.id LIMIT 1) AS event_start_local"""

# File / deployment / site joins + the project scope, shared by all three
# queries. Ends on a WHERE clause so `_build_query` can append " AND ..."
# filter clauses.
#
# The best-frame gate is part of the scope, not a user-facing filter. A
# video's detections sit on every sampled frame, but only the best frame is
# written to disk as a JPEG, so a detection on any other frame has no image
# to crop. `crop_service` would answer with the best frame cropped at a
# bbox belonging to a different moment: a picture of the wrong place, and
# of nothing at all once the animal has moved on. Those detections stay in
# the database (per-frame classifications are what smoothing and rollup
# consume) and still draw in the video player, which has the real frames.
#
# Verified detections pass on any frame. A human decision must never end up
# out of reach, and `rebuild_event_observations` already lets a species
# verified on some frame into the counts, so the grid has to be able to
# show the card that count came from.
#
# Images are never gated. This mirrors `on_embeddable_surface` in
# routers/labels.py, `build_embedding_input`, the annotated exports and
# `shouldDrawBbox` on the frontend.
_COMMON_JOINS = """
JOIN files f ON f.id = d.file_id
JOIN deployments dep ON dep.id = f.deployment_id
LEFT JOIN sites s ON s.id = dep.site_id
WHERE dep.project_id = ?
  AND (f.file_type != 'video'
       OR d.frame_number = f.best_frame_number
       OR d.verified = 1)"""

# Similarity / suggestions path: needs the embedding vector, so the base
# table is detection_embeddings (only embedded detections appear).
BASE_SQL = f"""
SELECT de.vector, de.l2_norm,
{_DETECTION_COLUMNS}
FROM detection_embeddings de
JOIN detections d ON d.id = de.detection_id
{_COMMON_JOINS}
"""

# Metadata-only path (event / time sorts): no embedding needed, so the
# base table is detections. Every detection passing the filters is
# included, embedded or not.
METADATA_SQL = f"""
SELECT
{_DETECTION_COLUMNS}
FROM detections d
{_COMMON_JOINS}
"""

# Which filtered detections have an embedding — ids only, no vectors. Used
# by "Sort by event" to pick one representative per event without loading
# every embedded vector.
PRESENCE_SQL = f"""
SELECT de.detection_id AS detection_id
FROM detection_embeddings de
JOIN detections d ON d.id = de.detection_id
{_COMMON_JOINS}
"""

def _build_query(
    project_id: str, filters: dict, base_sql: str = BASE_SQL
) -> tuple[str, list]:
    """Build filtered SQL query from filter dict. Returns (sql, params).

    ``base_sql`` is the SELECT ending on ``WHERE dep.project_id = ?``;
    the filter clauses are appended with AND. Defaults to the embedding
    query (``BASE_SQL``); the metadata sort path passes ``METADATA_SQL``.
    The filter clauses only reference ``d.`` / ``f.`` / ``dep.`` columns,
    so they are valid against both bases.
    """
    clauses: list[str] = []
    params: list = [project_id]

    if filters.get("labels"):
        placeholders = ",".join("?" for _ in filters["labels"])
        clauses.append(f"d.label_taxonomy_id IN ({placeholders})")
        params.extend(filters["labels"])

    if filters.get("site_ids"):
        # "null" is the reserved NO_SITE_SENTINEL token for deployments
        # with site_id IS NULL. Translate into the equivalent SQL and
        # handle mixed (sentinel + real site IDs) correctly.
        site_ids = list(filters["site_ids"])
        include_null = "null" in site_ids
        real_ids = [s for s in site_ids if s != "null"]
        if include_null and real_ids:
            placeholders = ",".join("?" for _ in real_ids)
            clauses.append(f"(dep.site_id IS NULL OR dep.site_id IN ({placeholders}))")
            params.extend(real_ids)
        elif include_null:
            clauses.append("dep.site_id IS NULL")
        elif real_ids:
            placeholders = ",".join("?" for _ in real_ids)
            clauses.append(f"dep.site_id IN ({placeholders})")
            params.extend(real_ids)

    if filters.get("date_from"):
        clauses.append("f.captured_at_local >= ?")
        params.append(filters["date_from"])

    if filters.get("date_to"):
        clauses.append("f.captured_at_local <= ?")
        params.append(filters["date_to"])

    # `project_floor` is the project's counting_threshold and applies the
    # global scope rule: pass on confidence, or verified as something
    # real. Hand-written copy of `threshold_or_verified` and its
    # NON_LABEL_CLASSES in `app/ml/label_exclusion.py` (this script runs
    # with no `app.*` on its path) — keep the three lists in step. The
    # label clause is what keeps a verified file's rejected weak boxes
    # ("false detection" below the floor) out of the grid. `min_confidence`
    # is the user's slider and is applied LITERALLY — a verified
    # low-confidence detection passes the floor's OR clause but cannot
    # satisfy a narrow user-set min.
    if filters.get("project_floor") is not None:
        clauses.append(
            "(d.confidence >= ? OR (d.verified = 1 AND (d.label IS NULL"
            " OR lower(d.label) NOT IN"
            " ('bait','blank','empty','false detection','none','vide'))))"
        )
        params.append(filters["project_floor"])

    if filters.get("min_confidence") is not None:
        clauses.append("d.confidence >= ?")
        params.append(filters["min_confidence"])

    if filters.get("max_confidence") is not None:
        clauses.append("d.confidence <= ?")
        params.append(filters["max_confidence"])

    # NULL label_confidence is excluded automatically by the comparison —
    # SQLite treats `NULL >= 0.0` as NULL, which a `WHERE` rejects.
    if filters.get("min_label_confidence") is not None:
        clauses.append("d.label_confidence >= ?")
        params.append(filters["min_label_confidence"])

    if filters.get("max_label_confidence") is not None:
        clauses.append("d.label_confidence <= ?")
        params.append(filters["max_label_confidence"])

    if filters.get("category"):
        clauses.append("d.category = ?")
        params.append(filters["category"])

    if filters.get("verified") is not None:
        clauses.append("d.verified = ?")
        params.append(1 if filters["verified"] else 0)

    # File-level triage marks (the Counts filters on Labels). Absent
    # means "all"; the values mirror the events endpoint's.
    if filters.get("flagged"):
        clauses.append("f.flagged = ?")
        params.append(1 if filters["flagged"] == "flagged" else 0)

    if filters.get("favorited"):
        clauses.append("f.favorited = ?")
        params.append(1 if filters["favorited"] == "favorited" else 0)

    sql = base_sql
    if clauses:
        sql += " AND " + " AND ".join(clauses)

    return sql, params


def _row_to_meta(row: sqlite3.Row) -> dict:
    """Build the per-detection metadata dict shared by both load paths.

    Reads only detection / file / event columns (no embedding fields),
    so it works for rows from either ``BASE_SQL`` or ``METADATA_SQL``.
    """
    ts = row["captured_at_local"]
    if ts and not isinstance(ts, str):
        ts = str(ts)

    event_start = row["event_start_local"]
    if event_start and not isinstance(event_start, str):
        event_start = str(event_start)

    return {
        "label": row["label"],
        "label_taxonomy_id": row["label_taxonomy_id"],
        "label_confidence": row["label_confidence"],
        "scientific_name": row["scientific_name"],
        "common_name": row["common_name"],
        "confidence": row["confidence"],
        "category": row["category"],
        "verified": bool(row["verified"]),
        "suggestion_dismissed": bool(row["suggestion_dismissed"]),
        "classification_method": row["classification_method"],
        "file_id": row["file_id"],
        "file_flagged": bool(row["file_flagged"]),
        "file_favorited": bool(row["file_favorited"]),
        "deployment_id": row["deployment_id"],
        "captured_at_local": ts,
        "site_name": row["site_name"],
        "event_id": row["event_id"],
        "event_sequence": row["event_sequence"],
        "event_start_local": event_start,
        "bbox_x": row["bbox_x"],
        "bbox_y": row["bbox_y"],
        "bbox_width": row["bbox_width"],
        "bbox_height": row["bbox_height"],
        "width_px": row["width_px"],
        "height_px": row["height_px"],
    }


def _load_metadata(
    db_path: str, project_id: str, filters: dict,
    max_detections: int = MAX_DETECTIONS,
) -> tuple[list[str], list[dict], int]:
    """Load detections for the metadata sorts, returning (ids, metas, total).

    Same filters and ``project_floor`` as the embedding path, but reads
    ``FROM detections`` so detections without an embedding are included.
    No vectors, no FAISS. Emits ``"detections"`` load progress: this is the
    dominant cost of the event / time sorts now that the cap is up to 50k
    rows, so it drives a real progress bar instead of a spinner.

    Capped to the newest ``max_detections`` by capture time (a memory
    guard); the caller reorders this subset. ``total`` is the uncapped
    count of the matching pool, so callers can tell the user when the
    result was capped. Dateless detections sort last, so they are the
    first dropped when capping.
    """
    sql, params = _build_query(project_id, filters, METADATA_SQL)

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        count_sql = f"SELECT COUNT(*) FROM ({sql})"
        total = conn.execute(count_sql, params).fetchone()[0] or 0
        if total == 0:
            return [], [], 0

        loaded = min(total, max_detections)
        load_sql = f"{sql} ORDER BY f.captured_at_local DESC LIMIT ?"
        _emit_progress("detections", 0, loaded)
        detection_ids: list[str] = []
        metas: list[dict] = []
        for i, row in enumerate(conn.execute(load_sql, [*params, max_detections])):
            detection_ids.append(row["detection_id"])
            metas.append(_row_to_meta(row))
            if (i + 1) % PROGRESS_LOAD_EVERY == 0:
                _emit_progress("detections", i + 1, loaded)
        _emit_progress("detections", loaded, loaded)
        return detection_ids, metas, total
    finally:
        conn.close()


def _load_embeddings(
    db_path: str, project_id: str, filters: dict, max_embeddings: int = MAX_EMBEDDINGS
) -> tuple[np.ndarray, list[str], list[dict], int]:
    """Load embeddings from SQLite, returning (vectors, ids, metadata, total).

    Streams the result set so the caller can report progress. Capped to
    the newest ``max_embeddings`` by capture time (the FAISS budget; the
    greedy walk holds every vector in RAM). ``total`` is the uncapped
    count of the matching pool so callers can tell the user when the
    result was capped.
    """
    sql, params = _build_query(project_id, filters)

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        # COUNT(*) on a wrapped subquery: clean and dialect-portable, and
        # SQLite plans it cheaply because the planner can short-circuit
        # the SELECT list. Gives both the progress size and the uncapped
        # total returned to the caller.
        count_sql = f"SELECT COUNT(*) FROM ({sql})"
        total = conn.execute(count_sql, params).fetchone()[0] or 0

        if total == 0:
            return np.empty((0, 0), dtype=np.float32), [], [], 0

        # Load only the newest cap by capture time (dateless rows last).
        loaded = min(total, max_embeddings)
        load_sql = f"{sql} ORDER BY f.captured_at_local DESC LIMIT ?"

        _emit_progress("load", 0, loaded)

        cursor = conn.execute(load_sql, [*params, max_embeddings])
        vectors: list[np.ndarray] = []
        detection_ids: list[str] = []
        metadata_list: list[dict] = []

        for i, row in enumerate(cursor):
            vec = np.frombuffer(row["vector"], dtype=np.float16).astype(np.float32)
            l2_norm = row["l2_norm"]
            if l2_norm and l2_norm > 0:
                vec = vec / l2_norm
            vectors.append(vec)
            detection_ids.append(row["detection_id"])
            metadata_list.append(_row_to_meta(row))

            if (i + 1) % PROGRESS_LOAD_EVERY == 0:
                _emit_progress("load", i + 1, loaded)

        _emit_progress("load", loaded, loaded)
    finally:
        conn.close()

    vectors_f32 = np.stack(vectors)
    return vectors_f32, detection_ids, metadata_list, total


def _load_embedding_presence(
    db_path: str, project_id: str, filters: dict
) -> set[str]:
    """Detection ids (matching the filters) that have an embedding.

    Ids only, no vectors, so it stays cheap even for a large pool. Lets
    the event sort pick each event's representative crop without loading
    every embedded vector.
    """
    sql, params = _build_query(project_id, filters, PRESENCE_SQL)
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        return {row["detection_id"] for row in conn.execute(sql, params)}
    finally:
        conn.close()


def _load_vectors_by_id(
    db_path: str, detection_ids: list[str]
) -> dict[str, np.ndarray]:
    """Load unit-normalised vectors for specific detection ids only.

    Chunked to respect SQLite's bound-variable limit. Used to fetch just
    the event representatives (one per event) rather than every vector.
    """
    if not detection_ids:
        return {}
    out: dict[str, np.ndarray] = {}
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        for start in range(0, len(detection_ids), 900):
            chunk = detection_ids[start : start + 900]
            placeholders = ",".join("?" for _ in chunk)
            rows = conn.execute(
                "SELECT detection_id, vector, l2_norm FROM detection_embeddings "
                f"WHERE detection_id IN ({placeholders})",
                chunk,
            )
            for row in rows:
                vec = np.frombuffer(
                    row["vector"], dtype=np.float16
                ).astype(np.float32)
                l2_norm = row["l2_norm"]
                if l2_norm and l2_norm > 0:
                    vec = vec / l2_norm
                out[row["detection_id"]] = vec
    finally:
        conn.close()
    return out


# Ordering of Linnaean ranks used by the descendant filter. A row with
# `level == "class"` sits at index 0 (broadest); `variant` (one rank
# below species) at 5. label_taxonomy stores levels exactly as these keys.
_TAXON_RANK_INDEX: dict[str, int] = {
    "class": 0,
    "order": 1,
    "family": 2,
    "genus": 3,
    "species": 4,
    "variant": 5,
}


def _load_label_taxonomy(
    db_path: str, project_id: str, label_list: list[str | None]
) -> dict[str, dict]:
    """Return {label_name: {level, taxon_class, ...}} for the labels in use.

    Scopes the lookup to the project's classification model so two
    models that share a label name cannot collide. Custom labels
    (per-project rows) are picked up alongside the model's taxonomy.
    """
    unique_labels = sorted({lab for lab in label_list if lab})
    if not unique_labels:
        return {}

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        # Match the model the project uses, plus this project's own
        # custom labels (which carry project_id and a NULL or matching
        # classification_model_id). A row with the same name but a
        # different model is ignored.
        cls_model_row = conn.execute(
            "SELECT classification_model_id FROM projects WHERE id = ?",
            (project_id,),
        ).fetchone()
        cls_model = cls_model_row[0] if cls_model_row else None

        placeholders = ",".join("?" for _ in unique_labels)
        params: list = list(unique_labels)
        where = f"name IN ({placeholders})"
        if cls_model is not None:
            where += " AND (classification_model_id = ? OR project_id = ?)"
            params.extend([cls_model, project_id])

        rows = conn.execute(
            f"""
            SELECT name, level, taxon_class, taxon_order, taxon_family,
                   taxon_genus, taxon_species
            FROM label_taxonomy
            WHERE {where}
            """,
            params,
        ).fetchall()
    finally:
        conn.close()

    return {r["name"]: dict(r) for r in rows}


def _is_useful_suggestion(
    suggested: dict | None, current: dict | None
) -> bool:
    """Decide whether to surface a neighbour-majority label as a suggestion.

    Keeps:
    - More specific (descendant) suggestions: e.g. canis → domestic dog.
    - Same-rank (lateral) suggestions: e.g. grey fox → coyote, mule deer
      → white-tailed deer. This is the dominant model-error mode for
      visually similar sibling species and is the case the user most
      often needs to fix.

    Drops:
    - Broader-rank (ancestor) suggestions: e.g. grey fox → mammals,
      domestic dog → canis. Going to a broader rank loses information
      and is almost never the right correction in a verification flow.
    - Anything where either side has no usable Linnaean rank
      (labels like "false detection" or custom labels without a
      taxonomy row). Without a rank we cannot compare safely.
    """
    if not suggested or not current:
        return False
    s_rank = _TAXON_RANK_INDEX.get(suggested.get("level") or "")
    c_rank = _TAXON_RANK_INDEX.get(current.get("level") or "")
    if s_rank is None or c_rank is None:
        return False
    return s_rank >= c_rank


def _compute_crop_bbox(meta: dict) -> dict | None:
    """Compute bbox position within the expanded crop (normalized 0-1).

    The crop is always centered on the bbox (no edge-shifting), matching
    the blurred-edge-fill behavior in crop_service.py.
    """
    img_w = meta.get("width_px")
    img_h = meta.get("height_px")
    if not img_w or not img_h:
        return None

    meta["bbox_x"] * img_w
    meta["bbox_y"] * img_h
    bw = meta["bbox_width"] * img_w
    bh = meta["bbox_height"] * img_h

    max_side = max(bw, bh)
    pad = max_side * 0.10
    crop_side = max_side + 2 * pad

    if crop_side <= 0:
        return None

    # Bbox is always centered: offset = pad / crop_side
    return {
        "x": (crop_side - bw) / 2 / crop_side,
        "y": (crop_side - bh) / 2 / crop_side,
        "w": bw / crop_side,
        "h": bh / crop_side,
    }


def _build_summary(
    detection_id: str,
    meta: dict,
    distance_to_centroid: float | None = None,
    similarity: float | None = None,
    neighbor_agreement: float | None = None,
    neighbor_top_label: str | None = None,
    neighbor_top_scientific_name: str | None = None,
    neighbor_top_common_name: str | None = None,
) -> dict:
    """Build detection summary dict (matches DetectionSummary schema)."""
    return {
        "detection_id": detection_id,
        "file_id": meta["file_id"],
        "label": meta["label"],
        "label_taxonomy_id": meta.get("label_taxonomy_id"),
        "label_confidence": meta["label_confidence"],
        "scientific_name": meta.get("scientific_name"),
        "common_name": meta.get("common_name"),
        "confidence": meta["confidence"],
        "category": meta["category"],
        "verified": meta["verified"],
        "classification_method": meta["classification_method"],
        "distance_to_centroid": distance_to_centroid,
        "similarity": similarity,
        "neighbor_agreement": neighbor_agreement,
        "neighbor_top_label": neighbor_top_label,
        "neighbor_top_scientific_name": neighbor_top_scientific_name,
        "neighbor_top_common_name": neighbor_top_common_name,
        "site_name": meta.get("site_name"),
        "deployment_id": meta.get("deployment_id"),
        "captured_at_local": meta.get("captured_at_local"),
        "event_id": meta.get("event_id"),
        "event_start_local": meta.get("event_start_local"),
        "crop_url": f"/api/detections/{detection_id}/crop?size=200",
        "crop_bbox": _compute_crop_bbox(meta),
        "frame_number": meta.get("frame_number"),
        "file_flagged": meta.get("file_flagged", False),
        "file_favorited": meta.get("file_favorited", False),
    }


# ── Neighbour signals (shared by sort and cohorts) ───────────────────────

def _pick_dense_seed(index, vectors: np.ndarray) -> int:
    """Pick the greedy walk's starting vector.

    Previous behaviour seeded at the embedding closest to the centroid
    (`vectors.mean(axis=0)`). On a dataset that spans several distinct
    label clusters the centroid sits in low-density space *between*
    clusters, and the vector closest to it is a between-cluster oddity
    — not a representative sample. The greedy walk then opens with
    visually mixed crops, which read as noise.

    Picking the densest vector instead (highest sum of cosine
    similarities to its k nearest neighbours) seeds inside the largest
    coherent region, so the walk's first row looks like a real cluster.

    Extra cost: one O(n·k) FAISS scan. At the 20k-detection cap that's
    well under a second on the existing flat index.
    """
    n = len(vectors)
    if n == 0:
        return 0
    k_neighbors = 10
    # +1 because the first neighbour is the vector itself (cosine sim = 1).
    k_query = min(k_neighbors + 1, n)
    similarities, _ = index.search(vectors, k_query)
    # Drop the self-similarity column and sum the rest. The vector with
    # the highest score has the densest immediate neighbourhood.
    density = similarities[:, 1:].sum(axis=1)
    return int(density.argmax())


def _greedy_walk(
    index, vectors: np.ndarray, *, progress_phase: str | None = None
) -> list[int]:
    """Greedy nearest-neighbour chain over ``vectors`` (already added to
    ``index``): start at a dense seed, then always jump to the nearest
    unvisited neighbour, so adjacent positions in the returned order look
    alike. Used for the per-detection similarity sort and for ordering
    events by their representative vectors. Emits progress only when
    ``progress_phase`` is set (the detection walk; the event walk is a
    handful of vectors and needs none).
    """
    n = len(vectors)
    current = _pick_dense_seed(index, vectors)

    if progress_phase:
        _emit_progress(progress_phase, 0, n)

    visited = np.zeros(n, dtype=bool)
    order = np.empty(n, dtype=np.int64)
    for step in range(n):
        order[step] = current
        visited[current] = True

        if progress_phase and (step + 1) % PROGRESS_LOOP_EVERY == 0:
            _emit_progress(progress_phase, step + 1, n)

        if step == n - 1:
            break

        # Search for k nearest neighbours (enough to find an unvisited one)
        k = min(64, n)
        while True:
            sims, idxs = index.search(vectors[current].reshape(1, -1), k)
            for idx in idxs[0]:
                if idx >= 0 and not visited[idx]:
                    current = int(idx)
                    break
            else:
                # All k neighbours visited — widen search
                if k >= n:
                    # All visited (shouldn't happen), pick first unvisited
                    remaining = np.where(~visited)[0]
                    current = int(remaining[0])
                    break
                k = min(k * 2, n)
                continue
            break

    if progress_phase:
        _emit_progress(progress_phase, n, n)

    return order.tolist()


def _greedy_order(vectors: np.ndarray) -> list[int]:
    """Build a FAISS index over ``vectors`` and return the greedy
    nearest-neighbour chain order.

    Convenience wrapper for callers that don't also need the index (the
    event-similarity sort). The per-detection similarity path builds its
    own index because it reuses it for the neighbour signals. Isolating
    the FAISS dependency here also lets the event-ordering logic be
    unit-tested without FAISS by patching this function.
    """
    import faiss

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return _greedy_walk(index, vectors)


def _compute_neighbor_signals(
    index,
    vectors: np.ndarray,
    label_list: list[str | None],
    db_path: str,
    project_id: str,
) -> tuple[np.ndarray, list[str | None]]:
    """Compute neighbour agreement + a per-detection label suggestion.

    For each detection, queries the 10 nearest embedding neighbours
    (via the caller-supplied FAISS `index`), scores how many share its
    label (agreement, in [0, 1]) and picks the most common neighbour
    label as a candidate suggestion. The candidate is kept only when
    `_is_useful_suggestion` says so: descendant or same-rank lateral.
    Broader-rank suggestions, no-op suggestions, and suggestions
    without taxonomy info collapse to `None`. Verified detections are
    not special-cased here, callers decide what to do with the signals.

    Shared by `do_sort` (drives the suggestions sort) and `do_cohorts`
    (drives the cohort review panel).
    """
    n = len(vectors)
    if n == 0:
        return np.zeros(0, dtype=np.float32), []

    k_neighbors = 10
    k_query = min(k_neighbors + 1, n)  # +1 because first result is self
    _, neighbor_idxs = index.search(vectors, k_query)

    _emit_progress("neighbors", 0, n)
    agreement_scores = np.zeros(n, dtype=np.float32)
    top_labels: list[str | None] = [None] * n
    for i in range(n):
        current_label = label_list[i]
        matches = 0
        count = 0
        neighbor_labels: list[str] = []
        for j in neighbor_idxs[i]:
            if j < 0 or j == i:
                continue
            count += 1
            if label_list[j]:
                neighbor_labels.append(label_list[j])
            if label_list[j] == current_label:
                matches += 1
        agreement_scores[i] = matches / count if count > 0 else 1.0
        if neighbor_labels:
            top_label, top_count = Counter(neighbor_labels).most_common(1)[0]
            # Plurality is too weak: a 3/10 winner produces noisy
            # suggestions. Require a clear majority share before
            # surfacing the candidate.
            if top_count / len(neighbor_labels) >= NEIGHBOR_MAJORITY_FRACTION:
                top_labels[i] = top_label
        if (i + 1) % PROGRESS_LOOP_EVERY == 0:
            _emit_progress("neighbors", i + 1, n)

    _emit_progress("neighbors", n, n)

    # Taxonomy filter: keep descendant promotions (canis → domestic dog)
    # and same-rank swaps (grey fox → coyote, which is the most common
    # model-error pattern), drop broader-rank suggestions (grey fox →
    # mammals). The agreement score is left untouched.
    label_to_taxonomy = _load_label_taxonomy(db_path, project_id, label_list)
    for i in range(n):
        suggested = top_labels[i]
        current = label_list[i]
        if not suggested or suggested == current:
            top_labels[i] = None
            continue
        if not _is_useful_suggestion(
            label_to_taxonomy.get(suggested),
            label_to_taxonomy.get(current),
        ):
            top_labels[i] = None

    return agreement_scores, top_labels


# ── Event sort ordered by similarity ─────────────────────────────────────


def _order_events_by_similarity(
    det_ids: list[str],
    metas: list[dict],
    has_embedding: set[str],
    load_vectors: Callable[[list[str]], dict[str, np.ndarray]],
    max_embeddings: int = MAX_EMBEDDINGS,
) -> list[int]:
    """Order detections for "Sort by event" using embedding similarity.

    Events stay atomic: a whole event's detections stay together, in
    capture-sequence order. The events themselves are ordered by a
    greedy nearest-neighbour walk over one representative vector each,
    so visually similar (usually same-species) events sit next to each
    other. Each event's representative is its most-confident detection
    that has an embedding. Events with no embedded detection (nothing to
    compare) fall to a chronological, camera-grouped tail; detections
    with no event go last.

    ``has_embedding`` is the set of ``det_ids`` that have a vector;
    ``load_vectors`` fetches vectors for a list of ids. Only the chosen
    representatives (one per event) are loaded, not every embedded
    detection.

    Returns the final index order into ``metas`` / ``det_ids``.
    """
    from observation_sort import order_events_by_deployment

    # Baseline: chronological, grouped by camera (within-event by
    # sequence, no-event last). The similarity walk overrides the order
    # of events that have a representative; events without one keep this
    # baseline, so the embedless tail is camera-grouped just like the
    # no-embedding fallback. Single-deployment folder runs reduce to
    # plain chronological.
    base_order = order_events_by_deployment(metas)

    # Group baseline indices by event, preserving first-seen (= time)
    # event order and within-event sequence order.
    event_to_indices: dict[str, list[int]] = {}
    events_in_baseline_order: list[str] = []
    no_event: list[int] = []
    for i in base_order:
        eid = metas[i].get("event_id")
        if not eid:
            no_event.append(i)
            continue
        if eid not in event_to_indices:
            event_to_indices[eid] = []
            events_in_baseline_order.append(eid)
        event_to_indices[eid].append(i)

    # Representative detection per event: the most-confident one that has
    # an embedding. No averaging, so a mixed event places by its clearest
    # crop instead of a meaningless centroid.
    rep_det: dict[str, str] = {}
    for eid, idxs in event_to_indices.items():
        best_i = -1
        best_conf = -1.0
        for i in idxs:
            if det_ids[i] in has_embedding:
                conf = metas[i].get("confidence") or 0.0
                if conf > best_conf:
                    best_conf = conf
                    best_i = i
        if best_i >= 0:
            rep_det[eid] = det_ids[best_i]

    # Load vectors for just the representatives, then greedy-walk the
    # events that have one. Cap the walk at the FAISS budget: only the
    # first `max_embeddings` events (baseline / time order) are similarity
    # ordered; any beyond that keep their baseline order in the tail, so a
    # project with more events than the budget still loads.
    rep_eids = [eid for eid in events_in_baseline_order if eid in rep_det]
    walk_eids = rep_eids[:max_embeddings]
    rep_vectors = load_vectors([rep_det[eid] for eid in walk_eids])
    walk_eids = [eid for eid in walk_eids if rep_det[eid] in rep_vectors]
    if len(walk_eids) >= 2:
        mat = np.stack([rep_vectors[rep_det[eid]] for eid in walk_eids])
        ordered_rep_eids = [walk_eids[j] for j in _greedy_order(mat)]
    else:
        ordered_rep_eids = walk_eids

    # Assemble: similarity-ordered events first, remaining (rep-less)
    # events in time order, no-event detections last.
    emitted = set(ordered_rep_eids)
    final: list[int] = []
    for eid in ordered_rep_eids:
        final.extend(event_to_indices[eid])
    for eid in events_in_baseline_order:
        if eid not in emitted:
            final.extend(event_to_indices[eid])
    final.extend(no_event)
    return final


# ── Similarity sort ──────────────────────────────────────────────────────

def _empty_result() -> dict:
    """The nothing-matched result, in the shape every caller expects."""
    return {
        "detections": [],
        "total_detections": 0,
        "total_loaded": 0,
        "total_matching": 0,
    }


def do_sort(db_path: str, project_id: str, params: dict) -> dict:
    """Sort detections for the Observations grid, chosen by ``params["sort"]``:

    - `similarity` (default): loads embeddings and does a greedy
      nearest-neighbour walk (via FAISS) so adjacent tiles look alike.
    - `events`: groups detections by event and orders the events by the
      similarity of one representative crop each (embedded or not).
    - `suggestions`: cohort-grouped review mode. Filters to unverified
      detections that carry a descendant-promotion suggestion, groups
      them by `(label, suggested_label, category)`, and orders cohorts
      by descending count. Skips the greedy walk because the embedding
      chain is irrelevant here.

    **Three counts come back, and they are not interchangeable.**

    - ``total_matching`` is the uncapped size of the pool the filters
      select.
    - ``total_loaded`` is how many of those rows this run actually read,
      i.e. ``min(total_matching, cap)``. **This is the only one that says
      whether the cap bit**, so ``total_matching > total_loaded`` is the
      one correct test for "the result was truncated".
    - ``total_detections`` is how many detections the sort returned. It
      equals ``total_loaded`` for `similarity` and `events`, which return
      every row they loaded, but `suggestions` narrows to cohort members
      by design and returns far fewer.

    That last difference is why ``total_loaded`` exists. The UI used to
    read truncation off ``total_matching > total_detections``, which is
    right for two sorts and wrong for the third: suggestions showed
    "showing the newest 8 of 814, capped to stay responsive" on a pool
    nowhere near the cap.
    """
    # `observation_sort` is a sibling file. Python prepends the running
    # script's dir to sys.path[0], so this resolves without the full
    # `app.ml.inference.*` package import (which would require pydantic
    # and other main-backend deps the conda ML env does not have).
    from observation_sort import VALID_SORTS, suggestions_order

    filters = params.get("filters", {})
    sort_mode = params.get("sort", "similarity")
    max_detections = int(params.get("max_detections", MAX_DETECTIONS))
    max_embeddings = int(params.get("max_embeddings", MAX_EMBEDDINGS))
    if sort_mode not in VALID_SORTS:
        raise ValueError(f"Unknown sort mode: {sort_mode}")

    # "Sort by event": always show the full detection population grouped
    # by event (embedded or not). With embeddings, order the still-atomic
    # events by similarity of their representative crops so same-species
    # events sit together; without embeddings, keep chronological order.
    if sort_mode == "events":
        det_ids, metas, total_matching = _load_metadata(
            db_path, project_id, filters, max_detections=max_detections
        )
        if not det_ids:
            return _empty_result()
        # Order events by similarity of one representative crop each,
        # loading only those rep vectors (not every embedded detection).
        # With no embeddings this falls back to chronological, camera-grouped
        # order; a single-deployment folder run reduces to plain chronological.
        final_order = _order_events_by_similarity(
            det_ids,
            metas,
            _load_embedding_presence(db_path, project_id, filters),
            lambda ids: _load_vectors_by_id(db_path, ids),
            max_embeddings=max_embeddings,
        )
        detections = [
            _build_summary(det_ids[i], metas[i]) for i in final_order
        ]
        return {
            "detections": detections,
            "total_detections": len(final_order),
            "total_loaded": len(det_ids),
            "total_matching": total_matching,
        }

    # similarity / suggestions: load embeddings and run FAISS.
    import faiss

    vectors, det_ids, metas, total_matching = _load_embeddings(
        db_path, project_id, filters, max_embeddings=max_embeddings
    )

    n = len(det_ids)
    if n == 0:
        return _empty_result()

    # Single detection — no sorting needed. Suggestions mode also
    # falls through here because a lone detection has no neighbours to
    # disagree with, so there's nothing to review.
    if n == 1:
        if sort_mode == "suggestions":
            return {
                "detections": [],
                "total_detections": 0,
                "total_loaded": n,
                "total_matching": total_matching,
            }
        return {
            "detections": [_build_summary(det_ids[0], metas[0])],
            "total_detections": 1,
            "total_loaded": n,
            "total_matching": total_matching,
        }

    # Build FAISS index for fast nearest-neighbor lookup
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)

    # The greedy walk only matters for the `similarity` order. Skip it
    # for suggestions so we don't pay for an order we throw away.
    similarity_order: list[int] = []
    if sort_mode == "similarity":
        similarity_order = _greedy_walk(index, vectors, progress_phase="sort")

    # Per-detection neighbour signals (agreement + descendant-filtered
    # suggestion). Shared with do_cohorts so the two paths can never
    # drift in how the suspicious flag and the promotion suggestion
    # are computed.
    label_list = [metas[i]["label"] for i in range(n)]
    agreement_scores, top_labels = _compute_neighbor_signals(
        index, vectors, label_list, db_path, project_id
    )

    if sort_mode == "suggestions":
        # Cohort-grouped review order. Filters the result set to the
        # cohort members in passing, so `total_detections` reflects the
        # filtered view rather than the project-wide population. The
        # `min_count` / `max_cohorts` defaults mirror the cohorts
        # endpoint so the toolbar's count signal matches what the grid
        # actually shows.
        final_order = suggestions_order(
            metas,
            top_labels,
            agreement_scores.tolist(),
            min_count=int(params.get("min_count", 8)),
            max_cohorts=int(params.get("max_cohorts", 200)),
        )
    else:
        # similarity: the greedy-walk order.
        final_order = similarity_order

    # Map raw label string → scientific / common name from the same
    # project's taxonomy. Used to render the suggested neighbor label with
    # the same display names shown elsewhere in the UI, instead of the raw
    # model class name (e.g. "M. meles" instead of "badger").
    label_to_scientific: dict[str, str] = {}
    label_to_common: dict[str, str] = {}
    for m in metas:
        label = m.get("label")
        if not label:
            continue
        sci = m.get("scientific_name")
        if sci and label not in label_to_scientific:
            label_to_scientific[label] = sci
        common = m.get("common_name")
        if common and label not in label_to_common:
            label_to_common[label] = common

    detections = [
        _build_summary(
            det_ids[i], metas[i],
            neighbor_agreement=float(agreement_scores[i]),
            neighbor_top_label=top_labels[i],
            neighbor_top_scientific_name=(
                label_to_scientific.get(top_labels[i]) if top_labels[i] else None
            ),
            neighbor_top_common_name=(
                label_to_common.get(top_labels[i]) if top_labels[i] else None
            ),
        )
        for i in final_order
    ]

    return {
        "detections": detections,
        "total_detections": len(final_order),
        "total_loaded": n,
        "total_matching": total_matching,
    }


# ── Cohorts ──────────────────────────────────────────────────────────────

def do_cohorts(db_path: str, project_id: str, params: dict) -> dict:
    """Group descendant-promotion suggestions for the cohort review panel.

    Loads every embedded detection in the project that passes the
    caller-supplied filters (typically just the project's
    `(confidence >= threshold OR verified)` floor, set by the service
    layer so the pill counts the same population the suggestions grid
    will actually load). Runs the same neighbour-agreement + strict-
    descendant pass as do_sort, and buckets unverified detections with
    a surviving suggestion by `(current_label, suggested_label,
    category)`. Returns the top `max_cohorts` cohorts (default 200)
    with at least `min_count` members each (default 8), ordered by
    descending count.

    Each cohort's `detection_ids` list is sorted by ascending neighbour
    agreement, so the frontend's thumbnail strip leads with the crops
    whose neighbours disagree most strongly, which are the clearest
    candidates for a one-click promotion.
    """
    import faiss

    min_count = int(params.get("min_count", 8))
    max_cohorts = int(params.get("max_cohorts", 200))
    if min_count < 1:
        raise ValueError(f"min_count must be >= 1, got {min_count}")
    if max_cohorts < 1:
        raise ValueError(f"max_cohorts must be >= 1, got {max_cohorts}")

    filters = params.get("filters", {})
    vectors, det_ids, metas, _ = _load_embeddings(
        db_path, project_id, filters, max_embeddings=MAX_EMBEDDINGS
    )

    n = len(det_ids)
    if n < 2:
        # Fewer than two embeddings: no neighbour structure to learn from.
        return {"cohorts": []}

    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)

    label_list = [metas[i]["label"] for i in range(n)]
    agreement_scores, top_labels = _compute_neighbor_signals(
        index, vectors, label_list, db_path, project_id
    )

    return {
        "cohorts": _group_cohorts(
            det_ids,
            metas,
            agreement_scores,
            top_labels,
            min_count,
            max_cohorts,
        )
    }


def _group_cohorts(
    det_ids: list[str],
    metas: list[dict],
    agreement_scores: np.ndarray,
    top_labels: list[str | None],
    min_count: int,
    max_cohorts: int,
) -> list[dict]:
    """Pure grouping pass for do_cohorts. Unit-testable without FAISS.

    Walks per-detection signals, buckets unverified detections with a
    surviving descendant suggestion by `(current_label, suggested_label,
    category)`, drops cohorts under `min_count`, sorts by descending
    count, and returns the first `max_cohorts`. Members inside each
    cohort are sorted by ascending neighbour agreement.

    The key normalises `None` to the empty string so dict hashing is
    well-defined; each cohort row preserves the original (possibly
    None) `current_label` and `category` for the relabel call.
    """
    label_to_scientific: dict[str, str] = {}
    label_to_common: dict[str, str] = {}
    for m in metas:
        label = m.get("label")
        if not label:
            continue
        sci = m.get("scientific_name")
        if sci and label not in label_to_scientific:
            label_to_scientific[label] = sci
        common = m.get("common_name")
        if common and label not in label_to_common:
            label_to_common[label] = common

    cohorts: dict[tuple[str, str, str], dict] = {}
    for i, det_id in enumerate(det_ids):
        if metas[i].get("verified"):
            continue
        # User dismissed this crop's suggestion: keep it as a neighbour
        # vote (it's still in metas) but never make it a cohort member.
        if metas[i].get("suggestion_dismissed"):
            continue
        suggested = top_labels[i]
        if not suggested:
            continue
        current_label = metas[i].get("label")
        category = metas[i].get("category")
        key = (current_label or "", suggested, category or "")
        bucket = cohorts.get(key)
        if bucket is None:
            bucket = {
                "current_label": current_label,
                # Carry the taxonomy id so the panel's "Review crops"
                # navigation can drop the user into the existing
                # Observations label filter (which takes taxonomy ids).
                "current_label_taxonomy_id": metas[i].get("label_taxonomy_id"),
                "current_scientific_name": label_to_scientific.get(current_label or ""),
                "current_common_name": label_to_common.get(current_label or ""),
                "suggested_label": suggested,
                "suggested_scientific_name": label_to_scientific.get(suggested),
                "suggested_common_name": label_to_common.get(suggested),
                "category": category,
                "members": [],
            }
            cohorts[key] = bucket
        bucket["members"].append((float(agreement_scores[i]), det_id))

    output: list[dict] = []
    for bucket in cohorts.values():
        if len(bucket["members"]) < min_count:
            continue
        bucket["members"].sort(key=lambda pair: pair[0])
        output.append(
            {
                "current_label": bucket["current_label"],
                "current_label_taxonomy_id": bucket["current_label_taxonomy_id"],
                "current_scientific_name": bucket["current_scientific_name"],
                "current_common_name": bucket["current_common_name"],
                "suggested_label": bucket["suggested_label"],
                "suggested_scientific_name": bucket["suggested_scientific_name"],
                "suggested_common_name": bucket["suggested_common_name"],
                "category": bucket["category"],
                "count": len(bucket["members"]),
                "detection_ids": [det_id for _, det_id in bucket["members"]],
            }
        )

    output.sort(key=lambda c: -c["count"])
    return output[:max_cohorts]


# ── Search ───────────────────────────────────────────────────────────────

def _search(
    anchor_vector: np.ndarray,
    vectors: np.ndarray,
    limit: int = 100,
    threshold: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """FAISS inner-product search. Returns (indices, similarities)."""
    import faiss

    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)

    k = min(limit + 1, len(vectors))
    similarities, indices = index.search(anchor_vector.reshape(1, -1), k)

    sims = similarities[0]
    idxs = indices[0]

    mask = sims >= threshold
    return idxs[mask][:limit], sims[mask][:limit]


def _load_anchor_embedding(
    db_path: str, detection_id: str
) -> tuple[np.ndarray, dict]:
    """Load a single detection's embedding and metadata."""
    sql = """
    SELECT de.vector, de.l2_norm,
           d.label, d.label_confidence, d.scientific_name, d.common_name,
           d.confidence, d.category,
           d.verified, d.classification_method, d.file_id,
           d.bbox_x, d.bbox_y, d.bbox_width, d.bbox_height,
           f.deployment_id, f.captured_at_local, f.width_px, f.height_px,
           f.flagged AS file_flagged, f.favorited AS file_favorited,
           s.name AS site_name
    FROM detection_embeddings de
    JOIN detections d ON d.id = de.detection_id
    JOIN files f ON f.id = d.file_id
    JOIN deployments dep ON dep.id = f.deployment_id
    LEFT JOIN sites s ON s.id = dep.site_id
    WHERE de.detection_id = ?
    """

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(sql, [detection_id]).fetchone()
    finally:
        conn.close()

    if not row:
        raise ValueError(f"No embedding found for detection {detection_id}")

    vec = np.frombuffer(row["vector"], dtype=np.float16).astype(np.float32)
    l2_norm = row["l2_norm"]
    if l2_norm and l2_norm > 0:
        vec = vec / l2_norm

    ts = row["captured_at_local"]
    if ts and isinstance(ts, str):
        pass
    elif ts:
        ts = str(ts)

    meta = {
        "label": row["label"],
        "label_confidence": row["label_confidence"],
        "scientific_name": row["scientific_name"],
        "common_name": row["common_name"],
        "confidence": row["confidence"],
        "category": row["category"],
        "verified": bool(row["verified"]),
        "classification_method": row["classification_method"],
        "file_id": row["file_id"],
        "file_flagged": bool(row["file_flagged"]),
        "file_favorited": bool(row["file_favorited"]),
        "deployment_id": row["deployment_id"],
        "captured_at_local": ts,
        "site_name": row["site_name"],
        "bbox_x": row["bbox_x"],
        "bbox_y": row["bbox_y"],
        "bbox_width": row["bbox_width"],
        "bbox_height": row["bbox_height"],
        "width_px": row["width_px"],
        "height_px": row["height_px"],
    }

    return vec, meta


def do_search(db_path: str, project_id: str, params: dict) -> dict:
    """Full search pipeline: load, search, format response."""
    anchor_id = params["anchor_detection_id"]
    filters = params.get("filters", {})
    limit = params.get("limit", 100)
    threshold = params.get("threshold", 0.0)
    max_embeddings = int(params.get("max_embeddings", MAX_EMBEDDINGS))

    vectors, det_ids, metas, _ = _load_embeddings(
        db_path, project_id, filters, max_embeddings=max_embeddings
    )

    # Find or load anchor
    anchor_idx = None
    if anchor_id in det_ids:
        anchor_idx = det_ids.index(anchor_id)

    if anchor_idx is not None:
        anchor_vector = vectors[anchor_idx]
        anchor_meta = metas[anchor_idx]
    else:
        anchor_vector, anchor_meta = _load_anchor_embedding(db_path, anchor_id)

    anchor_summary = _build_summary(anchor_id, anchor_meta, similarity=1.0)

    if len(vectors) == 0:
        return {
            "anchor": anchor_summary,
            "results": [],
            "total_results": 0,
            "threshold_applied": threshold,
        }

    indices, similarities = _search(anchor_vector, vectors, limit=limit, threshold=threshold)

    results = []
    for idx, sim in zip(indices, similarities, strict=False):
        idx = int(idx)
        if det_ids[idx] == anchor_id:
            continue
        results.append(
            _build_summary(det_ids[idx], metas[idx], similarity=float(sim))
        )

    return {
        "anchor": anchor_summary,
        "results": results[:limit],
        "total_results": len(results),
        "threshold_applied": threshold,
    }


# ── Main ─────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Similarity computation (FAISS)")
    parser.add_argument("--db-path", required=True, help="Path to SQLite database")
    parser.add_argument("--project-id", required=True, help="Project UUID")
    parser.add_argument(
        "--operation", required=True, choices=["sort", "search", "cohorts"]
    )
    parser.add_argument("--params", required=True, help="JSON string with operation parameters")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    params = json.loads(args.params)

    if args.operation == "sort":
        result = do_sort(args.db_path, args.project_id, params)
    elif args.operation == "search":
        result = do_search(args.db_path, args.project_id, params)
    elif args.operation == "cohorts":
        result = do_cohorts(args.db_path, args.project_id, params)
    else:
        _emit_event({"type": "error", "message": f"Unknown operation: {args.operation}"})
        sys.exit(1)

    _emit_event({"type": "result", **result})


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Surface a structured error event so the parent can render it
        # inline. Stderr is reserved for actual log noise; everything
        # the parent should react to goes via NDJSON on stdout.
        _emit_event({"type": "error", "message": str(e)})
        sys.exit(1)

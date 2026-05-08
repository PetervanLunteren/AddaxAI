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
    {"type": "result", "detections": [...], "total_detections": N}
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
from typing import Any

import numpy as np

# Hard fallback if the parent doesn't pass --max-detections. Real value
# comes from Project.observations_max_detections on every call.
MAX_DETECTIONS = 20_000

# How often to emit progress events during long loops. Tuned for ~50ms
# between updates at typical scales: 500 rows per SQL emission, 200
# steps per O(n) loop. Smaller intervals would flood the wire; larger
# would make the bar feel stuck.
PROGRESS_LOAD_EVERY = 500
PROGRESS_LOOP_EVERY = 200


def _emit_event(event: dict[str, Any]) -> None:
    """Write one NDJSON line to stdout. Flushed so the parent sees it live."""
    sys.stdout.write(json.dumps(event, default=str) + "\n")
    sys.stdout.flush()


def _emit_progress(phase: str, done: int, total: int) -> None:
    _emit_event({"type": "progress", "phase": phase, "done": done, "total": total})

# ── SQL ──────────────────────────────────────────────────────────────────

BASE_SQL = """
SELECT de.detection_id, de.vector, de.l2_norm,
       d.label, d.label_confidence, d.display_name, d.confidence, d.category,
       d.verified, d.classification_method, d.file_id,
       d.bbox_x, d.bbox_y, d.bbox_width, d.bbox_height,
       f.deployment_id, f.captured_at_local, f.width_px, f.height_px,
       s.name AS site_name
FROM detection_embeddings de
JOIN detections d ON d.id = de.detection_id
JOIN files f ON f.id = d.file_id
JOIN deployments dep ON dep.id = f.deployment_id
LEFT JOIN sites s ON s.id = dep.site_id
WHERE dep.project_id = ?
"""


def _build_query(project_id: str, filters: dict) -> tuple[str, list]:
    """Build filtered SQL query from filter dict. Returns (sql, params)."""
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

    # `project_floor` is the project's detection_threshold and applies the
    # global "threshold + verified override" rule. `min_confidence` is the
    # user's slider and is applied LITERALLY — a verified low-confidence
    # detection passes the floor's OR clause but cannot satisfy a narrow
    # user-set min.
    if filters.get("project_floor") is not None:
        clauses.append("(d.confidence >= ? OR d.verified = 1)")
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

    sql = BASE_SQL
    if clauses:
        sql += " AND " + " AND ".join(clauses)

    return sql, params


def _load_embeddings(
    db_path: str, project_id: str, filters: dict, max_detections: int = MAX_DETECTIONS
) -> tuple[np.ndarray, list[str], list[dict]]:
    """Load embeddings from SQLite, returning (vectors, ids, metadata).

    Streams the result set so the caller can report progress: a COUNT(*)
    pass establishes the total, then row-by-row iteration emits a
    progress event every PROGRESS_LOAD_EVERY rows.
    """
    sql, params = _build_query(project_id, filters)

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        # COUNT(*) on a wrapped subquery: clean and dialect-portable, and
        # SQLite plans it cheaply because the planner can short-circuit
        # the SELECT list. Used purely to size the progress bar.
        count_sql = f"SELECT COUNT(*) FROM ({sql})"
        total = conn.execute(count_sql, params).fetchone()[0] or 0

        if total == 0:
            return np.empty((0, 0), dtype=np.float32), [], []

        if total > max_detections:
            raise ValueError(
                f"Too many detections ({total}, current limit {max_detections}). "
                "Narrow the result by species, site, or date, or raise the "
                "limit in Settings → Verification."
            )

        _emit_progress("load", 0, total)

        cursor = conn.execute(sql, params)
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

            ts = row["captured_at_local"]
            if ts and isinstance(ts, str):
                pass
            elif ts:
                ts = str(ts)

            metadata_list.append({
                "label": row["label"],
                "label_confidence": row["label_confidence"],
                "display_name": row["display_name"],
                "confidence": row["confidence"],
                "category": row["category"],
                "verified": bool(row["verified"]),
                "classification_method": row["classification_method"],
                "file_id": row["file_id"],
                "deployment_id": row["deployment_id"],
                "captured_at_local": ts,
                "site_name": row["site_name"],
                "bbox_x": row["bbox_x"],
                "bbox_y": row["bbox_y"],
                "bbox_width": row["bbox_width"],
                "bbox_height": row["bbox_height"],
                "width_px": row["width_px"],
                "height_px": row["height_px"],
            })

            if (i + 1) % PROGRESS_LOAD_EVERY == 0:
                _emit_progress("load", i + 1, total)

        _emit_progress("load", total, total)
    finally:
        conn.close()

    vectors_f32 = np.stack(vectors)
    return vectors_f32, detection_ids, metadata_list


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
    neighbor_top_display_name: str | None = None,
) -> dict:
    """Build detection summary dict (matches DetectionSummary schema)."""
    return {
        "detection_id": detection_id,
        "file_id": meta["file_id"],
        "label": meta["label"],
        "label_confidence": meta["label_confidence"],
        "display_name": meta.get("display_name"),
        "confidence": meta["confidence"],
        "category": meta["category"],
        "verified": meta["verified"],
        "classification_method": meta["classification_method"],
        "distance_to_centroid": distance_to_centroid,
        "similarity": similarity,
        "neighbor_agreement": neighbor_agreement,
        "neighbor_top_label": neighbor_top_label,
        "neighbor_top_display_name": neighbor_top_display_name,
        "site_name": meta.get("site_name"),
        "deployment_id": meta.get("deployment_id"),
        "captured_at_local": meta.get("captured_at_local"),
        "crop_url": f"/api/detections/{detection_id}/crop?size=200",
        "crop_bbox": _compute_crop_bbox(meta),
    }


# ── Similarity sort ──────────────────────────────────────────────────────

def do_sort(db_path: str, project_id: str, params: dict) -> dict:
    """Sort detections for the Observations grid.

    Always loads embeddings and computes neighbor agreement / top label
    via FAISS, since the suspicious filter depends on those fields
    regardless of the visible order. The final ordering is then chosen
    by `_sort_index` according to `params["sort"]`:

    - `similarity` (default): greedy nearest-neighbor walk so adjacent
      tiles look alike.
    - `similarity_reverse`: same chain, reversed.
    - `newest` / `oldest`: by `captured_at_local`, NULL last.
    - `cls_low`: by `label_confidence` ascending, NULL last (verify
      hardest cases first).
    """
    import faiss

    # `observation_sort` is a sibling file. Python prepends the running
    # script's dir to sys.path[0], so this resolves without the full
    # `app.ml.inference.*` package import (which would require pydantic
    # and other main-backend deps the conda ML env does not have).
    from observation_sort import VALID_SORTS, order_indices

    filters = params.get("filters", {})
    sort_mode = params.get("sort", "similarity")
    max_detections = int(params.get("max_detections", MAX_DETECTIONS))
    if sort_mode not in VALID_SORTS:
        raise ValueError(f"Unknown sort mode: {sort_mode}")

    vectors, det_ids, metas = _load_embeddings(
        db_path, project_id, filters, max_detections=max_detections
    )

    n = len(det_ids)
    if n == 0:
        return {"detections": [], "total_detections": 0}

    # Single detection — no sorting needed
    if n == 1:
        return {
            "detections": [_build_summary(det_ids[0], metas[0])],
            "total_detections": 1,
        }

    # Build FAISS index for fast nearest-neighbor lookup
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)

    # Start from the most typical detection (closest to the centroid)
    centroid = vectors.mean(axis=0, keepdims=True)
    _, centroid_idx = index.search(centroid, 1)
    current = int(centroid_idx[0][0])

    _emit_progress("sort", 0, n)

    # Greedy walk: always jump to nearest unvisited neighbor
    visited = np.zeros(n, dtype=bool)
    order = np.empty(n, dtype=np.int64)
    for step in range(n):
        order[step] = current
        visited[current] = True

        if (step + 1) % PROGRESS_LOOP_EVERY == 0:
            _emit_progress("sort", step + 1, n)

        if step == n - 1:
            break

        # Search for k nearest neighbors (enough to find an unvisited one)
        k = min(64, n)
        while True:
            sims, idxs = index.search(vectors[current].reshape(1, -1), k)
            for idx in idxs[0]:
                if idx >= 0 and not visited[idx]:
                    current = int(idx)
                    break
            else:
                # All k neighbors visited — widen search
                if k >= n:
                    # All visited (shouldn't happen), pick first unvisited
                    remaining = np.where(~visited)[0]
                    current = int(remaining[0])
                    break
                k = min(k * 2, n)
                continue
            break

    _emit_progress("sort", n, n)

    # Compute neighbor agreement: for each detection, what fraction of its
    # k nearest embedding neighbors share the same label?
    k_neighbors = 10
    k_query = min(k_neighbors + 1, n)  # +1 because first result is self
    _, neighbor_idxs = index.search(vectors, k_query)

    _emit_progress("neighbors", 0, n)
    label_list = [metas[i]["label"] for i in range(n)]
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
            top_labels[i] = Counter(neighbor_labels).most_common(1)[0][0]
        if (i + 1) % PROGRESS_LOOP_EVERY == 0:
            _emit_progress("neighbors", i + 1, n)

    _emit_progress("neighbors", n, n)
    final_order = order_indices(sort_mode, order.tolist(), metas)

    # Map raw label string → display_name from the same project's
    # taxonomy. Used to render the suggested neighbor label as the same
    # Latin display name shown elsewhere in the UI, instead of the raw
    # model class name (e.g. "M. meles" instead of "badger").
    label_to_display: dict[str, str] = {}
    for m in metas:
        label = m.get("label")
        display = m.get("display_name")
        if label and display and label not in label_to_display:
            label_to_display[label] = display

    detections = [
        _build_summary(
            det_ids[i], metas[i],
            neighbor_agreement=float(agreement_scores[i]),
            neighbor_top_label=top_labels[i],
            neighbor_top_display_name=(
                label_to_display.get(top_labels[i]) if top_labels[i] else None
            ),
        )
        for i in final_order
    ]

    return {"detections": detections, "total_detections": n}


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
           d.label, d.label_confidence, d.display_name, d.confidence, d.category,
           d.verified, d.classification_method, d.file_id,
           d.bbox_x, d.bbox_y, d.bbox_width, d.bbox_height,
           f.deployment_id, f.captured_at_local, f.width_px, f.height_px,
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
        "display_name": row["display_name"],
        "confidence": row["confidence"],
        "category": row["category"],
        "verified": bool(row["verified"]),
        "classification_method": row["classification_method"],
        "file_id": row["file_id"],
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
    max_detections = int(params.get("max_detections", MAX_DETECTIONS))

    vectors, det_ids, metas = _load_embeddings(
        db_path, project_id, filters, max_detections=max_detections
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
    parser.add_argument("--operation", required=True, choices=["sort", "search"])
    parser.add_argument("--params", required=True, help="JSON string with operation parameters")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    params = json.loads(args.params)

    if args.operation == "sort":
        result = do_sort(args.db_path, args.project_id, params)
    elif args.operation == "search":
        result = do_search(args.db_path, args.project_id, params)
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

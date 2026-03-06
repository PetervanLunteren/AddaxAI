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

Output: JSON to stdout matching SortResponse / SearchResponse structure.
Errors: stderr + non-zero exit code.

Following CONVENTIONS.md: crash early and loudly, no silent failures.
"""

import argparse
import json
import sqlite3
import sys
from collections import Counter

import numpy as np

MAX_DETECTIONS = 20_000

# ── SQL ──────────────────────────────────────────────────────────────────

BASE_SQL = """
SELECT de.detection_id, de.vector, de.l2_norm,
       d.species, d.species_confidence, d.confidence, d.category,
       d.verified, d.classification_method, d.file_id,
       d.bbox_x, d.bbox_y, d.bbox_width, d.bbox_height,
       f.deployment_id, f.timestamp, f.width_px, f.height_px,
       s.name AS site_name
FROM detection_embeddings de
JOIN detections d ON d.id = de.detection_id
JOIN files f ON f.id = d.file_id
JOIN deployments dep ON dep.id = f.deployment_id
JOIN sites s ON s.id = dep.site_id
WHERE s.project_id = ?
"""


def _build_query(project_id: str, filters: dict) -> tuple[str, list]:
    """Build filtered SQL query from filter dict. Returns (sql, params)."""
    clauses: list[str] = []
    params: list = [project_id]

    if filters.get("species"):
        placeholders = ",".join("?" for _ in filters["species"])
        clauses.append(f"d.species IN ({placeholders})")
        params.extend(filters["species"])

    if filters.get("site_ids"):
        placeholders = ",".join("?" for _ in filters["site_ids"])
        clauses.append(f"s.id IN ({placeholders})")
        params.extend(filters["site_ids"])

    if filters.get("date_from"):
        clauses.append("f.timestamp >= ?")
        params.append(filters["date_from"])

    if filters.get("date_to"):
        clauses.append("f.timestamp <= ?")
        params.append(filters["date_to"])

    if filters.get("min_confidence") is not None:
        clauses.append("d.confidence >= ?")
        params.append(filters["min_confidence"])

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
    db_path: str, project_id: str, filters: dict
) -> tuple[np.ndarray, list[str], list[dict]]:
    """Load embeddings from SQLite, returning (vectors, ids, metadata)."""
    sql, params = _build_query(project_id, filters)

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        cursor = conn.execute(sql, params)
        rows = cursor.fetchall()
    finally:
        conn.close()

    if not rows:
        return np.empty((0, 0), dtype=np.float32), [], []

    if len(rows) > MAX_DETECTIONS:
        raise ValueError(
            f"Too many detections ({len(rows)}). "
            f"Add a species, site, or date filter to narrow below {MAX_DETECTIONS}."
        )

    vectors = []
    detection_ids = []
    metadata_list = []

    for row in rows:
        vec = np.frombuffer(row["vector"], dtype=np.float16).astype(np.float32)
        l2_norm = row["l2_norm"]
        if l2_norm and l2_norm > 0:
            vec = vec / l2_norm
        vectors.append(vec)
        detection_ids.append(row["detection_id"])

        ts = row["timestamp"]
        if ts and isinstance(ts, str):
            # Keep as ISO string for JSON serialization
            pass
        elif ts:
            ts = str(ts)

        metadata_list.append({
            "species": row["species"],
            "species_confidence": row["species_confidence"],
            "confidence": row["confidence"],
            "category": row["category"],
            "verified": bool(row["verified"]),
            "classification_method": row["classification_method"],
            "file_id": row["file_id"],
            "deployment_id": row["deployment_id"],
            "timestamp": ts,
            "site_name": row["site_name"],
            "bbox_x": row["bbox_x"],
            "bbox_y": row["bbox_y"],
            "bbox_width": row["bbox_width"],
            "bbox_height": row["bbox_height"],
            "width_px": row["width_px"],
            "height_px": row["height_px"],
        })

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
) -> dict:
    """Build detection summary dict (matches DetectionSummary schema)."""
    return {
        "detection_id": detection_id,
        "file_id": meta["file_id"],
        "species": meta["species"],
        "species_confidence": meta["species_confidence"],
        "confidence": meta["confidence"],
        "category": meta["category"],
        "verified": meta["verified"],
        "classification_method": meta["classification_method"],
        "distance_to_centroid": distance_to_centroid,
        "similarity": similarity,
        "neighbor_agreement": neighbor_agreement,
        "neighbor_top_label": neighbor_top_label,
        "site_name": meta.get("site_name"),
        "deployment_id": meta.get("deployment_id"),
        "timestamp": meta.get("timestamp"),
        "crop_url": f"/api/detections/{detection_id}/crop?size=200",
        "crop_bbox": _compute_crop_bbox(meta),
    }


# ── Similarity sort ──────────────────────────────────────────────────────

def do_sort(db_path: str, project_id: str, params: dict) -> dict:
    """Greedy nearest-neighbor chain: walk to closest unvisited neighbor.

    Uses FAISS IndexFlatIP (inner product on L2-normalised vectors = cosine).
    Result: a flat list where adjacent detections look visually similar.
    """
    import faiss

    filters = params.get("filters", {})
    vectors, det_ids, metas = _load_embeddings(db_path, project_id, filters)

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

    # Greedy walk: always jump to nearest unvisited neighbor
    visited = np.zeros(n, dtype=bool)
    order = np.empty(n, dtype=np.int64)
    for step in range(n):
        order[step] = current
        visited[current] = True

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

    # Compute neighbor agreement: for each detection, what fraction of its
    # k nearest embedding neighbors share the same species label?
    k_neighbors = 10
    k_query = min(k_neighbors + 1, n)  # +1 because first result is self
    _, neighbor_idxs = index.search(vectors, k_query)

    species_list = [metas[i]["species"] for i in range(n)]
    agreement_scores = np.zeros(n, dtype=np.float32)
    top_labels: list[str | None] = [None] * n
    for i in range(n):
        label = species_list[i]
        matches = 0
        count = 0
        neighbor_species: list[str] = []
        for j in neighbor_idxs[i]:
            if j < 0 or j == i:
                continue
            count += 1
            if species_list[j]:
                neighbor_species.append(species_list[j])
            if species_list[j] == label:
                matches += 1
        agreement_scores[i] = matches / count if count > 0 else 1.0
        if neighbor_species:
            top_labels[i] = Counter(neighbor_species).most_common(1)[0][0]

    detections = [
        _build_summary(
            det_ids[i], metas[i],
            neighbor_agreement=float(agreement_scores[i]),
            neighbor_top_label=top_labels[i],
        )
        for i in order
    ]

    if params.get("reverse", False):
        detections.reverse()

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
           d.species, d.species_confidence, d.confidence, d.category,
           d.verified, d.classification_method, d.file_id,
           d.bbox_x, d.bbox_y, d.bbox_width, d.bbox_height,
           f.deployment_id, f.timestamp, f.width_px, f.height_px,
           s.name AS site_name
    FROM detection_embeddings de
    JOIN detections d ON d.id = de.detection_id
    JOIN files f ON f.id = d.file_id
    JOIN deployments dep ON dep.id = f.deployment_id
    JOIN sites s ON s.id = dep.site_id
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

    ts = row["timestamp"]
    if ts and isinstance(ts, str):
        pass
    elif ts:
        ts = str(ts)

    meta = {
        "species": row["species"],
        "species_confidence": row["species_confidence"],
        "confidence": row["confidence"],
        "category": row["category"],
        "verified": bool(row["verified"]),
        "classification_method": row["classification_method"],
        "file_id": row["file_id"],
        "deployment_id": row["deployment_id"],
        "timestamp": ts,
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

    vectors, det_ids, metas = _load_embeddings(db_path, project_id, filters)

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
        print(f"Unknown operation: {args.operation}", file=sys.stderr)
        sys.exit(1)

    # Output JSON to stdout
    json.dump(result, sys.stdout, default=str)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

"""
Seed a throwaway AddaxAI data folder with a large image database, in the
shape of one of the databases that made the app slow.

``--shape simon`` (the default), after Simon Kravis: 923K photos, 4 GB, a
small project next to a huge one. Five projects, about 1M photos in all at
``--scale 1``:

- "Large": one site, one deployment of 900K photos in 100 camera
  subfolders, a whole season of cameras in one deployment.
- "Small": three sites, three deployments, 27K photos. The project that
  gets slow because the large one shares its database.
- "Normal 1" to "Normal 3": two sites and two deployments of 10K each.

``--shape elliot``, after Elliot Johnston's ``addaxai-db-stats.txt``
(v7.16, 3.8 GB): one project of 1.27M photos in 58 deployments of 2K to
35K photos, each deployment with the photo count, box count and share of
boxes at 0.2 or more that his has. About 270 species, a few common and
many rare, and long sequences of about 62 photos.

Boxes look like MegaDetector output stored down to 0.01. The Simon shape
adds a tail of legacy boxes between 0.005 and 0.01 (``--legacy-share`` of
all rows), which databases from before v7.2.0 still carry. Animal boxes
carry a species from ``--taxonomy-csv``. Every box that passes the
classification gate gets an embedding: random vectors, because only
their size and count matter here. Sequences come from the app's own sequence
generation, then the planner statistics are refreshed, as after a job.

Everything goes through the app's models and functions, so the schema
is always the current one. A fixed random seed makes every run give the
same database. Nothing is written outside the folder you name, and it
must not be your real ``~/AddaxAI``:

    ADDAXAI_USER_DATA_DIR=/Volumes/2TB-SSD/addaxai-speed/golden \\
        backend/venv/bin/python backend/scripts/seed_large_image_database.py \\
        --classification-model SPECIESNET-v4-0-2-A \\
        --taxonomy-csv ~/AddaxAI/models/cls/SPECIESNET-v4-0-2-A/taxonomy.csv

At full scale this takes the better part of an hour, most of it sequence
generation and embeddings, and about 3.5 GB of disk. The photos do not
exist, so crops and thumbnails fail fast: this measures the database,
not the drive. How it is used is in "How we measure" in
future-plans/large-database-speed.md.
"""

from __future__ import annotations

import argparse
import random
import tempfile
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from _seed_sandbox import open_seed_sandbox

SEED = 20261120
INSERT_CHUNK = 50_000
EMBEDDING_CHUNK = 50_000
EMBEDDING_MODEL_ID = "DINOV2-VITS14"
EMBEDDING_DIM = 384
TIMEZONE = "Australia/Sydney"
SEASON_START = datetime(2025, 1, 1)
# Photos per trigger burst, on average: randint(1, 5) in `captures`.
BURST_MEAN = 3
# Of the boxes below 0.2 in Elliot's database, the share at 0.1 or more,
# the classification gate: 232K of 2.03M. It sets his embedding count.
WEAK_ABOVE_GATE = 0.11


@dataclass(frozen=True)
class DeploymentShape:
    photos: int
    folders: int = 1
    # (boxes per photo, share of boxes at 0.2 or more), copied from a
    # user's database. None: the default MegaDetector-like box model.
    rates: tuple[float, float] | None = None


@dataclass(frozen=True)
class ProjectShape:
    name: str
    # One tuple of deployments per site.
    sites: tuple[tuple[DeploymentShape, ...], ...]
    # Share of photos a person has verified, so the Verification card
    # and the progress bars have something to show.
    verified_share: float = 0.0


@dataclass(frozen=True)
class Shape:
    projects: tuple[ProjectShape, ...]
    species_count: int
    # Weight species 1/rank, so a few are common and many are rare.
    # False: every species equally likely.
    zipf_species: bool
    legacy_share: float
    # Mean photos per sequence. None: every burst is its own sequence.
    sequence_photos: int | None


def _uniform(name: str, n_sites: int, per_site: int, folders: int, photos: int,
             verified_share: float = 0.0) -> ProjectShape:
    deployment = DeploymentShape(photos=photos, folders=folders)
    return ProjectShape(name, ((deployment,) * per_site,) * n_sites, verified_share)


# (photos, boxes, boxes at 0.2 or more) for each of Elliot's 58
# deployments, from his addaxai-db-stats.txt (2026-11-21).
ELLIOT_DEPLOYMENTS = [
    (32583, 8629, 307), (2256, 4159, 1021), (6833, 8159, 861), (25971, 40477, 4798),
    (32499, 48728, 1636), (32565, 18834, 1809), (32868, 11625, 392), (32643, 24871, 2246),
    (32691, 44837, 2769), (33177, 10317, 303), (3090, 6452, 2374), (32665, 248114, 27699),
    (3366, 7113, 2027), (32736, 36281, 6068), (5982, 9433, 1950), (32640, 6341, 1709),
    (32478, 3866, 128), (2244, 1850, 573), (32472, 11770, 470), (34995, 54604, 13477),
    (5433, 11008, 6469), (2631, 2620, 467), (27584, 68435, 13304), (33415, 77581, 4524),
    (5901, 19262, 1409), (34165, 21547, 3085), (10185, 22570, 10483), (32585, 26627, 6664),
    (33660, 548017, 125829), (23430, 25937, 1970), (3540, 2674, 388), (19697, 14975, 2744),
    (34311, 49807, 13837), (35235, 434284, 123601), (35430, 24168, 836), (32495, 1564, 73),
    (32712, 55689, 8504), (2619, 6923, 1032), (32892, 35729, 4394), (32581, 9505, 2365),
    (2619, 6964, 1162), (3381, 9395, 2922), (33012, 22476, 735), (33366, 101353, 43041),
    (3408, 4473, 1199), (31881, 57755, 5764), (3150, 3007, 538), (4044, 3518, 1336),
    (2583, 4070, 1178), (32499, 54655, 2618), (32590, 17633, 1846), (11670, 26228, 13686),
    (2661, 2907, 927), (23640, 11971, 1243), (26681, 29631, 2740), (33785, 5082, 118),
    (9264, 33368, 3168), (33570, 111814, 51645),
]

SHAPES = {
    "simon": Shape(
        projects=(
            _uniform("Large", 1, 1, 100, 900_000),
            _uniform("Small", 3, 1, 1, 9_000, verified_share=0.10),
            _uniform("Normal 1", 2, 1, 1, 10_000),
            _uniform("Normal 2", 2, 1, 1, 10_000),
            _uniform("Normal 3", 2, 1, 1, 10_000),
        ),
        species_count=30,
        zipf_species=False,
        legacy_share=0.19,
        sequence_photos=None,
    ),
    "elliot": Shape(
        projects=(
            ProjectShape(
                "Elliot",
                tuple(
                    (DeploymentShape(photos, rates=(boxes / photos, strong / boxes)),)
                    for photos, boxes, strong in ELLIOT_DEPLOYMENTS
                ),
                # 2.2K of his 2.57M boxes are verified.
                verified_share=0.002,
            ),
        ),
        # About 2.57M boxes over 9,455 rows per label value.
        species_count=270,
        zipf_species=True,
        # His database is from after v7.2.0, so it has no legacy tail.
        legacy_share=0.0,
        # 1.27M photos in 20.6K sequences.
        sequence_photos=62,
    ),
}


@contextmanager
def step(what: str) -> Iterator[None]:
    started = time.perf_counter()
    print(f"{what}...", flush=True)
    yield
    print(f"  {what} took {time.perf_counter() - started:.0f}s", flush=True)


class Seeder:
    def __init__(self, db, args: argparse.Namespace, shape: Shape) -> None:
        from app.ml.taxonomy_db import ensure_builtin_labels

        self.db = db
        self.args = args
        self.shape = shape
        self.rng = random.Random(SEED)
        self.np_rng = np.random.default_rng(SEED)
        self.builtin_ids = ensure_builtin_labels(db, ["animal", "person", "vehicle"])
        self.legacy_share = (
            shape.legacy_share if args.legacy_share is None else args.legacy_share
        )
        self.legacy_owed = 0.0
        self.species = self._pick_species()
        self.species_weights = (
            [1 / rank for rank in range(1, len(self.species) + 1)]
            if shape.zipf_species
            else None
        )
        self.files: list[dict] = []
        self.boxes: list[dict] = []
        self.embed_ids: list[str] = []
        self.totals = {"files": 0, "boxes": 0}

    def uuid(self) -> str:
        """A uuid4 from the seeded generator, so reruns give the same ids."""
        return str(uuid.UUID(int=self.rng.getrandbits(128), version=4))

    def _pick_species(self) -> list:
        from app.ml.taxonomy_db import populate_taxonomy_from_csv
        from app.models import LabelTaxonomy

        csv_path = Path(self.args.taxonomy_csv).expanduser()
        if not csv_path.is_file():
            raise SystemExit(f"Taxonomy CSV not found: {csv_path}")
        populate_taxonomy_from_csv(self.args.classification_model, csv_path, self.db)
        rows = (
            self.db.query(LabelTaxonomy)
            .filter(
                LabelTaxonomy.classification_model_id == self.args.classification_model,
                LabelTaxonomy.level == "species",
            )
            .order_by(LabelTaxonomy.name)
            .all()
        )
        if not rows:
            raise SystemExit(f"No species-level rows in {csv_path}")
        return self.rng.sample(rows, min(self.shape.species_count, len(rows)))

    # Boxes ---------------------------------------------------------------

    def _box(self, file_id: str, category: str, confidence: float) -> SimpleNamespace:
        x, y = self.rng.uniform(0, 0.8), self.rng.uniform(0, 0.8)
        return SimpleNamespace(
            id=self.uuid(),
            file_id=file_id,
            category=category,
            confidence=round(confidence, 4),
            bbox_x=x,
            bbox_y=y,
            bbox_width=self.rng.uniform(0.05, 0.2),
            bbox_height=self.rng.uniform(0.05, 0.2),
            verified=False,
            label=None,
            label_confidence=None,
            label_taxonomy_id=None,
            common_name=None,
            scientific_name=None,
            original_label=None,
            original_label_confidence=None,
            classification_method=None,
        )

    def _label(self, box: SimpleNamespace) -> None:
        """Label a box the way the worker does, from the box's confidence."""
        from app.ml.taxonomic_rollup import resolve_label_names
        from app.ml.taxonomy_db import clear_classification

        if box.category != "animal" or box.confidence < self.project.classification_gate:
            clear_classification(box, self.builtin_ids)
            return
        if self.species_weights is None:
            row = self.rng.choice(self.species)
        else:
            row = self.rng.choices(self.species, self.species_weights)[0]
        box.label = row.name
        box.label_confidence = round(self.rng.uniform(0.3, 0.99), 4)
        box.label_taxonomy_id = row.id
        box.common_name, box.scientific_name = resolve_label_names(row.name, row)
        box.original_label = box.label
        box.original_label_confidence = box.label_confidence
        box.classification_method = "machine"

    @staticmethod
    def _category(kind: float) -> str:
        return "animal" if kind < 0.97 else ("person" if kind < 0.99 else "vehicle")

    def _default_boxes(self, file_id: str) -> list[SimpleNamespace]:
        """MegaDetector-like boxes: 40% of photos hold an animal."""
        rng = self.rng
        kind = rng.random()
        boxes = []
        if kind < 0.60:  # blank: only weak boxes, if any
            for _ in range(rng.randint(0, 3)):
                boxes.append(self._box(file_id, "animal", rng.uniform(0.01, 0.2)))
        else:
            category = self._category(kind)
            for _ in range(rng.randint(1, 2)):
                boxes.append(self._box(file_id, category, rng.uniform(0.5, 0.99)))
            for _ in range(rng.randint(1, 3)):
                boxes.append(self._box(file_id, "animal", rng.uniform(0.01, 0.2)))
        return boxes

    def _rated_boxes(self, file_id: str, rates: tuple[float, float]) -> list[SimpleNamespace]:
        """Boxes at a deployment's own rates, copied from a user's database.

        The count per photo is spread evenly around the mean, and each box
        is strong (0.2 or more) with the deployment's share. A photo
        without a strong box is a blank one.
        """
        rng = self.rng
        per_photo, strong_share = rates
        x = rng.uniform(0, 2 * per_photo)
        count = int(x) + (rng.random() < x - int(x))
        boxes = []
        for _ in range(count):
            if rng.random() < strong_share:
                category, confidence = self._category(rng.random()), rng.uniform(0.2, 0.99)
            elif rng.random() < WEAK_ABOVE_GATE:
                category, confidence = "animal", rng.uniform(0.1, 0.2)
            else:
                category, confidence = "animal", rng.uniform(0.01, 0.1)
            boxes.append(self._box(file_id, category, confidence))
        return boxes

    def boxes_for(self, file_id: str, rates: tuple[float, float] | None) -> list[SimpleNamespace]:
        """Boxes for one photo, plus the legacy tail."""
        rng = self.rng
        if rates is None:
            boxes = self._default_boxes(file_id)
        else:
            boxes = self._rated_boxes(file_id, rates)
        # Legacy tail: L legacy rows next to n others make up share s when
        # L = n * s / (1 - s). The fraction carries over to the next photo,
        # so the share comes out exact over the whole database.
        share = self.legacy_share
        self.legacy_owed += len(boxes) * share / (1 - share)
        n_legacy = int(self.legacy_owed)
        self.legacy_owed -= n_legacy
        for _ in range(n_legacy):
            boxes.append(self._box(file_id, "animal", rng.uniform(0.005, 0.0099)))
        for box in boxes:
            self._label(box)
        return boxes

    # Files ---------------------------------------------------------------

    def add_photo(self, deployment_id: str, path: str, captured: datetime,
                  rates: tuple[float, float] | None) -> None:
        from app.ml.observation_type import derive_observation_type

        file_id = self.uuid()
        boxes = self.boxes_for(file_id, rates)
        verified = self.rng.random() < self.verified_share
        for box in boxes:
            box.verified = verified
            if box.confidence >= self.project.classification_gate:
                self.embed_ids.append(box.id)
        self.files.append(
            {
                "id": file_id,
                "deployment_id": deployment_id,
                "file_path": path,
                "file_type": "image",
                "file_format": "jpg",
                "captured_at_local": captured,
                "observation_type": derive_observation_type(
                    boxes, self.project.counting_threshold
                ),
                "verified": verified,
                "width_px": 2048,
                "height_px": 1536,
                "size_bytes": 900_000,
            }
        )
        self.boxes.extend(vars(b) for b in boxes)
        self.totals["files"] += 1
        self.totals["boxes"] += len(boxes)
        if len(self.boxes) >= INSERT_CHUNK:
            self.flush()

    def flush(self) -> None:
        from sqlalchemy import insert

        from app.models import Detection, File

        if self.files:
            self.db.execute(insert(File), self.files)
        if self.boxes:
            self.db.execute(insert(Detection), self.boxes)
        self.db.commit()
        self.files, self.boxes = [], []
        print(f"  {self.totals['files']:,} photos, {self.totals['boxes']:,} boxes", end="\r")

    def captures(self, count: int) -> Iterator[datetime]:
        """Trigger bursts of 1 to 5 photos 2 s apart.

        A gap of 31 min to 6 h starts a new sequence, and a new independent
        event at the default 30-min interval. With ``sequence_photos`` set,
        most gaps are 10 to 110 s instead, inside the 2-min sequence gap, so
        the bursts form sequences of about that many photos.
        """
        sequence_photos = self.shape.sequence_photos
        t = SEASON_START + timedelta(minutes=self.rng.uniform(0, 600))
        made = 0
        while made < count:
            for _ in range(min(self.rng.randint(1, 5), count - made)):
                yield t
                t += timedelta(seconds=2)
                made += 1
            if sequence_photos is None or self.rng.random() < BURST_MEAN / sequence_photos:
                t += timedelta(minutes=self.rng.uniform(31, 360))
            else:
                t += timedelta(seconds=self.rng.uniform(10, 110))

    def seed_project(self, spec: ProjectShape) -> str:
        from app.models import Deployment, Project, Site

        self.project = Project(
            id=self.uuid(),
            name=spec.name,
            timezone=TIMEZONE,
            classification_model_id=self.args.classification_model,
            embedding_model_id=EMBEDDING_MODEL_ID,
        )
        self.db.add(self.project)
        self.db.flush()
        self.verified_share = spec.verified_share
        slug = spec.name.lower().replace(" ", "-")
        for s, site_deployments in enumerate(spec.sites):
            site = Site(
                id=self.uuid(),
                project_id=self.project.id,
                name=f"{slug} site {s + 1}",
                latitude=-33.8 + s * 0.05,
                longitude=151.0 + s * 0.05,
            )
            self.db.add(site)
            for d, planned in enumerate(site_deployments):
                folders = planned.folders
                photos_per_folder = max(1, round(planned.photos * self.args.scale / folders))
                folder = f"/seed/{slug}/site-{s + 1}-deployment-{d + 1}"
                deployment = Deployment(
                    id=self.uuid(),
                    project_id=self.project.id,
                    site_id=site.id,
                    folder_path=folder,
                    start_date_local=SEASON_START.date(),
                )
                self.db.add(deployment)
                self.db.flush()
                last = SEASON_START
                for f in range(folders):
                    sub = f"{folder}/cam-{f + 1:03d}" if folders > 1 else folder
                    for i, captured in enumerate(self.captures(photos_per_folder)):
                        self.add_photo(
                            deployment.id, f"{sub}/IMG_{i + 1:06d}.JPG", captured, planned.rates
                        )
                        last = max(last, captured)
                deployment.end_date_local = last.date()
                self.flush()
        self.db.commit()
        print()
        return self.project.id

    # Embeddings ----------------------------------------------------------

    def seed_embeddings(self) -> int:
        from app.ml.embedding_utils import save_embeddings_to_db
        from app.models import Job

        job = Job(type="re_embedding", status="completed")
        self.db.add(job)
        self.db.commit()
        saved = 0
        with tempfile.TemporaryDirectory() as tmp:
            npz = Path(tmp) / "embeddings.npz"
            for start in range(0, len(self.embed_ids), EMBEDDING_CHUNK):
                ids = self.embed_ids[start : start + EMBEDDING_CHUNK]
                vectors = self.np_rng.standard_normal((len(ids), EMBEDDING_DIM))
                vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
                np.savez(npz, **dict(zip(ids, vectors.astype(np.float16), strict=True)))
                saved += save_embeddings_to_db(
                    npz, job.id, EMBEDDING_MODEL_ID, EMBEDDING_DIM, self.db
                )
                print(f"  {saved:,} of {len(self.embed_ids):,} embeddings", end="\r")
        print()
        return saved


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--classification-model", required=True,
                        help="model id the species come from, e.g. SPECIESNET-v4-0-2-A")
    parser.add_argument("--taxonomy-csv", required=True,
                        help="that model's taxonomy.csv")
    parser.add_argument("--shape", choices=sorted(SHAPES), default="simon",
                        help="whose database to copy (default simon)")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="multiplies every photo count (default 1: the full size)")
    parser.add_argument("--legacy-share", type=float, default=None,
                        help="share of boxes between 0.005 and 0.01 "
                             "(default: the shape's, 0.19 for simon, 0 for elliot)")
    args = parser.parse_args()
    shape = SHAPES[args.shape]
    if args.legacy_share is not None and not 0 <= args.legacy_share < 1:
        parser.error("--legacy-share must be at least 0 and below 1")

    db = open_seed_sandbox()

    from app.api.crud.sequence import generate_sequences_for_project
    from app.db.base import refresh_query_statistics

    started = time.perf_counter()
    seeder = Seeder(db, args, shape)
    project_ids = []
    for spec in shape.projects:
        with step(f"Project {spec.name}"):
            project_ids.append(seeder.seed_project(spec))
    with step("Embeddings"):
        embeddings = seeder.seed_embeddings()
    for name, project_id in zip((p.name for p in shape.projects), project_ids, strict=True):
        with step(f"Sequences for {name}"):
            sequences = generate_sequences_for_project(db, project_id)
        print(f"  {sequences:,} sequences")
    with step("ANALYZE"):
        refresh_query_statistics(db)
        db.commit()
    db.close()

    print(
        f"Seeded {seeder.totals['files']:,} photos, {seeder.totals['boxes']:,} boxes, "
        f"{embeddings:,} embeddings in {time.perf_counter() - started:.0f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

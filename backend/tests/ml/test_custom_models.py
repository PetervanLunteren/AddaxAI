"""Coverage for safe local custom-model pack management."""

import asyncio
import hashlib
import json
import os
import re
import shutil
import uuid
from pathlib import Path

import httpx
import pytest

from app.api.schemas.custom_models import CustomModelCreate
from app.db.base import get_db
from app.ml.catalog_updater import ModelCatalogUpdater
from app.ml.custom_model_manager import CustomModelError, CustomModelManager
from app.ml.manifest_manager import ManifestManager
from app.ml.model_storage import ModelStorage
from app.ml.model_usage import acquire_model_usage
from tests.conftest import make_project


def _model_roots(tmp_path: Path) -> Path:
    models_dir = tmp_path / "user-data" / "models"
    for category in ("det", "cls", "emb"):
        (models_dir / category).mkdir(parents=True)
    return models_dir


def _manager(tmp_path: Path) -> CustomModelManager:
    models_dir = _model_roots(tmp_path)
    return CustomModelManager(models_dir, ManifestManager(models_dir))


def _detection_pack(path: Path, *, companion: bool = True) -> Path:
    path.mkdir(parents=True)
    (path / "weights.pt").write_bytes(b"small test weights")
    if companion:
        (path / "labels.txt").write_text("fox\ndeer\n", encoding="utf-8")
    return path


def _classification_pack(path: Path) -> Path:
    path.mkdir(parents=True)
    (path / "weights.pth").write_bytes(b"classifier test weights")
    (path / "inference.py").write_text(
        "class ModelInference:\n"
        "    def check_gpu(self): pass\n"
        "    def load_model(self): pass\n"
        "    def get_crop(self): pass\n"
        "    def get_classification(self): pass\n"
        "    def get_class_names(self): pass\n",
        encoding="utf-8",
    )
    (path / "taxonomy.csv").write_text("id,name\n0,fox\n", encoding="utf-8")
    return path


def _create_payload(pack: Path, env: str, **changes) -> CustomModelCreate:
    values = {
        "type": "detection",
        "source_path": str(pack),
        "friendly_name": "Local wildlife detector",
        "env": env,
        "model_fname": "weights.pt",
        "detector_backend": "yolo",
        "class_names": ["fox", "deer"],
    }
    values.update(changes)
    return CustomModelCreate(**values)


def test_create_copies_detection_pack_with_managed_local_manifest(tmp_path: Path):
    manager = _manager(tmp_path)
    source = _detection_pack(tmp_path / "source-detector")
    env = manager.environments()[0]

    result = manager.create(_create_payload(source, env))

    assert re.fullmatch(r"custom-[0-9a-f]{32}", result["model_id"])
    assert result["type"] == "detection"
    assert result["local_only"] is True
    assert result["managed"] is True
    assert result["class_names"] == {"0": "fox", "1": "deer"}
    destination = manager.models_dir / "det" / result["model_id"]
    assert (destination / "weights.pt").read_bytes() == b"small test weights"
    assert (destination / "labels.txt").read_text(encoding="utf-8") == "fox\ndeer\n"
    manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["managed"] is True
    assert manifest["local_only"] is True
    assert manifest["weights_sha256"]
    assert manager.manifest_manager.get_model(result["model_id"]).detector_backend == "yolo"


def test_create_requires_addaxai_classifier_inference_pack(tmp_path: Path):
    manager = _manager(tmp_path)
    source = _classification_pack(tmp_path / "source-classifier")
    env = manager.environments()[0]
    payload = CustomModelCreate(
        type="classification",
        source_path=str(source),
        friendly_name="Local fox classifier",
        env=env,
        model_fname="weights.pth",
        region="global",
        full_image_cls=True,
    )

    result = manager.create(payload)

    assert result["type"] == "classification"
    assert result["full_image_cls"] is True
    destination = manager.models_dir / "cls" / result["model_id"]
    assert (destination / "inference.py").is_file()
    assert (destination / "taxonomy.csv").is_file()

    (source / "inference.py").write_text("class Other: pass\n", encoding="utf-8")
    with pytest.raises(CustomModelError, match="ModelInference"):
        manager.create(payload.model_copy(update={"friendly_name": "Invalid classifier"}))


def test_rejects_paths_missing_weights_and_unsafe_rtdetr_config(tmp_path: Path):
    manager = _manager(tmp_path)
    env = manager.environments()[0]
    source = _detection_pack(tmp_path / "source")

    with pytest.raises(CustomModelError, match="absolute"):
        manager.create(_create_payload(source, env, source_path="relative/model"))
    with pytest.raises(CustomModelError, match="weight file"):
        manager.create(_create_payload(source, env, model_fname="missing.pt"))

    with pytest.raises(CustomModelError, match="safe relative|inside"):
        manager.create(
            _create_payload(
                source,
                env,
                detector_backend="rtdetrv2",
                detector_config_fname="../outside.yml",
            )
        )

    try:
        (source / "link.pt").symlink_to(source / "weights.pt")
    except OSError as exc:
        pytest.skip(f"symlink creation is not permitted on this Windows host: {exc}")
    with pytest.raises(CustomModelError, match="links|junctions"):
        manager.create(_create_payload(source, env, model_fname="weights.pt"))


def test_rejects_source_that_contains_the_managed_models_directory(tmp_path: Path):
    manager = _manager(tmp_path)
    user_data_root = manager.models_dir.parent
    (user_data_root / "weights.pt").write_bytes(b"ancestor must not be copied")

    with pytest.raises(CustomModelError, match="outside AddaxAI's managed model folders"):
        manager.create(_create_payload(user_data_root, manager.environments()[0]))

    assert list((manager.models_dir / "det").iterdir()) == []


def test_failed_copy_removes_partial_staging_directory(tmp_path: Path, monkeypatch):
    manager = _manager(tmp_path)
    source = _detection_pack(tmp_path / "source")
    env = manager.environments()[0]
    fixed_id = uuid.UUID("12345678123456781234567812345678")
    monkeypatch.setattr("app.ml.custom_model_manager.uuid.uuid4", lambda: fixed_id)

    def fail_after_partial_copy(_source, destination, **_kwargs):
        destination.mkdir()
        (destination / "partial.bin").write_bytes(b"partial")
        raise OSError("simulated disk full")

    monkeypatch.setattr("app.ml.custom_model_manager.shutil.copytree", fail_after_partial_copy)
    with pytest.raises(OSError, match="disk full"):
        manager.create(_create_payload(source, env))

    assert list((manager.models_dir / "det").iterdir()) == []


def test_copy_verifies_model_weights_before_publishing_pack(tmp_path: Path, monkeypatch):
    manager = _manager(tmp_path)
    source = _detection_pack(tmp_path / "source")
    env = manager.environments()[0]
    real_copytree = shutil.copytree

    def alter_weights_after_copy(source_path, destination, **kwargs):
        copied = real_copytree(source_path, destination, **kwargs)
        (copied / "weights.pt").write_bytes(b"changed during copy")
        return copied

    monkeypatch.setattr("app.ml.custom_model_manager.shutil.copytree", alter_weights_after_copy)
    with pytest.raises(CustomModelError, match="changed while the pack was being copied"):
        manager.create(_create_payload(source, env))

    assert list((manager.models_dir / "det").iterdir()) == []


def test_weight_readiness_hashes_each_file_fingerprint_only_once(tmp_path: Path, monkeypatch):
    from app.ml import model_storage as storage_module
    from app.ml.schemas.model_manifest import ModelManifest

    models_dir = tmp_path / "models"
    weight_path = models_dir / "det" / "local-id" / "weights.pt"
    weight_path.parent.mkdir(parents=True)
    weight_path.write_bytes(b"weights")
    manifest = ModelManifest(
        model_id="local-id",
        friendly_name="Test model",
        env="addaxai-base",
        model_fname="weights.pt",
        description="test",
        developer="test",
        info_url="",
        min_app_version="0",
        model_category="detection",
        local_only=True,
        weights_sha256=hashlib.sha256(b"weights").hexdigest(),
    )
    real_hash = storage_module._sha256_file
    calls: list[Path] = []

    def count_hash(path: Path) -> str:
        calls.append(path)
        return real_hash(path)

    storage_module._cached_sha256_file.cache_clear()
    monkeypatch.setattr(storage_module, "_sha256_file", count_hash)
    try:
        storage = ModelStorage(models_dir)
        assert storage.check_weights_ready(manifest)
        assert storage.check_weights_ready(manifest)
        assert len(calls) == 1

        stat = weight_path.stat()
        weight_path.write_bytes(b"tamper")
        os.utime(weight_path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
        assert not storage.check_weights_ready(manifest)
        assert len(calls) == 2
    finally:
        storage_module._cached_sha256_file.cache_clear()


def test_id_collision_does_not_overwrite_pack_in_another_category(tmp_path: Path, monkeypatch):
    manager = _manager(tmp_path)
    source = _detection_pack(tmp_path / "source")
    env = manager.environments()[0]
    fixed_id = uuid.UUID("12345678123456781234567812345678")
    monkeypatch.setattr("app.ml.custom_model_manager.uuid.uuid4", lambda: fixed_id)
    existing = manager.models_dir / "cls" / f"custom-{fixed_id.hex}"
    existing.mkdir()
    (existing / "keep.txt").write_text("do not replace", encoding="utf-8")

    with pytest.raises(FileExistsError, match="collision"):
        manager.create(_create_payload(source, env))

    assert (existing / "keep.txt").read_text(encoding="utf-8") == "do not replace"
    assert list((manager.models_dir / "det").iterdir()) == []


def test_display_edit_is_allowed_but_inference_semantics_are_immutable(tmp_path: Path):
    manager = _manager(tmp_path)
    source = _detection_pack(tmp_path / "source")
    env = manager.environments()[0]
    created = manager.create(_create_payload(source, env))

    updated = manager.update(created["model_id"], {"friendly_name": "Renamed local detector"})
    assert updated["friendly_name"] == "Renamed local detector"

    for changes in (
        {"model_fname": "other.pt"},
        {"detector_backend": "rfdetr"},
        {"class_names": {"0": "new class"}},
        {"env": manager.environments()[-1]},
        {"full_image_cls": True},
    ):
        with pytest.raises(CustomModelError, match="fixed after registration"):
            manager.update(created["model_id"], changes)


def test_delete_protects_project_references_and_active_inference(tmp_path: Path, db):
    manager = _manager(tmp_path)
    source = _detection_pack(tmp_path / "source")
    created = manager.create(_create_payload(source, manager.environments()[0]))

    make_project(db, name="custom-model-reference", detection_model_id=created["model_id"])
    with pytest.raises(CustomModelError, match="referenced by a project"):
        manager.delete(created["model_id"], db)
    db.rollback()

    release = acquire_model_usage([created["model_id"]])
    try:
        with pytest.raises(CustomModelError, match="active inference"):
            manager.delete(created["model_id"], db)
    finally:
        release()

    manager.delete(created["model_id"], db)
    assert manager.list_models() == []


def test_catalog_collision_and_stale_scan_preserve_managed_manifest(tmp_path: Path):
    manager = _manager(tmp_path)
    source = _detection_pack(tmp_path / "source")
    created = manager.create(_create_payload(source, manager.environments()[0]))
    model_dir = manager.models_dir / "det" / created["model_id"]
    manifest_path = model_dir / "manifest.json"
    original = manifest_path.read_bytes()
    updater = ModelCatalogUpdater(manager.models_dir)
    collision = json.loads(original)
    collision["friendly_name"] = "Central catalog overwrite"

    assert updater.write_manifest("det", collision) == "unchanged"
    assert manifest_path.read_bytes() == original
    assert asyncio.run(updater._find_stale_files("det", collision)) is None


@pytest.fixture
def local_model_api(tmp_path: Path, db, monkeypatch):
    """Real API routes over an isolated models directory and the test DB."""
    from app import main
    from app.api.routers import ml_models

    models_dir = _model_roots(tmp_path)
    manager = ManifestManager(models_dir)
    monkeypatch.setattr(ml_models, "_get_managers", lambda: (manager, object(), object()))

    app = main.create_app()

    def override_db():
        yield db

    app.dependency_overrides[get_db] = override_db
    return app, models_dir


@pytest.mark.asyncio
async def test_local_custom_model_api_crud_and_model_selection_lists(
    local_model_api, tmp_path: Path, db, monkeypatch
):
    app, models_dir = local_model_api
    source = _detection_pack(tmp_path / "api-source")
    env = CustomModelManager.environments()[0]

    transport = httpx.ASGITransport(app=app, client=("127.0.0.1", 54123))
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        assert (await client.get("/api/ml/custom-models")).json()["models"] == []
        created_response = await client.post(
            "/api/ml/custom-models",
            json={
                "type": "detection",
                "source_path": str(source),
                "friendly_name": "API detector",
                "env": env,
                "model_fname": "weights.pt",
                "detector_backend": "yolo",
            },
        )
        assert created_response.status_code == 201, created_response.text
        created = created_response.json()
        model_id = created["model_id"]
        assert created["managed"] and created["local_only"]
        assert model_id in {
            row["model_id"]
            for row in (await client.get("/api/ml/custom-models")).json()["models"]
        }
        selectable = (await client.get("/api/ml/models/detection")).json()
        assert next(row for row in selectable if row["model_id"] == model_id)["local_only"] is True

        remote_transport = httpx.ASGITransport(app=app, client=("192.168.1.22", 54124))
        async with httpx.AsyncClient(
            transport=remote_transport, base_url="http://testserver"
        ) as remote_client:
            remote_models = (await remote_client.get("/api/ml/models/detection")).json()
        assert model_id in {row["model_id"] for row in remote_models}

        invalid_pack = await client.post(
            "/api/ml/custom-models",
            json={
                "type": "detection",
                "source_path": str(tmp_path / "missing-pack"),
                "friendly_name": "Missing pack",
                "env": env,
                "model_fname": "weights.pt",
                "detector_backend": "yolo",
            },
        )
        assert invalid_pack.status_code == 400

        collision_id = "custom-12345678123456781234567812345678"
        collision = models_dir / "cls" / collision_id
        collision.mkdir()
        (collision / "keep.txt").write_text("untouched", encoding="utf-8")
        with monkeypatch.context() as collision_patch:
            collision_patch.setattr(
                "app.ml.custom_model_manager.uuid.uuid4",
                lambda: uuid.UUID("12345678123456781234567812345678"),
            )
            collision_response = await client.post(
                "/api/ml/custom-models",
                json={
                    "type": "detection",
                    "source_path": str(source),
                    "friendly_name": "Colliding pack",
                    "env": env,
                    "model_fname": "weights.pt",
                    "detector_backend": "yolo",
                },
            )
        assert collision_response.status_code == 409
        assert (collision / "keep.txt").read_text(encoding="utf-8") == "untouched"

        changed = await client.put(
            f"/api/ml/custom-models/{model_id}",
            json={"friendly_name": "API detector renamed"},
        )
        assert changed.status_code == 200
        assert changed.json()["friendly_name"] == "API detector renamed"
        semantic_edit = await client.put(
            f"/api/ml/custom-models/{model_id}", json={"class_names": {"0": "fox"}}
        )
        assert semantic_edit.status_code == 422

        reference = make_project(db, name="api-custom-model-reference", detection_model_id=model_id)
        referenced_delete = await client.delete(f"/api/ml/custom-models/{model_id}")
        assert referenced_delete.status_code == 409
        db.delete(reference)
        db.flush()
        assert (await client.delete(f"/api/ml/custom-models/{model_id}")).status_code == 204
        assert not (models_dir / "det" / model_id).exists()


@pytest.mark.asyncio
async def test_custom_model_api_rejects_source_ancestor_of_managed_models(
    local_model_api, tmp_path: Path
):
    app, models_dir = local_model_api
    user_data_root = models_dir.parent
    (user_data_root / "weights.pt").write_bytes(b"ancestor must not be copied")
    env = CustomModelManager.environments()[0]
    transport = httpx.ASGITransport(app=app, client=("127.0.0.1", 54125))
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post(
            "/api/ml/custom-models",
            json={
                "type": "detection",
                "source_path": str(user_data_root),
                "friendly_name": "Models-root recursion attempt",
                "env": env,
                "model_fname": "weights.pt",
                "detector_backend": "yolo",
            },
        )

    assert response.status_code == 400
    assert "outside AddaxAI's managed model folders" in response.json()["detail"]
    assert list((models_dir / "det").iterdir()) == []


@pytest.mark.asyncio
async def test_custom_model_management_api_rejects_non_loopback_clients(tmp_path: Path, monkeypatch):
    from app import main
    from app.api.routers import ml_models

    models_dir = _model_roots(tmp_path)
    manifest_manager = ManifestManager(models_dir)
    monkeypatch.setattr(
        ml_models,
        "_get_managers",
        lambda: (manifest_manager, object(), object()),
    )
    app = main.create_app()
    transport = httpx.ASGITransport(
        app=app,
        client=("192.168.1.22", 54123),
        raise_app_exceptions=False,
    )
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.get("/api/ml/custom-models")

    assert response.status_code == 403


@pytest.mark.asyncio
async def test_custom_model_api_rejects_remote_origin_from_loopback_client(
    tmp_path: Path, db, monkeypatch
):
    from app import main
    from app.api.routers import ml_models

    models_dir = _model_roots(tmp_path)
    manifest_manager = ManifestManager(models_dir)
    monkeypatch.setattr(
        ml_models,
        "_get_managers",
        lambda: (manifest_manager, object(), object()),
    )
    app = main.create_app()

    def override_db():
        yield db

    app.dependency_overrides[get_db] = override_db
    source = _detection_pack(tmp_path / "origin-source")
    payload = {
        "type": "detection",
        "source_path": str(source),
        "friendly_name": "Origin-checked detector",
        "env": CustomModelManager.environments()[0],
        "model_fname": "weights.pt",
        "detector_backend": "yolo",
    }
    transport = httpx.ASGITransport(app=app, client=("127.0.0.1", 54126))
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        denied_create = await client.post(
            "/api/ml/custom-models",
            json=payload,
            headers={"Origin": "https://attacker.invalid"},
        )
        denied_delete = await client.delete(
            "/api/ml/custom-models/custom-0123456789abcdef0123456789abcdef",
            headers={"Origin": "https://attacker.invalid"},
        )
        assert denied_create.status_code == 403
        assert denied_delete.status_code == 403
        assert list((models_dir / "det").iterdir()) == []

        localhost_transport = httpx.ASGITransport(
            app=app, client=("localhost", 54127)
        )
        async with httpx.AsyncClient(
            transport=localhost_transport, base_url="http://testserver"
        ) as localhost_client:
            denied_localhost_origin = await localhost_client.post(
                "/api/ml/custom-models",
                json=payload,
                headers={"Origin": "https://attacker.invalid"},
            )
        assert denied_localhost_origin.status_code == 403
        assert list((models_dir / "det").iterdir()) == []

        allowed_create = await client.post(
            "/api/ml/custom-models",
            json=payload,
            headers={"Origin": "http://127.0.0.1:5173"},
        )
        assert allowed_create.status_code == 201, allowed_create.text
        model_id = allowed_create.json()["model_id"]
        allowed_delete = await client.delete(
            f"/api/ml/custom-models/{model_id}",
            headers={"Origin": "http://localhost:8000"},
        )
        assert allowed_delete.status_code == 204


def test_deployment_job_schema_accepts_custom_model_ids():
    from app.api.schemas.job import DeploymentAnalysisPayload

    payload = DeploymentAnalysisPayload(
        project_id="project-id",
        folder_path="C:/camera-traps",
        detection_model="custom-0123456789abcdef0123456789abcdef",
        classification_model="custom-fedcba9876543210fedcba9876543210",
    )
    assert payload.detection_model.startswith("custom-")
    assert payload.classification_model.startswith("custom-")

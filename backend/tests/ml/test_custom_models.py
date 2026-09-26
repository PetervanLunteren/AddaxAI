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


def _mdv6_style_rtdetrv2_pack(path: Path) -> Path:
    path.mkdir(parents=True)
    (path / "best.pth").write_bytes(b"checkpoint bytes must not be read during inspect")
    (path / "rtdetrv2_r101_mdv6_20cls.yml").write_text(
        "__include__: []\n"
        "num_classes: 20\n"
        "PResNet: {}\n"
        "RTDETRTransformerv2: {}\n",
        encoding="utf-8",
    )
    labels = [
        "person", "bird", "boar", "deer", "tanuki", "araiguma", "fox", "hakubishin",
        "rabbit", "ten", "itachi", "monkey", "bear", "kamosika", "japanese squirrel",
        "mouse", "anaguma", "cat", "other", "vehicle",
    ]
    (path / "rtdetrv2_mdv6_20cls_dataset.yaml").write_text(
        "names: " + json.dumps(labels, ensure_ascii=False) + "\n", encoding="utf-8"
    )
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


def test_create_does_not_guess_detector_backend_or_variant(tmp_path: Path):
    manager = _manager(tmp_path)
    source = _detection_pack(tmp_path / "explicit-settings")
    env = manager.environments()[0]

    with pytest.raises(CustomModelError, match="Select detector_backend explicitly"):
        manager.create(_create_payload(source, env, detector_backend=None))
    with pytest.raises(CustomModelError, match="explicit detector_model_class"):
        manager.create(
            _create_payload(
                source,
                env,
                detector_backend="rfdetr",
                detector_model_class=None,
            )
        )
    with pytest.raises(CustomModelError, match="explicit detector_model_variant"):
        manager.create(
            _create_payload(
                source,
                env,
                detector_backend="rtdetr",
                detector_model_variant=None,
            )
        )


def test_create_validates_rtdetrv2_yaml_and_preserves_safe_include_config(tmp_path: Path):
    manager = _manager(tmp_path)
    env = "rtdetr"
    invalid = _detection_pack(tmp_path / "invalid-rtdetrv2")
    (invalid / "config.yml").write_text("RTDETRTransformerv2: [\n", encoding="utf-8")
    with pytest.raises(CustomModelError, match="Could not parse YAML file config.yml"):
        manager.create(
            _create_payload(
                invalid,
                env,
                detector_backend="rtdetrv2",
                detector_config_fname="config.yml",
            )
        )

    valid = _mdv6_style_rtdetrv2_pack(tmp_path / "valid-rtdetrv2")
    config_path = valid / "rtdetrv2_r101_mdv6_20cls.yml"
    config_path.write_text(
        "__include__:\n"
        "  - ../third_party/rtdetr/rtdetrv2_pytorch/configs/rtdetr/include.yml\n"
        "num_classes: 20\n"
        "PResNet: {}\n"
        "RTDETRTransformerv2: {}\n",
        encoding="utf-8",
    )
    result = manager.create(
        CustomModelCreate(
            type="detection",
            source_path=str(valid),
            friendly_name="Included RT-DETRv2 config",
            env=env,
            model_fname="best.pth",
            detector_backend="rtdetrv2",
            detector_config_fname="rtdetrv2_r101_mdv6_20cls.yml",
        )
    )

    assert result["detector_backend"] == "rtdetrv2"
    assert result["detector_config_fname"] == "rtdetrv2_r101_mdv6_20cls.yml"


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
async def test_custom_model_management_api_rejects_non_loopback_clients(
    tmp_path: Path, monkeypatch
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


def test_inspect_identifies_mdv6_rtdetrv2_pack_without_reading_weights(
    tmp_path: Path, monkeypatch
):
    manager = _manager(tmp_path)
    source = _mdv6_style_rtdetrv2_pack(tmp_path / "mdv6-custom")
    monkeypatch.setattr(
        "app.ml.custom_model_manager._sha256_file",
        lambda _path: pytest.fail("inspect must not read or hash checkpoint contents"),
    )

    result = manager.inspect(str(source))

    assert result["type_candidates"] == ["detection"]
    assert result["suggested_type"] == "detection"
    assert result["weights"] == ["best.pth"]
    assert result["suggested_model_fname"] == "best.pth"
    assert result["detector_backend_candidates"] == ["rtdetrv2"]
    assert result["suggested_detector_backend"] == "rtdetrv2"
    assert result["suggested_detector_config_fname"] == "rtdetrv2_r101_mdv6_20cls.yml"
    assert result["suggested_env"] == "rtdetr"
    assert result["suggested_class_names_source"] == "rtdetrv2_mdv6_20cls_dataset.yaml"
    assert result["suggested_class_names"]["0"] == "person"
    assert result["suggested_class_names"]["19"] == "vehicle"
    assert result["total_file_count"] == 3
    assert {row["path"] for row in result["files"]} == {
        "best.pth",
        "rtdetrv2_r101_mdv6_20cls.yml",
        "rtdetrv2_mdv6_20cls_dataset.yaml",
    }
    assert not list((manager.models_dir / "det").iterdir())
    assert not list((manager.models_dir / "cls").iterdir())


def test_inspect_reports_ambiguous_type_and_multiple_weights(tmp_path: Path):
    manager = _manager(tmp_path)
    source = _mdv6_style_rtdetrv2_pack(tmp_path / "ambiguous")
    (source / "weights.pth").write_bytes(b"second checkpoint")
    (source / "inference.py").write_text(
        "class ModelInference:\n"
        "    def check_gpu(self): pass\n"
        "    def load_model(self): pass\n"
        "    def get_crop(self): pass\n"
        "    def get_classification(self): pass\n"
        "    def get_class_names(self): pass\n",
        encoding="utf-8",
    )

    result = manager.inspect(str(source))

    assert result["type_candidates"] == ["classification", "detection"]
    assert result["suggested_type"] is None
    assert result["weights"] == ["best.pth", "weights.pth"]
    assert result["suggested_model_fname"] is None
    assert (
        "Choose whether this pack is for detection or classification."
        in result["missing_required"]
    )
    assert "Choose one model weight file." in result["missing_required"]


def test_inspect_recognizes_compatible_classification_pack(tmp_path: Path):
    manager = _manager(tmp_path)
    source = _classification_pack(tmp_path / "classifier-pack")

    result = manager.inspect(str(source))

    assert result["type_candidates"] == ["classification"]
    assert result["suggested_type"] == "classification"
    assert result["classifier_inference_compatible"] is True
    assert result["suggested_model_fname"] == "weights.pth"
    assert result["suggested_detector_backend"] is None
    assert result["suggested_env"] is None
    assert "Choose a packaged inference environment." in result["missing_required"]


def test_inspect_does_not_guess_between_different_dataset_label_files(tmp_path: Path):
    manager = _manager(tmp_path)
    source = _detection_pack(tmp_path / "multiple-labels")
    (source / "dataset-a.yaml").write_text('names: ["fox", "deer"]\n', encoding="utf-8")
    (source / "dataset-b.yaml").write_text('names: ["person", "vehicle"]\n', encoding="utf-8")

    result = manager.inspect(str(source))

    assert result["suggested_class_names"] is None
    assert result["suggested_class_names_source"] is None
    assert len(result["dataset_candidates"]) == 2
    assert (
        "Choose the dataset YAML that defines this model's class labels."
        in result["missing_required"]
    )


def test_inspect_marks_manifest_backend_conflicting_with_rtdetrv2_config_ambiguous(
    tmp_path: Path,
):
    manager = _manager(tmp_path)
    source = _mdv6_style_rtdetrv2_pack(tmp_path / "backend-conflict")
    (source / "manifest.json").write_text(
        json.dumps({"detector_backend": "yolo", "env": "pytorch", "model_fname": "best.pth"}),
        encoding="utf-8",
    )

    result = manager.inspect(str(source))

    assert result["detector_backend_candidates"] == ["rtdetrv2", "yolo"]
    assert result["suggested_detector_backend"] is None
    assert result["suggested_env"] is None
    assert result["suggested_env_source"] is None
    assert "Choose the detection backend." in result["missing_required"]
    assert "Choose a packaged inference environment." in result["missing_required"]


def test_inspect_leaves_backend_and_environment_unselected_when_unknown(tmp_path: Path):
    manager = _manager(tmp_path)
    source = _detection_pack(tmp_path / "unknown-detector")

    result = manager.inspect(str(source))

    assert result["suggested_type"] == "detection"
    assert result["suggested_detector_backend"] is None
    assert result["detector_backend_candidates"] == ["yolo", "rfdetr", "rtdetr", "rtdetrv2"]
    assert result["suggested_env"] is None
    assert "Choose the detection backend." in result["missing_required"]
    assert "Choose a packaged inference environment." in result["missing_required"]


def test_inspect_handles_empty_and_invalid_yaml_packs(tmp_path: Path):
    manager = _manager(tmp_path)
    empty = tmp_path / "empty"
    empty.mkdir()
    empty_result = manager.inspect(str(empty))
    assert empty_result["type_candidates"] == []
    assert "Add a supported model weight file." in empty_result["missing_required"]

    invalid = _detection_pack(tmp_path / "invalid-yaml")
    (invalid / "broken.yml").write_text("RTDETRTransformerv2: [\n", encoding="utf-8")
    invalid_result = manager.inspect(str(invalid))
    assert invalid_result["suggested_detector_backend"] is None
    assert any(
        "Could not parse YAML file broken.yml" in warning
        for warning in invalid_result["warnings"]
    )


def test_inspect_reuses_source_path_and_reparse_protections(tmp_path: Path):
    manager = _manager(tmp_path)
    source = _detection_pack(tmp_path / "source")
    user_data_root = manager.models_dir.parent
    with pytest.raises(CustomModelError, match="outside AddaxAI's managed model folders"):
        manager.inspect(str(user_data_root))
    with pytest.raises(CustomModelError, match="parent-directory traversal"):
        manager.inspect(str(source / ".." / "source"))

    linked = tmp_path / "linked-source"
    try:
        linked.symlink_to(source, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"Symlink creation is unavailable: {exc}")
    with pytest.raises(CustomModelError, match="Symbolic links and junctions"):
        manager.inspect(str(linked))


def test_inspect_missing_folder_returns_actionable_message(tmp_path: Path):
    manager = _manager(tmp_path)
    missing = tmp_path / "missing-model-pack"

    with pytest.raises(
        CustomModelError,
        match=re.escape("Model folder not found on this PC. Check the full path and try again."),
    ) as exc_info:
        manager.inspect(str(missing))

    assert "WinError" not in str(exc_info.value)


@pytest.mark.parametrize(
    ("failure", "expected"),
    [
        (
            PermissionError,
            "Cannot access the selected model folder. Check its permissions and try again.",
        ),
        (
            OSError,
            "Selected model folder is unavailable. Check the path or permissions and try again.",
        ),
    ],
)
def test_inspect_os_errors_return_concise_messages(
    tmp_path: Path, monkeypatch, failure: type[OSError], expected: str
):
    manager = _manager(tmp_path)
    source = tmp_path / "unavailable-model-pack"
    source.mkdir()
    original_resolve = Path.resolve

    def fail_selected_path(path: Path, strict: bool = False, **kwargs):
        if path == source:
            raise failure("private OS error details")
        return original_resolve(path, strict=strict, **kwargs)

    monkeypatch.setattr(Path, "resolve", fail_selected_path)

    with pytest.raises(CustomModelError, match=re.escape(expected)) as exc_info:
        manager.inspect(str(source))

    assert "private OS error details" not in str(exc_info.value)


@pytest.mark.asyncio
async def test_custom_model_inspect_api_is_read_only_and_loopback_only(
    local_model_api, tmp_path: Path
):
    app, models_dir = local_model_api
    source = _mdv6_style_rtdetrv2_pack(tmp_path / "api-mdv6-custom")
    transport = httpx.ASGITransport(app=app, client=("127.0.0.1", 54128))
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post(
            "/api/ml/custom-models/inspect",
            json={"source_path": str(source)},
            headers={"Origin": "http://127.0.0.1:5173"},
        )
        assert response.status_code == 200, response.text
        assert response.json()["suggested_detector_backend"] == "rtdetrv2"
        assert response.json()["suggested_env"] == "rtdetr"
        assert response.json()["suggested_class_names"]["19"] == "vehicle"
        assert list((models_dir / "det").iterdir()) == []
        assert list((models_dir / "cls").iterdir()) == []

        missing_response = await client.post(
            "/api/ml/custom-models/inspect",
            json={"source_path": str(tmp_path / "missing-model-pack")},
            headers={"Origin": "http://127.0.0.1:5173"},
        )
        assert missing_response.status_code == 400
        assert missing_response.json()["detail"] == (
            "Model folder not found on this PC. Check the full path and try again."
        )

        conflict_source = _mdv6_style_rtdetrv2_pack(tmp_path / "api-backend-conflict")
        (conflict_source / "manifest.json").write_text(
            json.dumps({"detector_backend": "yolo", "env": "pytorch"}),
            encoding="utf-8",
        )
        conflict_response = await client.post(
            "/api/ml/custom-models/inspect",
            json={"source_path": str(conflict_source)},
            headers={"Origin": "http://127.0.0.1:5173"},
        )
        assert conflict_response.status_code == 200, conflict_response.text
        assert conflict_response.json()["detector_backend_candidates"] == ["rtdetrv2", "yolo"]
        assert conflict_response.json()["suggested_detector_backend"] is None
        assert conflict_response.json()["suggested_env"] is None

        ancestor = await client.post(
            "/api/ml/custom-models/inspect",
            json={"source_path": str(models_dir.parent)},
        )
        assert ancestor.status_code == 400
        assert list((models_dir / "det").iterdir()) == []

    remote_transport = httpx.ASGITransport(app=app, client=("192.168.1.22", 54129))
    async with httpx.AsyncClient(
        transport=remote_transport, base_url="http://testserver"
    ) as remote_client:
        remote_response = await remote_client.post(
            "/api/ml/custom-models/inspect", json={"source_path": str(source)}
        )
    assert remote_response.status_code == 403


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

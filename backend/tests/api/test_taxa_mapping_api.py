"""A custom SpeciesNet taxa list on a project, through the API (AddaxAI#136)."""

import shutil
import uuid
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from app.ml.taxonomy_db import (
    batch_resolve_taxonomy_ids,
    populate_taxonomy_from_csv,
    resolve_taxonomy_id,
)
from app.models import Detection, EventObservation, File
from app.models.label_taxonomy import LabelTaxonomy
from tests.conftest import (
    make_deployment,
    make_detection,
    make_event_with_files,
    make_file,
    make_project,
)

TAXONOMY = """\
g3;mammalia;artiodactyla;bovidae;aepyceros;melampus;impala
g6;aves;accipitriformes;accipitridae;;;hawk family
g7;aves;strigiformes;strigidae;bubo;bubo;eurasian eagle-owl
"""

ROWS = [
    {"input_latin": "aepyceros", "user_taxon_name": "impala"},
    {"input_latin": "accipitridae", "user_taxon_name": "bird of prey"},
    {"input_latin": "strigiformes", "user_taxon_name": "bird of prey"},
]


@pytest.fixture(autouse=True)
def mock_manifest_manager():
    """Patch ManifestManager so model validation always succeeds."""
    mock_mgr = MagicMock()
    mock_mgr.get_model.return_value = MagicMock(model_id="MD5A-0-0")
    # The taxa-list route reads the cached managers, which an earlier test
    # may have filled with the real catalog.
    with (
        patch("app.ml.manifest_manager.ManifestManager", return_value=mock_mgr),
        patch(
            "app.api.routers.ml_models._get_managers",
            return_value=(mock_mgr, None, None),
        ),
    ):
        yield mock_mgr


@pytest.fixture
def model_id():
    """A classification model dir holding a SpeciesNet taxonomy file.
    A new id per test, as the taxonomy is cached by path."""
    from app.core.config import get_settings

    model_id = f"taxa-model-{uuid.uuid4().hex[:6]}"
    model_dir = get_settings().models_dir / "cls" / model_id
    model_dir.mkdir(parents=True)
    (model_dir / "taxonomy_release.1.txt").write_text(TAXONOMY, encoding="utf-8")
    return model_id


def _mapping(rows=ROWS):
    return {"file_name": "mara.csv", "rows": rows}


def test_reading_a_list_returns_its_rows(client, model_id):
    resp = client.post(
        f"/api/ml/models/{model_id}/taxa-list",
        json={
            "file_name": "mara.csv",
            "text": "input_latin,user_taxon_name\nAepyceros,Impala\n",
        },
    )
    assert resp.status_code == 200
    assert resp.json() == {
        "file_name": "mara.csv",
        "rows": [{"input_latin": "aepyceros", "user_taxon_name": "impala"}],
    }


def test_a_bad_list_is_refused_with_its_rows(client, model_id):
    resp = client.post(
        f"/api/ml/models/{model_id}/taxa-list",
        json={"file_name": "x.csv", "text": "input_latin,user_taxon_name\nlynx,lynx\n"},
    )
    assert resp.status_code == 422
    assert resp.json()["detail"]["problems"] == ["Row 2 (lynx): not a SpeciesNet taxon"]


def test_a_model_without_a_taxonomy_file_cannot_take_a_list(client):
    resp = client.post(
        "/api/ml/models/no-such-model/taxa-list",
        json={"file_name": "x.csv", "text": "input_latin,user_taxon_name\naepyceros,impala\n"},
    )
    assert resp.status_code == 422


def test_saving_a_list_clears_the_country_and_gives_each_label_its_row(
    client, db, model_id
):
    project = make_project(
        db,
        classification_model_id=model_id,
        country_code="KEN",
        excluded_classes=["impala"],
    )
    db.commit()

    resp = client.patch(f"/api/projects/{project.id}", json={"taxa_mapping": _mapping()})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["taxa_mapping"] == _mapping()
    assert body["country_code"] is None
    assert body["excluded_classes"] == []

    rows = {
        r.name: r
        for r in db.query(LabelTaxonomy).filter(LabelTaxonomy.project_id == project.id)
    }
    assert set(rows) == {"impala", "bird of prey"}
    assert (rows["impala"].level, rows["impala"].taxon_genus) == ("genus", "aepyceros")
    assert (rows["bird of prey"].level, rows["bird of prey"].taxon_class) == ("class", "aves")
    assert rows["bird of prey"].scientific_name == "Aves"


def test_a_country_for_a_project_with_a_list_is_refused(client, db, model_id):
    project = make_project(db, classification_model_id=model_id)
    db.commit()
    client.patch(f"/api/projects/{project.id}", json={"taxa_mapping": _mapping()})

    resp = client.patch(f"/api/projects/{project.id}", json={"country_code": "KEN"})
    assert resp.status_code == 400

    # Removing the list in the same save is the way to switch.
    resp = client.patch(
        f"/api/projects/{project.id}",
        json={"taxa_mapping": None, "country_code": "KEN"},
    )
    assert resp.status_code == 200
    assert resp.json()["taxa_mapping"] is None
    assert resp.json()["country_code"] == "KEN"


def test_a_tampered_list_is_refused_on_save(client, db, model_id):
    project = make_project(db, classification_model_id=model_id, country_code="KEN")
    db.commit()
    bad = [{"input_latin": "panthera leo", "user_taxon_name": "lion"}]

    resp = client.patch(f"/api/projects/{project.id}", json={"taxa_mapping": _mapping(bad)})
    assert resp.status_code == 422
    db.expire_all()
    assert client.get(f"/api/projects/{project.id}").json()["country_code"] == "KEN"


def test_create_with_a_list(client, model_id):
    resp = client.post(
        "/api/projects",
        json={
            "name": "with a list",
            "timezone": "UTC",
            "classification_model_id": model_id,
            "taxa_mapping": _mapping(),
        },
    )
    assert resp.status_code == 201, resp.text
    assert resp.json()["taxa_mapping"]["rows"] == ROWS


def test_the_projects_own_row_wins_a_name_clash(client, db, model_id):
    """A custom "impala" on the genus, next to SpeciesNet's own species
    row "impala": this project reads its own, another project the model's."""
    mapped = make_project(db, classification_model_id=model_id)
    plain = make_project(db, classification_model_id=model_id)
    db.commit()
    client.patch(f"/api/projects/{mapped.id}", json={"taxa_mapping": _mapping()})
    # Inserted after the project's row, so the win cannot come from
    # insertion order.
    model_row = LabelTaxonomy(
        classification_model_id=model_id,
        name="impala",
        level="species",
        taxon_genus="aepyceros",
        taxon_species="melampus",
    )
    db.add(model_row)
    db.commit()

    own = (
        db.query(LabelTaxonomy)
        .filter(LabelTaxonomy.project_id == mapped.id, LabelTaxonomy.name == "impala")
        .one()
    )
    def batch(project_id):
        return batch_resolve_taxonomy_ids(["impala"], model_id, project_id, db)["impala"][0]

    assert batch(mapped.id) == own.id
    assert resolve_taxonomy_id("impala", mapped.id, db) == own.id
    assert batch(plain.id) == model_row.id
    assert resolve_taxonomy_id("impala", plain.id, db) == model_row.id
    # Another project's list label is not offered to this one.
    assert "bird of prey" not in batch_resolve_taxonomy_ids(
        ["bird of prey"], model_id, plain.id, db
    )

    taxonomy_map = client.get(f"/api/projects/{mapped.id}/label-taxonomy-map").json()
    assert taxonomy_map["impala"]["taxon_species"] is None
    assert "bird of prey" not in client.get(
        f"/api/projects/{plain.id}/label-taxonomy-map"
    ).json()


def _model_row(db, model_id, name="impala"):
    row = LabelTaxonomy(
        classification_model_id=model_id,
        name=name,
        level="species",
        taxon_genus="aepyceros",
        taxon_species="melampus",
    )
    db.add(row)
    db.flush()
    return row


def _detection(db, project, **kw):
    deployment = make_deployment(db, project_id=project.id)
    file = make_file(db, deployment_id=deployment.id)
    return make_detection(db, file_id=file.id, label="impala", **kw)


def _own_row(db, project, name="impala"):
    return (
        db.query(LabelTaxonomy)
        .filter(LabelTaxonomy.project_id == project.id, LabelTaxonomy.name == name)
        .one_or_none()
    )


def test_an_unknown_model_id_is_refused_before_any_path_is_built(client):
    manifests = MagicMock()
    manifests.get_model.side_effect = ValueError("Unknown model")
    with patch(
        "app.api.routers.ml_models._get_managers", return_value=(manifests, None, None)
    ):
        resp = client.post(
            "/api/ml/models/..%5C..%5Cx/taxa-list",
            json={
                "file_name": "x.csv",
                "text": "input_latin,user_taxon_name\naepyceros,impala\n",
            },
        )
    assert resp.status_code == 404


def test_the_models_own_rows_still_go_in_next_to_a_list_row(db, model_id, tmp_path):
    """A list project saved before the model's first run must not keep
    the model's own "impala" out of the table for every other project."""
    project = make_project(db, classification_model_id=model_id)
    db.add(
        LabelTaxonomy(
            classification_model_id=model_id, name="impala", level="genus", project_id=project.id
        )
    )
    db.flush()
    csv_path = tmp_path / "taxonomy.csv"
    csv_path.write_text(
        "model_class,class,order,family,genus,species\n"
        "impala,mammalia,artiodactyla,bovidae,aepyceros,melampus\n",
        encoding="utf-8",
    )
    assert populate_taxonomy_from_csv(model_id, csv_path, db) == 1


def test_a_list_relinks_its_labels_and_removing_it_gives_them_back(client, db, model_id):
    project = make_project(db, classification_model_id=model_id)
    model_row = _model_row(db, model_id)
    detection = _detection(db, project, label_taxonomy_id=model_row.id)
    event = make_event_with_files(
        db,
        deployment_id=db.get(File, detection.file_id).deployment_id,
        event_start_local=datetime(2024, 1, 1, 12),
    )
    observation = EventObservation(
        event_id=event.id,
        label="impala",
        label_taxonomy_id=model_row.id,
        category="animal",
        max_n=1,
    )
    db.add(observation)
    db.commit()

    client.patch(f"/api/projects/{project.id}", json={"taxa_mapping": _mapping()})
    db.expire_all()
    own = _own_row(db, project)
    assert db.get(Detection, detection.id).label_taxonomy_id == own.id
    assert db.get(EventObservation, observation.id).label_taxonomy_id == own.id

    resp = client.patch(f"/api/projects/{project.id}", json={"taxa_mapping": None})
    assert resp.status_code == 200, resp.text
    db.expire_all()
    assert _own_row(db, project) is None
    assert _own_row(db, project, "bird of prey") is None
    assert db.get(Detection, detection.id).label_taxonomy_id == model_row.id
    assert db.get(EventObservation, observation.id).label_taxonomy_id == model_row.id


def test_resending_a_list_with_a_country_still_drops_the_country(client, db, model_id):
    """The settings form resends the unchanged list on every save."""
    project = make_project(db, classification_model_id=model_id)
    db.commit()
    client.patch(f"/api/projects/{project.id}", json={"taxa_mapping": _mapping()})

    resp = client.patch(
        f"/api/projects/{project.id}",
        json={"taxa_mapping": _mapping(), "country_code": "KEN", "excluded_classes": ["x"]},
    )
    assert resp.status_code == 200, resp.text
    assert (resp.json()["country_code"], resp.json()["excluded_classes"]) == (None, [])


def test_a_settings_save_works_after_the_model_folder_is_gone(client, db, model_id):
    """The settings form sends the list and the model on every save."""
    from app.core.config import get_settings

    project = make_project(db, classification_model_id=model_id)
    db.commit()
    client.patch(f"/api/projects/{project.id}", json={"taxa_mapping": _mapping()})
    shutil.rmtree(get_settings().models_dir / "cls" / model_id)

    resp = client.patch(
        f"/api/projects/{project.id}",
        json={
            "name": "renamed",
            "classification_model_id": model_id,
            "taxa_mapping": _mapping(),
        },
    )
    assert resp.status_code == 200, resp.text


def test_a_custom_label_may_take_another_projects_list_label(client, db, model_id):
    mapped = make_project(db, classification_model_id=model_id)
    plain = make_project(db, classification_model_id=model_id)
    db.commit()
    client.patch(f"/api/projects/{mapped.id}", json={"taxa_mapping": _mapping()})
    label = client.post(f"/api/projects/{plain.id}/custom-labels", json={"name": "raptor"})
    assert label.status_code == 201, label.text

    resp = client.patch(
        f"/api/projects/{plain.id}/custom-labels/{label.json()['id']}",
        json={"name": "bird of prey"},
    )
    assert resp.status_code == 200, resp.text


def test_deleting_a_project_deletes_its_list_rows(client, db, model_id):
    project = make_project(db, classification_model_id=model_id)
    db.commit()
    client.patch(f"/api/projects/{project.id}", json={"taxa_mapping": _mapping()})

    assert client.delete(f"/api/projects/{project.id}").status_code == 204
    db.expire_all()
    assert db.query(LabelTaxonomy).filter(LabelTaxonomy.project_id == project.id).count() == 0

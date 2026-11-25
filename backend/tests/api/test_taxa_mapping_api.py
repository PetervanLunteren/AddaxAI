"""A custom SpeciesNet taxa list on a project, through the API (AddaxAI#136)."""

import shutil
import uuid
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import text

from app.api.schemas.project import TaxaRow
from app.ml.taxa_mapping import LABEL, LATIN, RETIRED
from app.ml.taxonomy_db import (
    batch_resolve_taxonomy_ids,
    populate_taxonomy_from_csv,
    resolve_taxonomy_id,
)
from app.models import Detection, Event, EventObservation, File
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
    {"latin": "aepyceros", "label": "impala"},
    {"latin": "accipitridae", "label": "bird of prey"},
    {"latin": "strigiformes", "label": "bird of prey"},
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
    return {"file_name": "mara.csv", "text": "the file as read", "rows": rows}


def test_reading_a_list_returns_its_rows_and_the_file_as_read(client, model_id):
    text = "input_latin;output_name;notes\nAepyceros;Impala;check the reedbuck\n"
    resp = client.post(
        f"/api/ml/models/{model_id}/taxa-list",
        json={"file_name": "mara.csv", "text": text},
    )
    assert resp.status_code == 200
    assert resp.json() == {
        "file_name": "mara.csv",
        "text": text,
        "rows": [{"latin": "aepyceros", "label": "impala"}],
    }


def test_the_file_as_read_is_kept_with_the_project(client, db, model_id):
    """Download hands the user's own file back, notes included, also
    months later: the text is saved and read back unchanged."""
    project = make_project(db, classification_model_id=model_id)
    db.commit()
    mapping = {**_mapping(), "text": "input_latin,output_name,notes\naepyceros,impala,a note\n"}

    client.patch(f"/api/projects/{project.id}", json={"taxa_mapping": mapping})
    stored = client.get(f"/api/projects/{project.id}").json()["taxa_mapping"]
    assert stored["text"] == mapping["text"]
    assert stored["file_name"] == "mara.csv"


def test_a_bad_list_is_refused_with_its_rows(client, model_id):
    resp = client.post(
        f"/api/ml/models/{model_id}/taxa-list",
        json={"file_name": "x.csv", "text": "input_latin,output_name\nlynx,lynx\n"},
    )
    assert resp.status_code == 422
    assert resp.json()["detail"]["problems"] == ["Row 2 (lynx): not a SpeciesNet taxon"]


def test_a_model_without_a_taxonomy_file_cannot_take_a_list(client):
    resp = client.post(
        "/api/ml/models/no-such-model/taxa-list",
        json={"file_name": "x.csv", "text": "input_latin,output_name\naepyceros,impala\n"},
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


def test_a_country_for_a_project_with_a_list_is_not_kept(client, db, model_id):
    project = make_project(db, classification_model_id=model_id)
    db.commit()
    client.patch(f"/api/projects/{project.id}", json={"taxa_mapping": _mapping()})

    resp = client.patch(
        f"/api/projects/{project.id}",
        json={"country_code": "KEN", "excluded_classes": ["lion"]},
    )
    assert resp.status_code == 200
    assert resp.json()["taxa_mapping"] is not None
    assert (resp.json()["country_code"], resp.json()["excluded_classes"]) == (None, [])

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
    bad = [{"latin": "panthera leo", "label": "lion"}]

    resp = client.patch(f"/api/projects/{project.id}", json={"taxa_mapping": _mapping(bad)})
    assert resp.status_code == 422
    db.expire_all()
    assert client.get(f"/api/projects/{project.id}").json()["country_code"] == "KEN"


def test_a_row_with_other_keys_is_refused_at_the_door(client, db, model_id):
    project = make_project(db, classification_model_id=model_id, country_code="KEN")
    db.commit()
    other_keys = [{"taxon": "aepyceros", "name": "impala"}]

    resp = client.patch(
        f"/api/projects/{project.id}", json={"taxa_mapping": _mapping(other_keys)}
    )
    # FastAPI's own validation error, before any taxa list code runs.
    assert resp.status_code == 422
    assert any(error["loc"][-1] == LATIN for error in resp.json()["detail"])
    db.expire_all()
    assert client.get(f"/api/projects/{project.id}").json()["country_code"] == "KEN"


def test_the_schema_row_has_the_stored_keys():
    """The API row and the stored row cannot drift apart."""
    assert set(TaxaRow.model_fields) == {LATIN, LABEL}


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
                "text": "input_latin,output_name\naepyceros,impala\n",
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
    # SQL NULL, not the JSON text null, as the model says "no list" is.
    stored = db.execute(
        text("SELECT taxa_mapping IS NULL FROM projects WHERE id = :id"), {"id": project.id}
    ).scalar()
    assert stored == 1


def test_what_a_person_checked_keeps_its_row_through_list_changes(client, db, model_id):
    """AddaxAI's rule for every reprocess: a verified detection or a
    confirmed observation stays as it is, taxonomy included."""

    project = make_project(db, classification_model_id=model_id)
    model_row = _model_row(db, model_id)
    checked_before = _detection(db, project, label_taxonomy_id=model_row.id, verified=True)
    db.commit()

    # A list that names its label does not move it.
    client.patch(f"/api/projects/{project.id}", json={"taxa_mapping": _mapping()})
    db.expire_all()
    own = _own_row(db, project)
    assert db.get(Detection, checked_before.id).label_taxonomy_id == model_row.id

    verified = _detection(db, project, label_taxonomy_id=own.id, verified=True)
    track_box = _detection(
        db, project, label_taxonomy_id=own.id, classification_method="track"
    )
    unverified = _detection(db, project, label_taxonomy_id=own.id)
    deployment_id = db.get(File, verified.file_id).deployment_id
    signed_event = make_event_with_files(
        db, deployment_id=deployment_id, event_start_local=datetime(2024, 1, 1, 12)
    )
    signed_event.confirmed = True
    open_event = make_event_with_files(
        db, deployment_id=deployment_id, event_start_local=datetime(2024, 1, 2, 12)
    )

    def observation(event, **kw):
        obs = EventObservation(
            event_id=event.id, label="impala", label_taxonomy_id=own.id,
            category="animal", max_n=1, **kw,
        )
        db.add(obs)
        return obs

    signed = observation(signed_event)
    counted = observation(open_event, human_count=3)
    open_obs = observation(open_event)
    db.commit()
    checked = [
        (Detection, verified.id),
        (Detection, track_box.id),
        (EventObservation, signed.id),
        (EventObservation, counted.id),
    ]

    resp = client.patch(f"/api/projects/{project.id}", json={"taxa_mapping": None})
    assert resp.status_code == 200, resp.text
    db.expire_all()
    # The row stays for what a person checked, out of the name lookups, so
    # the rest gets the model's impala back.
    retired = _own_row(db, project)
    assert retired.id == own.id
    assert retired.classification_model_id == RETIRED + own.id
    assert [db.get(m, i).label_taxonomy_id for m, i in checked] == [own.id] * 4
    assert db.get(Detection, unverified.id).label_taxonomy_id == model_row.id
    assert db.get(EventObservation, open_obs.id).label_taxonomy_id == model_row.id
    assert resolve_taxonomy_id("impala", project.id, db) == model_row.id
    assert _own_row(db, project, "bird of prey") is None

    # A list that names it again takes the row back.
    client.patch(f"/api/projects/{project.id}", json={"taxa_mapping": _mapping()})
    db.expire_all()
    assert _own_row(db, project).classification_model_id == model_id
    assert db.get(Detection, unverified.id).label_taxonomy_id == own.id

    # Once nothing checked uses it, a list without the label drops it.
    db.get(Detection, verified.id).verified = False
    db.get(Detection, track_box.id).classification_method = None
    db.get(Event, signed_event.id).confirmed = False
    db.get(EventObservation, counted.id).human_count = None
    db.commit()
    client.patch(f"/api/projects/{project.id}", json={"taxa_mapping": _mapping(ROWS[1:])})
    db.expire_all()
    assert _own_row(db, project) is None


def test_a_retired_row_nothing_uses_any_more_is_taken_back(client, db, model_id):
    """The verified box went (relabelled or deleted) while the row was
    retired: naming the label again revives the row, not a 500."""
    project = make_project(db, classification_model_id=model_id)
    db.commit()
    client.patch(f"/api/projects/{project.id}", json={"taxa_mapping": _mapping()})
    db.expire_all()
    own = _own_row(db, project)
    verified = _detection(db, project, label_taxonomy_id=own.id, verified=True)
    db.commit()
    client.patch(f"/api/projects/{project.id}", json={"taxa_mapping": None})
    db.delete(db.get(Detection, verified.id))
    db.commit()

    resp = client.patch(f"/api/projects/{project.id}", json={"taxa_mapping": _mapping()})
    assert resp.status_code == 200, resp.text
    db.expire_all()
    assert (_own_row(db, project).id, _own_row(db, project).classification_model_id) == (
        own.id,
        model_id,
    )


def test_a_new_model_keeps_its_own_row_next_to_a_retired_one(client, db, model_id):
    """After a model switch the old model's row of a label is retired for
    a verified detection and the new model has its own. A later save of
    the list updates the new model's row and leaves the retired one."""
    from app.core.config import get_settings

    other_model = f"{model_id}-b"
    other_dir = get_settings().models_dir / "cls" / other_model
    other_dir.mkdir(parents=True)
    (other_dir / "taxonomy_release.1.txt").write_text(TAXONOMY, encoding="utf-8")

    project = make_project(db, classification_model_id=model_id)
    db.commit()
    client.patch(f"/api/projects/{project.id}", json={"taxa_mapping": _mapping()})
    db.expire_all()
    old_row = _own_row(db, project)
    _detection(db, project, label_taxonomy_id=old_row.id, verified=True)
    db.commit()

    client.patch(f"/api/projects/{project.id}", json={"classification_model_id": other_model})
    resp = client.patch(f"/api/projects/{project.id}", json={"taxa_mapping": _mapping(ROWS[:2])})
    assert resp.status_code == 200, resp.text
    db.expire_all()
    rows = {
        r.classification_model_id: r.id
        for r in db.query(LabelTaxonomy).filter(
            LabelTaxonomy.project_id == project.id, LabelTaxonomy.name == "impala"
        )
    }
    assert rows[RETIRED + old_row.id] == old_row.id
    assert rows[other_model] != old_row.id


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

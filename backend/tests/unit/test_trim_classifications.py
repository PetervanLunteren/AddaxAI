"""Tests for trim_classification_results in json_utils."""

from app.ml.json_utils import trim_classification_results


def _make_results(
    classifications_per_det: list[list],
    categories: dict | None = None,
    descriptions: dict | None = None,
) -> dict:
    """Build a minimal results dict for testing."""
    detections = []
    for cls_list in classifications_per_det:
        det = {"category": "1", "conf": 0.9, "bbox": [0.1, 0.2, 0.3, 0.4]}
        if cls_list is not None:
            det["classifications"] = cls_list
        detections.append(det)

    result: dict = {
        "images": [{"file": "img.jpg", "detections": detections}],
    }
    if categories is not None:
        result["classification_categories"] = categories
    if descriptions is not None:
        result["classification_category_descriptions"] = descriptions
    return result


def test_trim_basic():
    cats = {str(i): f"species_{i}" for i in range(10)}
    descs = {str(i): f"desc_{i}" for i in range(10)}
    cls = [[str(i), 0.9 - i * 0.1] for i in range(10)]

    results = _make_results([cls], categories=cats, descriptions=descs)
    removed = trim_classification_results(results)

    det = results["images"][0]["detections"][0]
    assert len(det["classifications"]) == 5
    assert det["classifications"][0] == ["0", 0.9]
    assert det["classifications"][4] == ["4", 0.5]
    assert removed == 5
    assert len(results["classification_categories"]) == 5
    assert len(results["classification_category_descriptions"]) == 5
    assert "5" not in results["classification_categories"]
    assert "5" not in results["classification_category_descriptions"]


def test_trim_fewer_than_five():
    cats = {"1": "lion", "2": "zebra", "3": "deer"}
    cls = [["1", 0.8], ["2", 0.15], ["3", 0.05]]

    results = _make_results([cls], categories=cats)
    removed = trim_classification_results(results)

    det = results["images"][0]["detections"][0]
    assert len(det["classifications"]) == 3
    assert removed == 0


def test_trim_no_classifications_field():
    cats = {"1": "lion"}
    results = _make_results([None], categories=cats)
    removed = trim_classification_results(results)

    # Category "1" is unreferenced (no detection has classifications), so it is pruned
    assert removed == 1
    det = results["images"][0]["detections"][0]
    assert "classifications" not in det


def test_trim_empty_classifications():
    cats = {"1": "lion"}
    results = _make_results([[]], categories=cats)
    removed = trim_classification_results(results)

    assert removed == 1
    det = results["images"][0]["detections"][0]
    assert det["classifications"] == []


def test_trim_prunes_descriptions():
    cats = {str(i): f"sp_{i}" for i in range(8)}
    descs = {str(i): f"sp_{i};cls;ord;fam;gen;spe;sp_{i}" for i in range(8)}
    cls = [[str(i), 0.9 - i * 0.1] for i in range(8)]

    results = _make_results([cls], categories=cats, descriptions=descs)
    trim_classification_results(results)

    assert set(results["classification_category_descriptions"].keys()) == {
        "0", "1", "2", "3", "4",
    }


def test_trim_no_descriptions_key():
    cats = {str(i): f"sp_{i}" for i in range(8)}
    cls = [[str(i), 0.9 - i * 0.1] for i in range(8)]

    results = _make_results([cls], categories=cats)
    removed = trim_classification_results(results)

    assert removed == 3
    assert "classification_category_descriptions" not in results


def test_trim_no_classification_categories():
    cls = [["1", 0.8], ["2", 0.15]]
    results = _make_results([cls])
    removed = trim_classification_results(results)

    assert removed == 0


def test_trim_mixed_detections():
    """Animal detections have classifications; person/vehicle do not."""
    cats = {str(i): f"sp_{i}" for i in range(8)}

    animal_cls = [[str(i), 0.9 - i * 0.1] for i in range(8)]
    person_det = {"category": "2", "conf": 0.95, "bbox": [0, 0, 1, 1]}
    vehicle_det = {"category": "3", "conf": 0.85, "bbox": [0, 0, 1, 1]}

    results = {
        "images": [{
            "file": "img.jpg",
            "detections": [
                {
                    "category": "1", "conf": 0.9,
                    "bbox": [0.1, 0.2, 0.3, 0.4],
                    "classifications": animal_cls,
                },
                person_det,
                vehicle_det,
            ],
        }],
        "classification_categories": cats,
    }

    removed = trim_classification_results(results)

    assert removed == 3
    assert len(results["images"][0]["detections"][0]["classifications"]) == 5
    assert "classifications" not in results["images"][0]["detections"][1]
    assert "classifications" not in results["images"][0]["detections"][2]


def test_trim_custom_max():
    cats = {str(i): f"sp_{i}" for i in range(10)}
    cls = [[str(i), 0.9 - i * 0.05] for i in range(10)]

    results = _make_results([cls], categories=cats)
    removed = trim_classification_results(results, max_classifications=3)

    det = results["images"][0]["detections"][0]
    assert len(det["classifications"]) == 3
    assert removed == 7


def test_trim_returns_removed_count():
    cats = {str(i): f"sp_{i}" for i in range(20)}
    cls = [[str(i), 0.9 - i * 0.01] for i in range(20)]

    results = _make_results([cls], categories=cats)
    removed = trim_classification_results(results)

    assert removed == 15
    assert len(results["classification_categories"]) == 5


def test_trim_multiple_detections_share_categories():
    """Two detections reference different top-5, union is kept."""
    cats = {str(i): f"sp_{i}" for i in range(10)}
    cls_a = [[str(i), 0.9 - i * 0.1] for i in range(10)]
    cls_b = [[str(9 - i), 0.9 - i * 0.1] for i in range(10)]

    results = _make_results([cls_a, cls_b], categories=cats)
    trim_classification_results(results)

    # cls_a keeps 0-4, cls_b keeps 9-5
    assert len(results["classification_categories"]) == 10

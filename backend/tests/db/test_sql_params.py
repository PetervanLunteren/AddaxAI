"""Unit tests for the SQL bound-parameter chunking helper."""

from __future__ import annotations

from app.db.sql_params import SQL_VAR_CHUNK, iter_id_chunks


def test_empty_yields_nothing():
    assert list(iter_id_chunks([])) == []


def test_shorter_than_chunk_is_single_chunk():
    ids = ["a", "b", "c"]
    assert list(iter_id_chunks(ids)) == [ids]


def test_exactly_chunk_size_is_single_chunk():
    ids = [str(i) for i in range(SQL_VAR_CHUNK)]
    chunks = list(iter_id_chunks(ids))
    assert len(chunks) == 1
    assert chunks[0] == ids


def test_one_over_chunk_size_splits_into_two():
    ids = [str(i) for i in range(SQL_VAR_CHUNK + 1)]
    chunks = list(iter_id_chunks(ids))
    assert len(chunks) == 2
    assert len(chunks[0]) == SQL_VAR_CHUNK
    assert len(chunks[1]) == 1


def test_preserves_order_and_all_ids():
    ids = [str(i) for i in range(SQL_VAR_CHUNK * 2 + 7)]
    flattened = [x for chunk in iter_id_chunks(ids) for x in chunk]
    assert flattened == ids


def test_custom_size():
    ids = [str(i) for i in range(10)]
    chunks = list(iter_id_chunks(ids, size=4))
    assert chunks == [
        ["0", "1", "2", "3"],
        ["4", "5", "6", "7"],
        ["8", "9"],
    ]


def test_accepts_any_iterable():
    # Generators and dict-keys views must work, not just lists.
    chunks = list(iter_id_chunks((str(i) for i in range(5)), size=2))
    assert chunks == [["0", "1"], ["2", "3"], ["4"]]

    mapping = {str(i): i for i in range(3)}
    assert list(iter_id_chunks(mapping.keys(), size=2)) == [["0", "1"], ["2"]]

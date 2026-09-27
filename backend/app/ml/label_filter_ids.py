"""Comma-safe IDs for label leaves that have no taxonomy row.

Taxonomy leaves keep their existing UUIDs. Unmapped labels use an encoded
token so names containing commas can safely travel in the existing
comma-separated filter query parameters.
"""

from __future__ import annotations

import base64
import binascii
import re

UNMAPPED_LABEL_PREFIX = "unmapped:"
_TOKEN_BODY = re.compile(r"^[A-Za-z0-9_-]+$")


def encode_unmapped_label(label: str) -> str:
    """Encode an unmapped label as a stable, comma-safe filter ID."""
    if not label:
        raise ValueError("Unmapped labels must not be empty")
    payload = base64.urlsafe_b64encode(label.encode("utf-8")).decode("ascii")
    return f"{UNMAPPED_LABEL_PREFIX}{payload.rstrip('=')}"


def decode_unmapped_label(token: str) -> str | None:
    """Decode a canonical unmapped-label ID, or return None for other IDs."""
    if not token.startswith(UNMAPPED_LABEL_PREFIX):
        return None
    body = token[len(UNMAPPED_LABEL_PREFIX) :]
    if not body or not _TOKEN_BODY.fullmatch(body):
        return None
    try:
        padded = body + "=" * (-len(body) % 4)
        label = base64.urlsafe_b64decode(padded.encode("ascii")).decode("utf-8")
    except (binascii.Error, UnicodeDecodeError, ValueError):
        return None
    if not label or encode_unmapped_label(label) != token:
        return None
    return label


def parse_label_filter_ids(
    label_ids: list[str] | tuple[str, ...] | set[str] | frozenset[str],
) -> tuple[list[str], list[str], list[str]]:
    """Return taxonomy IDs, legacy raw names, and encoded unmapped names.

    Plain values are kept in both the taxonomy and raw-name groups. This
    preserves the older folder-run exclusion contract, which accepted raw
    label strings, while new unmapped tokens match only null-taxonomy rows.
    """
    taxonomy_ids: list[str] = []
    legacy_raw_labels: list[str] = []
    unmapped_labels: list[str] = []
    for token in label_ids:
        label = decode_unmapped_label(token)
        if label is not None:
            unmapped_labels.append(label)
            continue
        if token:
            taxonomy_ids.append(token)
            legacy_raw_labels.append(token)
    return (
        list(dict.fromkeys(taxonomy_ids)),
        list(dict.fromkeys(legacy_raw_labels)),
        list(dict.fromkeys(unmapped_labels)),
    )


def label_matches_filter(
    label: str | None,
    label_taxonomy_id: str | None,
    label_ids: list[str] | tuple[str, ...] | set[str] | frozenset[str],
    category: str | None = None,
) -> bool:
    """Check a label/taxonomy pair against a mixed legacy and encoded set."""
    if not label_ids:
        return False
    taxonomy_ids, legacy_raw_labels, unmapped_labels = parse_label_filter_ids(
        label_ids
    )
    if label_taxonomy_id and label_taxonomy_id in taxonomy_ids:
        return True
    effective_label = label or category
    if effective_label and effective_label in legacy_raw_labels:
        return True
    return (
        label_taxonomy_id is None
        and effective_label is not None
        and effective_label in unmapped_labels
    )


def label_filter_expression(
    taxonomy_column, label_column, label_ids: list[str], category_column=None
):
    """Build a SQLAlchemy expression matching taxonomy and raw-label tokens."""
    from sqlalchemy import and_, false, or_

    taxonomy_ids, legacy_raw_labels, unmapped_labels = parse_label_filter_ids(
        label_ids
    )
    expressions = []
    if taxonomy_ids:
        expressions.append(taxonomy_column.in_(taxonomy_ids))
    if legacy_raw_labels or unmapped_labels:
        if category_column is not None:
            from sqlalchemy import func

            effective_label = func.coalesce(
                func.nullif(label_column, ""), category_column
            )
        else:
            effective_label = label_column
        if legacy_raw_labels:
            expressions.append(effective_label.in_(legacy_raw_labels))
    if unmapped_labels:
        if category_column is not None:
            effective_label = func.coalesce(
                func.nullif(label_column, ""), category_column
            )
        else:
            effective_label = label_column
        expressions.append(
            and_(taxonomy_column.is_(None), effective_label.in_(unmapped_labels))
        )
    return or_(*expressions) if expressions else false()

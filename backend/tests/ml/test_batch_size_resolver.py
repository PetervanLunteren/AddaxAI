"""Tests for the batch size display constants."""

from app.ml.batch_size import (
    CLASSIFICATION_DEFAULT_CPU,
    CLASSIFICATION_DEFAULT_GPU,
    DETECTION_DEFAULT_CPU,
    DETECTION_DEFAULT_GPU,
    EMBEDDING_DEFAULT_CPU,
    EMBEDDING_DEFAULT_GPU,
)


class TestDefaultsAreSane:
    """The hardcoded display constants must be positive and GPU >= CPU."""

    def test_detection_defaults(self):
        assert DETECTION_DEFAULT_GPU >= DETECTION_DEFAULT_CPU >= 1

    def test_classification_defaults(self):
        assert CLASSIFICATION_DEFAULT_GPU >= CLASSIFICATION_DEFAULT_CPU >= 1

    def test_embedding_defaults(self):
        assert EMBEDDING_DEFAULT_GPU >= EMBEDDING_DEFAULT_CPU >= 1

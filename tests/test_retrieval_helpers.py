"""Tests for minimal retrieval helper behavior."""

from labelrag.retrieval.coverage import uncovered_overlap_size


def test_uncovered_overlap_size_counts_shared_labels() -> None:
    """Coverage helper should count only uncovered shared labels."""

    assert uncovered_overlap_size({"a", "b"}, {"b", "c"}) == 1

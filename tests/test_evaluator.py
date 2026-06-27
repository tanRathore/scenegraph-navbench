import pytest

from scenegraph_navbench.evaluator import evaluate_relations, load_expected_relations


def test_perfect_match_metrics() -> None:
    generated = [
        ("chair_1", "left_of", "door_1"),
        ("chair_1", "near", "table_1"),
    ]
    expected = [
        ("chair_1", "left_of", "door_1"),
        ("chair_1", "near", "table_1"),
    ]

    metrics = evaluate_relations(generated, expected)

    assert metrics.mode == "exact"
    assert metrics.precision == 1.0
    assert metrics.recall == 1.0
    assert metrics.f1 == 1.0
    assert metrics.required_relation_recall == 1.0


def test_partial_match_metrics() -> None:
    generated = [
        ("chair_1", "left_of", "door_1"),
        ("table_1", "right_of", "chair_1"),
    ]
    expected = [
        ("chair_1", "left_of", "door_1"),
        ("chair_1", "near", "table_1"),
        ("box_1", "below", "wall_art_1"),
    ]

    metrics = evaluate_relations(generated, expected)

    assert metrics.precision == 0.5
    assert metrics.recall == 1 / 3
    assert round(metrics.f1, 6) == round(0.4, 6)


def test_empty_prediction_case() -> None:
    metrics = evaluate_relations(
        [],
        [("chair_1", "left_of", "door_1")],
    )

    assert metrics.precision == 0.0
    assert metrics.recall == 0.0
    assert metrics.f1 == 0.0


def test_must_include_mode_reports_required_recall_without_precision() -> None:
    generated = [
        ("chair_1", "left_of", "door_1"),
        ("table_1", "right_of", "chair_1"),
    ]
    expected = [
        ("chair_1", "left_of", "door_1"),
    ]

    metrics = evaluate_relations(generated, expected, mode="must_include")

    assert metrics.mode == "must_include"
    assert metrics.required_relation_recall == 1.0
    assert metrics.precision is None
    assert metrics.recall is None
    assert metrics.f1 is None
    assert metrics.true_positives == 1
    assert metrics.predicted == 2
    assert metrics.expected == 1


def test_load_expected_relations_reads_mode(tmp_path) -> None:
    expected_path = tmp_path / "expected.json"
    expected_path.write_text(
        """
        {
          "mode": "must_include",
          "relations": [
            {"source": "chair_1", "relation": "left_of", "target": "door_1"}
          ]
        }
        """,
        encoding="utf-8",
    )

    expected = load_expected_relations(expected_path)

    assert expected.mode == "must_include"
    assert expected.relations == [
        {"source": "chair_1", "relation": "left_of", "target": "door_1"}
    ]


def test_rejects_malformed_relation_tuple() -> None:
    with pytest.raises(ValueError, match="source, relation, and target"):
        evaluate_relations(
            [("chair_1", "left_of")],  # type: ignore[list-item]
            [],
        )

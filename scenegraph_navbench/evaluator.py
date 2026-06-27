"""Relation-level evaluation helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal, Mapping, cast

from scenegraph_navbench.models import Relation

RelationTriple = tuple[str, str, str]
RelationInput = Relation | Mapping[str, object] | RelationTriple
EvaluationMode = Literal["exact", "must_include"]


@dataclass(frozen=True)
class ExpectedRelations:
    """Expected relation triples plus metadata describing how to score them."""

    mode: EvaluationMode
    relations: list[dict[str, str]]


@dataclass(frozen=True)
class EvaluationMetrics:
    """Mode-aware relation evaluation metrics."""

    mode: EvaluationMode
    precision: float | None
    recall: float | None
    f1: float | None
    required_relation_recall: float
    true_positives: int
    predicted: int
    expected: int


def load_expected_relations(path: str | Path) -> ExpectedRelations:
    """Load expected relation dictionaries and evaluation metadata."""
    with Path(path).open("r", encoding="utf-8") as file:
        payload = json.load(file)

    mode = _parse_mode(payload.get("mode", "exact"))
    relations = payload.get("relations")
    if not isinstance(relations, list):
        raise ValueError("expected JSON must contain a 'relations' list")

    expected: list[dict[str, str]] = []
    for index, relation in enumerate(relations):
        if not isinstance(relation, dict):
            raise ValueError(f"relation at index {index} must be an object")

        try:
            source = str(relation["source"])
            relation_name = str(relation["relation"])
            target = str(relation["target"])
        except KeyError as exc:
            raise ValueError(
                f"relation at index {index} is missing key {exc.args[0]!r}"
            ) from exc

        expected.append(
            {
                "source": source,
                "relation": relation_name,
                "target": target,
            }
        )

    return ExpectedRelations(mode=mode, relations=expected)


def relation_triples(
    relations: Iterable[RelationInput],
) -> set[RelationTriple]:
    """Convert Relation models, dictionaries, or triples into comparable triples."""
    return {_relation_triple(relation) for relation in relations}


def evaluate_relations(
    generated: Iterable[RelationInput],
    expected: Iterable[RelationInput] | ExpectedRelations,
    *,
    mode: EvaluationMode = "exact",
) -> EvaluationMetrics:
    """Evaluate generated triples against exact or sparse expected relations."""
    if isinstance(expected, ExpectedRelations):
        mode = expected.mode
        expected_relations = expected.relations
    else:
        expected_relations = expected

    generated_triples = relation_triples(generated)
    expected_triples = relation_triples(expected_relations)

    true_positives = len(generated_triples & expected_triples)
    predicted_count = len(generated_triples)
    expected_count = len(expected_triples)
    required_relation_recall = (
        true_positives / expected_count if expected_count else 1.0
    )

    if mode == "must_include":
        return EvaluationMetrics(
            mode=mode,
            precision=None,
            recall=None,
            f1=None,
            required_relation_recall=required_relation_recall,
            true_positives=true_positives,
            predicted=predicted_count,
            expected=expected_count,
        )

    precision = true_positives / predicted_count if predicted_count else 0.0
    recall = true_positives / expected_count if expected_count else 0.0
    if predicted_count == 0 and expected_count == 0:
        precision = 1.0
        recall = 1.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0.0
        else 0.0
    )

    return EvaluationMetrics(
        mode=mode,
        precision=precision,
        recall=recall,
        f1=f1,
        required_relation_recall=required_relation_recall,
        true_positives=true_positives,
        predicted=predicted_count,
        expected=expected_count,
    )


def _parse_mode(value: object) -> EvaluationMode:
    if value in {"exact", "must_include"}:
        return cast(EvaluationMode, value)
    raise ValueError("expected JSON mode must be 'exact' or 'must_include'")


def _relation_triple(relation: RelationInput) -> RelationTriple:
    if isinstance(relation, Relation):
        return relation.source, relation.relation, relation.target

    if isinstance(relation, tuple):
        if len(relation) != 3:
            raise ValueError("relation tuples must contain source, relation, and target")
        source, relation_name, target = relation
        return str(source), str(relation_name), str(target)

    return (
        str(relation["source"]),
        str(relation["relation"]),
        str(relation["target"]),
    )

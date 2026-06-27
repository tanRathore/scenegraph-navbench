from scenegraph_navbench.models import Detection
from scenegraph_navbench.spatial import SpatialConfig, infer_pairwise_relations


def make_detection(
    detection_id: str,
    bbox: tuple[float, float, float, float],
    *,
    depth: float | None = None,
) -> Detection:
    return Detection(
        id=detection_id,
        label=detection_id,
        bbox=bbox,
        confidence=0.9,
        depth=depth,
    )


def relation_names(source: Detection, target: Detection) -> set[str]:
    relations = infer_pairwise_relations(source, target, 1000, 1000)
    return {relation.relation for relation in relations}


def test_left_of_and_right_of() -> None:
    left = make_detection("left", (100, 100, 200, 200))
    right = make_detection("right", (700, 100, 800, 200))

    assert "left_of" in relation_names(left, right)
    assert "right_of" in relation_names(right, left)


def test_above_and_below() -> None:
    top = make_detection("top", (100, 100, 200, 200))
    bottom = make_detection("bottom", (100, 700, 200, 800))

    assert "above" in relation_names(top, bottom)
    assert "below" in relation_names(bottom, top)


def test_near_and_far() -> None:
    origin = make_detection("origin", (100, 100, 200, 200))
    nearby = make_detection("nearby", (180, 120, 280, 220))
    distant = make_detection("distant", (900, 900, 980, 980))

    assert "near" in relation_names(origin, nearby)
    assert "far" in relation_names(origin, distant)


def test_overlaps() -> None:
    first = make_detection("first", (100, 100, 300, 300))
    second = make_detection("second", (200, 200, 400, 400))

    assert "overlaps" in relation_names(first, second)


def test_in_front_of_and_behind_with_depth() -> None:
    close = make_detection("close", (100, 100, 200, 200), depth=1.0)
    far = make_detection("far", (300, 100, 400, 200), depth=3.0)

    assert "in_front_of" in relation_names(close, far)
    assert "behind" in relation_names(far, close)


def test_missing_depth_does_not_produce_depth_relations() -> None:
    source = make_detection("source", (100, 100, 200, 200), depth=1.0)
    target = make_detection("target", (300, 100, 400, 200))

    names = relation_names(source, target)

    assert "in_front_of" not in names
    assert "behind" not in names


def test_configurable_thresholds_change_near_and_far_behavior() -> None:
    origin = make_detection("origin", (100, 100, 200, 200))
    nearby = make_detection("nearby", (180, 120, 280, 220))
    distant = make_detection("distant", (900, 900, 980, 980))

    strict_config = SpatialConfig(near_threshold=0.03, far_threshold=0.95)

    strict_near_names = {
        relation.relation
        for relation in infer_pairwise_relations(
            origin,
            nearby,
            1000,
            1000,
            config=strict_config,
        )
    }
    strict_far_names = {
        relation.relation
        for relation in infer_pairwise_relations(
            origin,
            distant,
            1000,
            1000,
            config=strict_config,
        )
    }

    assert "near" in relation_names(origin, nearby)
    assert "near" not in strict_near_names
    assert "far" in relation_names(origin, distant)
    assert "far" not in strict_far_names

"""Deterministic spatial geometry helpers."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import hypot
from typing import Sequence

from scenegraph_navbench.models import Detection, Relation

BBoxLike = Sequence[float]


@dataclass(frozen=True)
class SpatialConfig:
    """Configurable thresholds for deterministic spatial relation inference."""

    min_confidence: float = 0.25
    near_threshold: float = 0.22
    far_threshold: float = 0.50
    overlap_threshold: float = 0.0
    depth_epsilon: float = 0.05
    horizontal_margin: float = 0.0
    vertical_margin: float = 0.0

    def __post_init__(self) -> None:
        _check_unit_interval("min_confidence", self.min_confidence)
        _check_unit_interval("near_threshold", self.near_threshold)
        _check_unit_interval("far_threshold", self.far_threshold)
        _check_unit_interval("overlap_threshold", self.overlap_threshold)

        if self.near_threshold > self.far_threshold:
            raise ValueError("near_threshold must be less than or equal to far_threshold")

        _check_non_negative("depth_epsilon", self.depth_epsilon)
        _check_non_negative("horizontal_margin", self.horizontal_margin)
        _check_non_negative("vertical_margin", self.vertical_margin)


def bbox_center(bbox: BBoxLike) -> tuple[float, float]:
    """Return the center point of a bounding box."""
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def bbox_area(bbox: BBoxLike) -> float:
    """Return the area of a bounding box."""
    x1, y1, x2, y2 = bbox
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def intersection_area(a: BBoxLike, b: BBoxLike) -> float:
    """Return the intersection area for two bounding boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    width = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    height = max(0.0, min(ay2, by2) - max(ay1, by1))
    return width * height


def iou(a: BBoxLike, b: BBoxLike) -> float:
    """Return intersection-over-union for two bounding boxes."""
    intersection = intersection_area(a, b)
    union = bbox_area(a) + bbox_area(b) - intersection
    if union <= 0.0:
        return 0.0
    return intersection / union


def center_distance(a: BBoxLike, b: BBoxLike) -> float:
    """Return Euclidean distance between bounding-box centers."""
    ax, ay = bbox_center(a)
    bx, by = bbox_center(b)
    return hypot(ax - bx, ay - by)


def normalized_center_distance(
    a: BBoxLike,
    b: BBoxLike,
    image_width: int,
    image_height: int,
) -> float:
    """Return center distance normalized by the image diagonal."""
    diagonal = hypot(image_width, image_height)
    if diagonal <= 0.0:
        raise ValueError("image dimensions must produce a positive diagonal")
    return center_distance(a, b) / diagonal


def infer_pairwise_relations(
    source: Detection,
    target: Detection,
    image_width: int,
    image_height: int,
    *,
    config: SpatialConfig | None = None,
    near_threshold: float | None = None,
    far_threshold: float | None = None,
    overlap_iou_threshold: float | None = None,
    overlap_threshold: float | None = None,
    depth_epsilon: float | None = None,
    horizontal_margin: float | None = None,
    vertical_margin: float | None = None,
) -> list[Relation]:
    """Infer directed spatial relations from source to target."""
    config = _resolve_config(
        config,
        near_threshold=near_threshold,
        far_threshold=far_threshold,
        overlap_iou_threshold=overlap_iou_threshold,
        overlap_threshold=overlap_threshold,
        depth_epsilon=depth_epsilon,
        horizontal_margin=horizontal_margin,
        vertical_margin=vertical_margin,
    )
    relations: list[Relation] = []

    source_center = bbox_center(source.bbox)
    target_center = bbox_center(target.bbox)
    dx = target_center[0] - source_center[0]
    dy = target_center[1] - source_center[1]

    if dx > config.horizontal_margin:
        relations.append(
            _relation(
                source,
                target,
                "left_of",
                min((abs(dx) - config.horizontal_margin) / image_width, 1.0),
                (
                    f"center_x(source)={source_center[0]:.1f} < "
                    f"center_x(target)={target_center[0]:.1f}; "
                    f"dx={abs(dx):.1f} > "
                    f"horizontal_margin={config.horizontal_margin:.1f}."
                ),
            )
        )
    elif dx < -config.horizontal_margin:
        relations.append(
            _relation(
                source,
                target,
                "right_of",
                min((abs(dx) - config.horizontal_margin) / image_width, 1.0),
                (
                    f"center_x(source)={source_center[0]:.1f} > "
                    f"center_x(target)={target_center[0]:.1f}; "
                    f"dx={abs(dx):.1f} > "
                    f"horizontal_margin={config.horizontal_margin:.1f}."
                ),
            )
        )

    if dy > config.vertical_margin:
        relations.append(
            _relation(
                source,
                target,
                "above",
                min((abs(dy) - config.vertical_margin) / image_height, 1.0),
                (
                    f"center_y(source)={source_center[1]:.1f} < "
                    f"center_y(target)={target_center[1]:.1f}; "
                    f"dy={abs(dy):.1f} > "
                    f"vertical_margin={config.vertical_margin:.1f}."
                ),
            )
        )
    elif dy < -config.vertical_margin:
        relations.append(
            _relation(
                source,
                target,
                "below",
                min((abs(dy) - config.vertical_margin) / image_height, 1.0),
                (
                    f"center_y(source)={source_center[1]:.1f} > "
                    f"center_y(target)={target_center[1]:.1f}; "
                    f"dy={abs(dy):.1f} > "
                    f"vertical_margin={config.vertical_margin:.1f}."
                ),
            )
        )

    normalized_distance = normalized_center_distance(
        source.bbox,
        target.bbox,
        image_width,
        image_height,
    )
    if normalized_distance <= config.near_threshold:
        score = max(0.0, 1.0 - (normalized_distance / config.near_threshold))
        relations.append(
            _relation(
                source,
                target,
                "near",
                score,
                (
                    f"normalized_center_distance={normalized_distance:.3f} <= "
                    f"near_threshold={config.near_threshold:.3f}."
                ),
            )
        )
    elif normalized_distance >= config.far_threshold:
        score = min(
            1.0,
            (normalized_distance - config.far_threshold)
            / max(1.0 - config.far_threshold, 1e-9),
        )
        relations.append(
            _relation(
                source,
                target,
                "far",
                score,
                (
                    f"normalized_center_distance={normalized_distance:.3f} >= "
                    f"far_threshold={config.far_threshold:.3f}."
                ),
            )
        )

    overlap_iou = iou(source.bbox, target.bbox)
    overlap_area = intersection_area(source.bbox, target.bbox)
    if overlap_area > 0.0 and overlap_iou >= config.overlap_threshold:
        relations.append(
            _relation(
                source,
                target,
                "overlaps",
                min(overlap_iou, 1.0),
                (
                    f"intersection_area={overlap_area:.1f} and "
                    f"iou={overlap_iou:.3f} >= "
                    f"overlap_threshold={config.overlap_threshold:.3f}."
                ),
            )
        )

    if source.depth is not None and target.depth is not None:
        depth_delta = source.depth - target.depth
        depth_score = min(abs(depth_delta) / max(source.depth, target.depth), 1.0)
        if depth_delta < -config.depth_epsilon:
            relations.append(
                _relation(
                    source,
                    target,
                    "in_front_of",
                    depth_score,
                    (
                        f"source_depth={source.depth:.2f} < "
                        f"target_depth={target.depth:.2f}; "
                        f"depth_delta={depth_delta:.2f} < "
                        f"-depth_epsilon={-config.depth_epsilon:.2f}."
                    ),
                )
            )
        elif depth_delta > config.depth_epsilon:
            relations.append(
                _relation(
                    source,
                    target,
                    "behind",
                    depth_score,
                    (
                        f"source_depth={source.depth:.2f} > "
                        f"target_depth={target.depth:.2f}; "
                        f"depth_delta={depth_delta:.2f} > "
                        f"depth_epsilon={config.depth_epsilon:.2f}."
                    ),
                )
            )

    return relations


def _relation(
    source: Detection,
    target: Detection,
    relation: str,
    score: float,
    reason: str,
) -> Relation:
    return Relation(
        source=source.id,
        target=target.id,
        relation=relation,
        score=round(score, 3),
        reason=reason,
    )


def _resolve_config(
    config: SpatialConfig | None,
    *,
    near_threshold: float | None,
    far_threshold: float | None,
    overlap_iou_threshold: float | None,
    overlap_threshold: float | None,
    depth_epsilon: float | None,
    horizontal_margin: float | None,
    vertical_margin: float | None,
) -> SpatialConfig:
    config = config or SpatialConfig()
    values = {
        "near_threshold": near_threshold,
        "far_threshold": far_threshold,
        "depth_epsilon": depth_epsilon,
        "horizontal_margin": horizontal_margin,
        "vertical_margin": vertical_margin,
    }
    updates = {name: value for name, value in values.items() if value is not None}

    # Keep the older argument working, while preferring the shorter public name.
    if overlap_iou_threshold is not None:
        updates["overlap_threshold"] = overlap_iou_threshold
    if overlap_threshold is not None:
        updates["overlap_threshold"] = overlap_threshold

    return replace(config, **updates) if updates else config


def _check_unit_interval(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")


def _check_non_negative(name: str, value: float) -> None:
    if value < 0.0:
        raise ValueError(f"{name} must be non-negative")

"""Validated data models for SceneGraphNavBench."""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator


BBox = tuple[float, float, float, float]


class Detection(BaseModel):
    """One object detection from a perception system."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    label: str = Field(min_length=1)
    bbox: BBox
    confidence: Annotated[float, Field(ge=0.0, le=1.0)]
    depth: Annotated[float | None, Field(gt=0.0)] = None

    @field_validator("bbox")
    @classmethod
    def validate_bbox(cls, bbox: BBox) -> BBox:
        if len(bbox) != 4:
            raise ValueError("bbox must contain exactly four numbers: x1, y1, x2, y2")

        x1, y1, x2, y2 = bbox
        if x2 <= x1:
            raise ValueError("bbox x2 must be greater than x1")
        if y2 <= y1:
            raise ValueError("bbox y2 must be greater than y1")
        return bbox


class Scene(BaseModel):
    """A perception snapshot containing image dimensions and detections."""

    model_config = ConfigDict(extra="forbid")

    image_width: int = Field(gt=0)
    image_height: int = Field(gt=0)
    detections: list[Detection]


class Relation(BaseModel):
    """A directed spatial relation between two object detections."""

    model_config = ConfigDict(extra="forbid")

    source: str = Field(min_length=1)
    target: str = Field(min_length=1)
    relation: str = Field(min_length=1)
    score: Annotated[float, Field(ge=0.0, le=1.0)]
    reason: str = Field(min_length=1)


class SceneGraph(BaseModel):
    """A scene graph with object nodes and directed spatial relations."""

    model_config = ConfigDict(extra="forbid")

    nodes: list[Detection]
    relations: list[Relation]

    def get_node(self, node_id: str) -> Detection | None:
        """Return the node with the requested id, if present."""
        return next((node for node in self.nodes if node.id == node_id), None)

    def relations_for(self, node_id: str) -> list[Relation]:
        """Return all relations where the requested id is the source or target."""
        return [
            relation
            for relation in self.relations
            if relation.source == node_id or relation.target == node_id
        ]

    def find_by_label(self, label: str) -> list[Detection]:
        """Return all nodes with a matching label."""
        return [node for node in self.nodes if node.label == label]

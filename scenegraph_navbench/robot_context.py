"""Validated world-frame context for deterministic robot queries."""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, model_validator


class RobotPose(BaseModel):
    """A planar robot pose in the navigation context's coordinate frame."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    x: float
    y: float
    heading_radians: float

    @property
    def position(self) -> tuple[float, float]:
        return self.x, self.y


class NavigationObject(BaseModel):
    """World-frame navigation metadata associated with one graph node."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    object_id: str = Field(min_length=1)
    x: float | None
    y: float | None
    radius: Annotated[float, Field(ge=0.0)]
    role: str | None = None
    is_obstacle: bool

    @model_validator(mode="after")
    def validate_position(self) -> NavigationObject:
        if (self.x is None) != (self.y is None):
            raise ValueError("x and y must either both be provided or both be null")
        if self.role == "":
            raise ValueError("role must be non-empty when provided")
        return self

    @property
    def position(self) -> tuple[float, float] | None:
        if self.x is None or self.y is None:
            return None
        return self.x, self.y


class NavigationContext(BaseModel):
    """Robot pose and per-object metadata in one explicit world frame."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    frame_id: str = Field(min_length=1)
    robot_pose: RobotPose
    robot_radius: Annotated[float, Field(ge=0.0)]
    objects: list[NavigationObject]

    @model_validator(mode="after")
    def validate_unique_object_ids(self) -> NavigationContext:
        object_ids = [item.object_id for item in self.objects]
        if len(object_ids) != len(set(object_ids)):
            raise ValueError("navigation object_id values must be unique")
        return self

    def get_object(self, object_id: str) -> NavigationObject | None:
        """Return navigation metadata for an object, if supplied."""
        return next(
            (item for item in self.objects if item.object_id == object_id),
            None,
        )

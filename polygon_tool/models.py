from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple


class DrawingMode(Enum):
    POLYGON = "polygon"


@dataclass
class Shape:
    shape_id: int
    class_id: int
    class_name: str
    shape_type: str
    points: List[Tuple[float, float]]
    color: Tuple[int, int, int]
    width: int = 2
    description: str = ""

    def to_dict(self) -> dict:
        return {
            "shape_id": self.shape_id,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "shape_type": self.shape_type,
            "points": [list(p) for p in self.points],
            "color": list(self.color),
            "width": self.width,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Shape":
        return cls(
            shape_id=data["shape_id"],
            class_id=data["class_id"],
            class_name=data["class_name"],
            shape_type=data["shape_type"],
            points=[tuple(p) for p in data["points"]],
            color=tuple(data["color"]),
            width=data.get("width", 2),
            description=data.get("description", ""),
        )

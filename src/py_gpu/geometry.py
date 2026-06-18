"""Simple screen-space geometry contracts for raster backends."""

from __future__ import annotations

from dataclasses import dataclass

from .buffers import ColorLike, as_rgb


@dataclass(frozen=True)
class ScreenVertex:
    """A projected vertex in pixel coordinates plus depth."""

    x: float
    y: float
    z: float


@dataclass(frozen=True)
class ScreenTriangle:
    """A projected triangle with one flat RGB color."""

    a: ScreenVertex
    b: ScreenVertex
    c: ScreenVertex
    color: ColorLike = (255, 255, 255)

    def __post_init__(self) -> None:
        object.__setattr__(self, "color", as_rgb(self.color))


@dataclass(frozen=True)
class RasterBatch:
    """A batch of projected triangles ready for a raster backend."""

    triangles: tuple[ScreenTriangle, ...]

    def __init__(self, triangles) -> None:
        object.__setattr__(self, "triangles", tuple(triangles))

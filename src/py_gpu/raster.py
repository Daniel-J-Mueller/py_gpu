"""Reference screen-space raster backend."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, floor

from .backends import BackendCapabilities
from .buffers import ColorLike, DepthBuffer, FrameBuffer, as_rgb
from .geometry import RasterBatch, ScreenTriangle


@dataclass
class CPURasterBackend:
    """Small dependency-free raster backend for correctness and fallback use."""

    capabilities: BackendCapabilities = BackendCapabilities(
        name="CPU screen-space rasterizer",
        accelerated=False,
        supports_depth=True,
    )

    def render(
        self,
        batch: RasterBatch,
        width: int,
        height: int,
        *,
        background: ColorLike = (0, 0, 0),
        target: FrameBuffer | None = None,
    ) -> FrameBuffer:
        frame = target or FrameBuffer.new(width, height, background)
        if frame.width != width or frame.height != height:
            raise ValueError("target frame dimensions must match render dimensions")
        frame.clear(background)
        depth = DepthBuffer.new(width, height)
        for triangle in batch.triangles:
            _draw_triangle(frame, depth, triangle)
        return frame


def _draw_triangle(frame: FrameBuffer, depth: DepthBuffer, triangle: ScreenTriangle) -> None:
    a, b, c = triangle.a, triangle.b, triangle.c
    min_x = max(0, floor(min(a.x, b.x, c.x)))
    max_x = min(frame.width - 1, ceil(max(a.x, b.x, c.x)))
    min_y = max(0, floor(min(a.y, b.y, c.y)))
    max_y = min(frame.height - 1, ceil(max(a.y, b.y, c.y)))
    area = _edge(a.x, a.y, b.x, b.y, c.x, c.y)
    if abs(area) < 1e-12:
        return

    inv_area = 1.0 / area
    ax, ay, az = a.x, a.y, a.z
    bx, by, bz = b.x, b.y, b.z
    cx, cy, cz = c.x, c.y, c.z
    width = frame.width
    pixels = frame.pixels
    depth_values = depth.values
    red, green, blue = as_rgb(triangle.color)

    for y in range(min_y, max_y + 1):
        row_index = y * width
        py = y + 0.5
        for x in range(min_x, max_x + 1):
            px = x + 0.5
            w0 = ((px - bx) * (cy - by) - (py - by) * (cx - bx)) * inv_area
            w1 = ((px - cx) * (ay - cy) - (py - cy) * (ax - cx)) * inv_area
            w2 = ((px - ax) * (by - ay) - (py - ay) * (bx - ax)) * inv_area
            if w0 < -1e-9 or w1 < -1e-9 or w2 < -1e-9:
                continue
            z = w0 * az + w1 * bz + w2 * cz
            index = row_index + x
            if z < depth_values[index]:
                depth_values[index] = z
                offset = index * 3
                pixels[offset] = red
                pixels[offset + 1] = green
                pixels[offset + 2] = blue


def _edge(ax: float, ay: float, bx: float, by: float, px: float, py: float) -> float:
    return (px - ax) * (by - ay) - (py - ay) * (bx - ax)

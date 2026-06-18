"""Optional NumPy raster backend."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, floor

import numpy as np

from .backends import BackendCapabilities
from .buffers import ColorLike, FrameBuffer, as_rgb
from .geometry import RasterBatch, ScreenTriangle


@dataclass
class NumpyRasterBackend:
    """Vectorized screen-space raster backend for machines with NumPy."""

    capabilities: BackendCapabilities = BackendCapabilities(
        name="NumPy screen-space rasterizer",
        accelerated=True,
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
        bg = np.array(as_rgb(background), dtype=np.uint8)
        frame = np.empty((height, width, 3), dtype=np.uint8)
        frame[:, :] = bg
        depth = np.full((height, width), np.inf, dtype=np.float32)
        for triangle in batch.triangles:
            _draw_triangle(frame, depth, triangle)

        payload = frame.tobytes()
        if target is not None:
            if target.width != width or target.height != height:
                raise ValueError("target frame dimensions must match render dimensions")
            target.pixels[:] = payload
            return target
        return FrameBuffer(width, height, bytearray(payload))


def _draw_triangle(frame: np.ndarray, depth: np.ndarray, triangle: ScreenTriangle) -> None:
    height, width = depth.shape
    a, b, c = triangle.a, triangle.b, triangle.c
    min_x = max(0, floor(min(a.x, b.x, c.x)))
    max_x = min(width - 1, ceil(max(a.x, b.x, c.x)))
    min_y = max(0, floor(min(a.y, b.y, c.y)))
    max_y = min(height - 1, ceil(max(a.y, b.y, c.y)))
    area = _edge(a.x, a.y, b.x, b.y, c.x, c.y)
    if abs(area) < 1e-12 or min_x > max_x or min_y > max_y:
        return

    xs = np.arange(min_x, max_x + 1, dtype=np.float32) + 0.5
    ys = np.arange(min_y, max_y + 1, dtype=np.float32) + 0.5
    px, py = np.meshgrid(xs, ys)
    inv_area = 1.0 / area
    w0 = ((px - b.x) * (c.y - b.y) - (py - b.y) * (c.x - b.x)) * inv_area
    w1 = ((px - c.x) * (a.y - c.y) - (py - c.y) * (a.x - c.x)) * inv_area
    w2 = ((px - a.x) * (b.y - a.y) - (py - a.y) * (b.x - a.x)) * inv_area
    mask = (w0 >= -1e-9) & (w1 >= -1e-9) & (w2 >= -1e-9)
    if not mask.any():
        return

    z = w0 * a.z + w1 * b.z + w2 * c.z
    depth_view = depth[min_y : max_y + 1, min_x : max_x + 1]
    update = mask & (z < depth_view)
    if not update.any():
        return

    depth_view[update] = z[update]
    frame_view = frame[min_y : max_y + 1, min_x : max_x + 1]
    frame_view[update] = np.array(as_rgb(triangle.color), dtype=np.uint8)


def _edge(ax: float, ay: float, bx: float, by: float, px, py):
    return (px - ax) * (by - ay) - (py - ay) * (bx - ax)

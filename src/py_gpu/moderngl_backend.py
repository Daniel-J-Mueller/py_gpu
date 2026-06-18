"""ModernGL raster backend."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from struct import pack

from .backends import BackendCapabilities
from .buffers import ColorLike, FrameBuffer, as_rgb
from .geometry import RasterBatch

_NUMPY = None
_NUMPY_CHECKED = False


VERTEX_SHADER = """
#version 330
in vec3 in_position;
in vec3 in_color;
out vec3 v_color;

void main() {
    gl_Position = vec4(in_position, 1.0);
    v_color = in_color;
}
"""


FRAGMENT_SHADER = """
#version 330
in vec3 v_color;
out vec4 frag_color;

void main() {
    frag_color = vec4(v_color, 1.0);
}
"""


@dataclass
class ModernGLRasterBackend:
    """GPU raster backend using an offscreen ModernGL context."""

    capabilities: BackendCapabilities = BackendCapabilities(
        name="ModernGL GPU rasterizer",
        accelerated=True,
        supports_depth=True,
        supports_vertex_attributes=True,
    )
    _ctx: object | None = field(default=None, init=False, repr=False)
    _program: object | None = field(default=None, init=False, repr=False)
    _framebuffer_cache: dict[tuple[int, int], tuple[object, object, object]] = field(default_factory=dict, init=False, repr=False)

    def _ensure_context(self):
        if self._ctx is not None and self._program is not None:
            return self._ctx, self._program

        import moderngl

        self._ctx = moderngl.create_standalone_context()
        self._ctx.enable(moderngl.DEPTH_TEST)
        self._program = self._ctx.program(vertex_shader=VERTEX_SHADER, fragment_shader=FRAGMENT_SHADER)
        return self._ctx, self._program

    def render(
        self,
        batch: RasterBatch,
        width: int,
        height: int,
        *,
        background: ColorLike = (0, 0, 0),
        target: FrameBuffer | None = None,
    ) -> FrameBuffer:
        return self.render_vertex_bytes(_batch_to_vertices(batch, width, height), width, height, background=background, target=target)

    def render_vertex_bytes(
        self,
        vertex_bytes: bytes,
        width: int,
        height: int,
        *,
        background: ColorLike = (0, 0, 0),
        target: FrameBuffer | None = None,
    ) -> FrameBuffer:
        ctx, program = self._ensure_context()
        framebuffer = self._framebuffer(width, height)
        framebuffer.use()
        red, green, blue = (channel / 255.0 for channel in as_rgb(background))
        framebuffer.clear(red, green, blue, 1.0, depth=1.0)

        if vertex_bytes:
            vbo = ctx.buffer(vertex_bytes)
            vao = ctx.vertex_array(program, [(vbo, "3f 3f", "in_position", "in_color")])
            vao.render()
            vao.release()
            vbo.release()

        data = framebuffer.read(components=3, alignment=1)
        payload = _flip_rows(data, width, height)
        if target is not None:
            if target.width != width or target.height != height:
                raise ValueError("target frame dimensions must match render dimensions")
            target.pixels[:] = payload
            return target
        return FrameBuffer(width, height, bytearray(payload))

    def _framebuffer(self, width: int, height: int):
        key = (width, height)
        cached = self._framebuffer_cache.get(key)
        if cached is not None:
            return cached[0]

        ctx, _program = self._ensure_context()
        color = ctx.texture((width, height), 3)
        depth = ctx.depth_renderbuffer((width, height))
        framebuffer = ctx.framebuffer(color_attachments=[color], depth_attachment=depth)
        self._framebuffer_cache[key] = (framebuffer, color, depth)
        return framebuffer


def _batch_to_vertices(batch: RasterBatch, width: int, height: int) -> bytes:
    if not batch.triangles:
        return b""

    z_values = [
        vertex.z
        for triangle in batch.triangles
        for vertex in (triangle.a, triangle.b, triangle.c)
        if isfinite(vertex.z)
    ]
    if z_values:
        near = min(z_values)
        far = max(z_values)
    else:
        near = 0.0
        far = 1.0
    z_span = max(1e-9, far - near)
    x_scale = 2.0 / max(1.0, width - 1)
    y_scale = 2.0 / max(1.0, height - 1)

    payload = bytearray(len(batch.triangles) * 3 * 6 * 4)
    offset = 0
    for triangle in batch.triangles:
        color = tuple(channel / 255.0 for channel in triangle.color)
        for vertex in (triangle.a, triangle.b, triangle.c):
            ndc_x = vertex.x * x_scale - 1.0
            ndc_y = 1.0 - vertex.y * y_scale
            ndc_z = -1.0 + 2.0 * ((vertex.z - near) / z_span)
            payload[offset : offset + 24] = pack("ffffff", ndc_x, ndc_y, ndc_z, color[0], color[1], color[2])
            offset += 24
    return bytes(payload)


def _flip_rows(data: bytes, width: int, height: int) -> bytes:
    numpy = _numpy()
    if numpy is not None:
        try:
            return numpy.frombuffer(data, dtype=numpy.uint8).reshape((height, width * 3))[::-1].copy().tobytes()
        except Exception:
            pass
    row_size = width * 3
    payload = bytearray(len(data))
    for y in range(height):
        src_start = (height - 1 - y) * row_size
        dst_start = y * row_size
        payload[dst_start : dst_start + row_size] = data[src_start : src_start + row_size]
    return bytes(payload)


def _numpy():
    global _NUMPY, _NUMPY_CHECKED
    if _NUMPY_CHECKED:
        return _NUMPY
    _NUMPY_CHECKED = True
    try:
        import numpy
    except Exception:
        _NUMPY = None
    else:
        _NUMPY = numpy
    return _NUMPY

"""Adapter for rendering py_3d scenes through py_gpu batches."""

from __future__ import annotations

from dataclasses import dataclass
from math import radians, tan

from ..backends import RasterBackend, select_backend
from ..buffers import FrameBuffer
from ..geometry import RasterBatch, ScreenTriangle, ScreenVertex


def scene_to_raster_batch(scene, camera, settings) -> RasterBatch:
    """Project a py_3d scene into a flat screen-space triangle batch."""

    projector = _Projector(camera, settings.width, settings.height)
    triangles: list[ScreenTriangle] = []
    for obj in scene.objects:
        for triangle in _triangles_for_py3d(obj, settings):
            a = projector.project(triangle.a)
            b = projector.project(triangle.b)
            c = projector.project(triangle.c)
            if a is None or b is None or c is None:
                continue
            triangles.append(ScreenTriangle(a, b, c, _material_color(triangle.material)))
    return RasterBatch(triangles)


@dataclass
class Py3DRasterRenderer:
    """Renderer-compatible adapter for py_3d's ``RenderEngine``.

    By default this adapter prioritizes py_3d visual parity over raw speed and
    delegates to py_3d's reference renderer. Set ``reference_compatible=False``
    to use the experimental flat batch GPU path for benchmarks.
    """

    backend_impl: RasterBackend | None = None
    reference_compatible: bool = True
    name: str = "py_gpu py_3d batch renderer"
    backend: str = "py_gpu"

    def __post_init__(self) -> None:
        if self.backend_impl is None and not self.reference_compatible:
            self.backend_impl = select_backend()

    def render(self, scene, camera, settings, target=None):
        if self.reference_compatible:
            from py_3d import CPURenderer

            return CPURenderer(cache_static_geometry=False).render(scene, camera, settings, target)
        if self.backend_impl is None:
            self.backend_impl = select_backend()
        batch = scene_to_raster_batch(scene, camera, settings)
        background = _material_color(settings.background)
        frame = self.backend_impl.render(batch, settings.width, settings.height, background=background)
        return frame_to_py3d_pixel_buffer(frame, target)


def frame_to_py3d_pixel_buffer(frame: FrameBuffer, target=None):
    from py_3d import Color, PixelBuffer

    pixels = [
        Color(frame.pixels[index], frame.pixels[index + 1], frame.pixels[index + 2])
        for index in range(0, len(frame.pixels), 3)
    ]
    if target is not None:
        if target.width != frame.width or target.height != frame.height:
            raise ValueError("target buffer dimensions must match rendered frame")
        target.pixels[:] = pixels
        return target
    return PixelBuffer(frame.width, frame.height, pixels)


@dataclass(frozen=True)
class _Projector:
    camera: object
    width: int
    height: int

    def __post_init__(self) -> None:
        right, true_up, forward = self.camera.basis()
        object.__setattr__(self, "right", right)
        object.__setattr__(self, "true_up", true_up)
        object.__setattr__(self, "forward", forward)
        object.__setattr__(self, "aspect", self.width / self.height)
        object.__setattr__(self, "focal", 1.0 / tan(radians(self.camera.fov_degrees) / 2.0))
        object.__setattr__(self, "half_width", 0.5 * (self.width - 1))
        object.__setattr__(self, "half_height", 0.5 * (self.height - 1))

    def project(self, point) -> ScreenVertex | None:
        relative = point - self.camera.position
        camera_x = relative.dot(self.right)
        camera_y = relative.dot(self.true_up)
        camera_z = relative.dot(self.forward)
        if camera_z < self.camera.near or camera_z > self.camera.far:
            return None
        ndc_x = (camera_x * self.focal / self.aspect) / camera_z
        ndc_y = (camera_y * self.focal) / camera_z
        return ScreenVertex(
            (ndc_x + 1.0) * self.half_width,
            (1.0 - ndc_y) * self.half_height,
            camera_z,
        )


def _triangles_for_py3d(obj, settings):
    try:
        from py_3d import Bowl, Capsule, Mesh, Sphere, Triangle
    except Exception as exc:  # pragma: no cover - only reached without optional py_3d
        raise RuntimeError("py_3d must be importable to use py_gpu.adapters.py3d") from exc

    if isinstance(obj, Triangle):
        return (obj,)
    if isinstance(obj, Mesh):
        return obj.to_triangles()
    if isinstance(obj, (Bowl, Capsule, Sphere)):
        return obj.to_triangles(segments=settings.sphere_segments, rings=settings.sphere_rings)
    to_triangles = getattr(obj, "to_triangles", None)
    if callable(to_triangles):
        return tuple(to_triangles())
    return ()


def _material_color(value) -> tuple[int, int, int]:
    color = getattr(value, "color", value)
    to_tuple = getattr(color, "to_tuple", None)
    if callable(to_tuple):
        return to_tuple()
    if hasattr(color, "r") and hasattr(color, "g") and hasattr(color, "b"):
        return (color.r, color.g, color.b)
    return (int(color[0]), int(color[1]), int(color[2]))

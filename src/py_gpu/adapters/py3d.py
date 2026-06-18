"""Adapter for rendering py_3d scenes through py_gpu batches."""

from __future__ import annotations

from dataclasses import dataclass
from math import radians, sqrt, tan

from ..backends import RasterBackend, select_backend
from ..buffers import FrameBuffer
from ..geometry import RasterBatch, ScreenTriangle, ScreenVertex


def scene_to_raster_batch(scene, camera, settings, *, wire_width: float = 1.5) -> RasterBatch:
    """Project a py_3d scene into a flat screen-space triangle batch."""

    projector = _Projector(camera, settings.width, settings.height)
    triangles: list[ScreenTriangle] = []
    for obj in scene.objects:
        if _is_py3d_line(obj):
            start = projector.project(obj.start)
            end = projector.project(obj.end)
            if start is not None and end is not None:
                triangles.extend(_line_to_triangles(start, end, _material_color(obj.material), wire_width))
            continue
        for triangle in _triangles_for_py3d(obj, settings):
            if _culled_by_distance(triangle, camera, settings):
                continue
            wireframe = getattr(settings, "wireframe", False)
            color = _wire_material_color(triangle, settings) if wireframe else _lit_material_color(scene, triangle, camera, settings)
            if _culled_by_facing(triangle, camera, settings):
                continue
            a = projector.project(triangle.a)
            b = projector.project(triangle.b)
            c = projector.project(triangle.c)
            if a is None or b is None or c is None:
                continue
            if wireframe:
                triangles.extend(_line_to_triangles(a, b, color, wire_width))
                triangles.extend(_line_to_triangles(b, c, color, wire_width))
                triangles.extend(_line_to_triangles(c, a, color, wire_width))
            else:
                triangles.append(ScreenTriangle(a, b, c, color))
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
    prefer_backend: str = "auto"
    name: str = "py_gpu py_3d batch renderer"
    backend: str = "py_gpu"

    def __post_init__(self) -> None:
        if self.backend_impl is None and not self.reference_compatible:
            self.backend_impl = select_backend(self.prefer_backend)

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
    from py_3d import PixelBuffer

    if target is not None:
        if target.width != frame.width or target.height != frame.height:
            raise ValueError("target buffer dimensions must match rendered frame")
        target.pixels = PixelBuffer.from_rgb_bytes(frame.width, frame.height, frame.pixels).pixels
        return target
    return PixelBuffer.from_rgb_bytes(frame.width, frame.height, frame.pixels)


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
        from py_3d import BlobSurface, Bowl, Capsule, Mesh, Sphere, Triangle
    except Exception as exc:  # pragma: no cover - only reached without optional py_3d
        raise RuntimeError("py_3d must be importable to use py_gpu.adapters.py3d") from exc

    if isinstance(obj, Triangle):
        return (obj,)
    if isinstance(obj, Mesh):
        return obj.to_triangles()
    if isinstance(obj, (BlobSurface, Bowl, Capsule, Sphere)):
        return obj.to_triangles(segments=settings.sphere_segments, rings=settings.sphere_rings)
    to_triangles = getattr(obj, "to_triangles", None)
    if callable(to_triangles):
        return tuple(to_triangles())
    return ()


def _is_py3d_line(obj) -> bool:
    try:
        from py_3d import Line3
    except Exception:
        return False
    return isinstance(obj, Line3)


def _culled_by_distance(triangle, camera, settings) -> bool:
    max_distance = getattr(settings, "max_render_distance", None)
    return max_distance is not None and triangle.center().distance_to(camera.position) > max_distance


def _culled_by_facing(triangle, camera, settings) -> bool:
    if not getattr(settings, "cull_backfaces", False):
        return False
    center = triangle.center()
    normal = triangle.normal()
    view_direction = (camera.position - center).normalized()
    return normal.dot(view_direction) <= 0.0


def _lit_material_color(scene, triangle, camera, settings) -> tuple[int, int, int]:
    material = triangle.material
    center = triangle.center()
    normal = triangle.normal()
    view_direction = (camera.position - center).normalized(normal)
    if getattr(settings, "two_sided_lighting", True) and normal.dot(view_direction) < 0.0:
        normal = -normal

    diffuse = [0.0, 0.0, 0.0]
    specular = [0.0, 0.0, 0.0]
    specular_enabled = getattr(material, "specular", 0.0) > 0.0 or getattr(material, "reflectivity", 0.0) > 0.0
    shininess = getattr(material, "shininess", 32.0)
    for light in getattr(scene, "lights", ()):
        sample_method = getattr(light, "sample", None)
        if not callable(sample_method):
            continue
        sample = sample_method(center)
        light_direction = sample.direction.normalized()
        strength = max(0.0, normal.dot(light_direction)) * sample.intensity
        lr, lg, lb = sample.color.to_floats()
        diffuse[0] += lr * strength
        diffuse[1] += lg * strength
        diffuse[2] += lb * strength
        if specular_enabled and strength > 0.0:
            halfway = (light_direction + view_direction).normalized(light_direction)
            highlight = max(0.0, normal.dot(halfway)) ** shininess * sample.intensity
            specular[0] += lr * highlight
            specular[1] += lg * highlight
            specular[2] += lb * highlight

    base_color = None
    if getattr(material, "texture", None) is not None and triangle.has_texture_coordinates():
        u = ((triangle.uv_a or (0.0, 0.0))[0] + (triangle.uv_b or (0.0, 0.0))[0] + (triangle.uv_c or (0.0, 0.0))[0]) / 3.0
        v = ((triangle.uv_a or (0.0, 0.0))[1] + (triangle.uv_b or (0.0, 0.0))[1] + (triangle.uv_c or (0.0, 0.0))[1]) / 3.0
        base_color = material.color_at((u, v))

    color = material.shade(tuple(diffuse), ambient=getattr(settings, "ambient", 0.0), base_color=base_color, specular_light=tuple(specular))
    return _apply_gamma(_color_to_rgb(color), getattr(settings, "gamma", 1.0))


def _wire_material_color(triangle, settings) -> tuple[int, int, int]:
    material = triangle.material
    if getattr(material, "texture", None) is not None and triangle.has_texture_coordinates():
        u = ((triangle.uv_a or (0.0, 0.0))[0] + (triangle.uv_b or (0.0, 0.0))[0] + (triangle.uv_c or (0.0, 0.0))[0]) / 3.0
        v = ((triangle.uv_a or (0.0, 0.0))[1] + (triangle.uv_b or (0.0, 0.0))[1] + (triangle.uv_c or (0.0, 0.0))[1]) / 3.0
        return _apply_gamma(_color_to_rgb(material.color_at((u, v))), getattr(settings, "gamma", 1.0))
    base = material.color
    emission = getattr(material, "emission", None)
    if emission is not None:
        rgb = _color_to_rgb(base)
        ergb = _color_to_rgb(emission)
        base = tuple(min(255, rgb[index] + ergb[index]) for index in range(3))
    return _apply_gamma(_color_to_rgb(base), getattr(settings, "gamma", 1.0))


def _material_color(value) -> tuple[int, int, int]:
    color = getattr(value, "color", value)
    return _color_to_rgb(color)


def _color_to_rgb(color) -> tuple[int, int, int]:
    to_tuple = getattr(color, "to_tuple", None)
    if callable(to_tuple):
        return to_tuple()
    if hasattr(color, "r") and hasattr(color, "g") and hasattr(color, "b"):
        return (color.r, color.g, color.b)
    return (int(color[0]), int(color[1]), int(color[2]))


def _apply_gamma(color: tuple[int, int, int], gamma: float) -> tuple[int, int, int]:
    gamma = max(0.01, float(gamma))
    if abs(gamma - 1.0) < 1e-9:
        return color
    inverse = 1.0 / gamma
    return tuple(max(0, min(255, int(round(((channel / 255.0) ** inverse) * 255.0)))) for channel in color)


def _line_to_triangles(start: ScreenVertex, end: ScreenVertex, color: tuple[int, int, int], width: float) -> tuple[ScreenTriangle, ...]:
    dx = end.x - start.x
    dy = end.y - start.y
    length = sqrt(dx * dx + dy * dy)
    if length <= 1e-9:
        return ()
    half_width = max(0.5, width * 0.5)
    nx = -dy / length * half_width
    ny = dx / length * half_width
    a = ScreenVertex(start.x + nx, start.y + ny, start.z)
    b = ScreenVertex(start.x - nx, start.y - ny, start.z)
    c = ScreenVertex(end.x + nx, end.y + ny, end.z)
    d = ScreenVertex(end.x - nx, end.y - ny, end.z)
    return (ScreenTriangle(a, b, c, color), ScreenTriangle(c, b, d, color))

"""Adapter for rendering py_3d scenes through py_gpu batches."""

from __future__ import annotations

from array import array
from dataclasses import dataclass, field
from math import radians, sqrt, tan

from ..backends import RasterBackend, select_backend
from ..buffers import FrameBuffer
from ..geometry import RasterBatch, ScreenTriangle, ScreenVertex


def scene_to_raster_batch(
    scene,
    camera,
    settings,
    *,
    wire_width: float = 1.5,
    fast_materials: bool = False,
    fast_color_cache: dict | None = None,
) -> RasterBatch:
    """Project a py_3d scene into a flat screen-space triangle batch."""

    projector = _Projector(camera, settings.width, settings.height)
    triangles: list[ScreenTriangle] = []
    max_distance = getattr(settings, "max_render_distance", None)
    cull_backfaces = getattr(settings, "cull_backfaces", False)
    for obj in scene.objects:
        if _is_py3d_line(obj):
            start = projector.project(obj.start)
            end = projector.project(obj.end)
            if start is not None and end is not None:
                triangles.extend(_line_to_triangles(start, end, _material_color(obj.material), wire_width))
            continue
        for triangle in _triangles_for_py3d(obj, settings):
            center = None
            normal = None
            if max_distance is not None:
                center = triangle.center()
                if center.distance_to(camera.position) > max_distance:
                    continue
            if cull_backfaces:
                center = center or triangle.center()
                normal = triangle.normal()
                if _culled_by_facing(center, normal, camera):
                    continue
            a = projector.project(triangle.a)
            b = projector.project(triangle.b)
            c = projector.project(triangle.c)
            if a is None or b is None or c is None:
                continue
            wireframe = getattr(settings, "wireframe", False)
            if wireframe:
                color = _wire_material_color(triangle, settings)
            elif fast_materials:
                color = _fast_material_color(triangle, settings, fast_color_cache)
            else:
                center = center or triangle.center()
                normal = normal or triangle.normal()
                color = _lit_material_color(scene, triangle, camera, settings, center, normal)
            if wireframe:
                triangles.extend(_line_to_triangles(a, b, color, wire_width))
                triangles.extend(_line_to_triangles(b, c, color, wire_width))
                triangles.extend(_line_to_triangles(c, a, color, wire_width))
            else:
                triangles.append(ScreenTriangle(a, b, c, color))
    return RasterBatch(triangles)


def scene_to_moderngl_vertex_bytes(
    scene,
    camera,
    settings,
    *,
    wire_width: float = 1.5,
    fast_color_cache: dict | None = None,
) -> bytes:
    """Project a py_3d scene directly into ModernGL vertex bytes."""

    api = _direct_geometry_api()
    projector = _NDCProjector(camera, settings.width, settings.height)
    screen_projector = _Projector(camera, settings.width, settings.height)
    payload = array("f")
    max_distance = getattr(settings, "max_render_distance", None)
    max_distance_squared = None if max_distance is None else max_distance * max_distance

    for obj in scene.objects:
        if isinstance(obj, api["Line3"]):
            start = screen_projector.project(obj.start)
            end = screen_projector.project(obj.end)
            if start is None or end is None:
                continue
            for triangle in _line_to_triangles(start, end, _material_color(obj.material), wire_width):
                _append_screen_triangle_vertices(payload, triangle, settings.width, settings.height)
            continue

        if isinstance(obj, api["Sphere"]):
            _append_sphere_vertices(payload, obj, settings, projector, max_distance_squared, fast_color_cache, api)
            continue
        if isinstance(obj, api["Bowl"]):
            _append_bowl_vertices(payload, obj, settings, projector, max_distance_squared, fast_color_cache, api)
            continue

        for triangle in _triangles_for_py3d(obj, settings):
            color = _fast_material_color(triangle, settings, fast_color_cache)
            _append_triangle_vertices(
                payload,
                projector,
                triangle.a.x,
                triangle.a.y,
                triangle.a.z,
                triangle.b.x,
                triangle.b.y,
                triangle.b.z,
                triangle.c.x,
                triangle.c.y,
                triangle.c.z,
                color,
                max_distance_squared,
            )
    return payload.tobytes()


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
    fast_materials: bool = False
    name: str = "py_gpu py_3d batch renderer"
    backend: str = "py_gpu"
    _fast_color_cache: dict = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.backend_impl is None and not self.reference_compatible:
            self.backend_impl = select_backend(self.prefer_backend)

    def render(self, scene, camera, settings, target=None):
        if self.reference_compatible:
            from py_3d import CPURenderer

            return CPURenderer(cache_static_geometry=False).render(scene, camera, settings, target)
        if self.backend_impl is None:
            self.backend_impl = select_backend()
        if len(self._fast_color_cache) > 20000:
            self._fast_color_cache.clear()
        background = _material_color(settings.background)
        render_vertex_bytes = getattr(self.backend_impl, "render_vertex_bytes", None)
        if self.fast_materials and callable(render_vertex_bytes) and not getattr(settings, "wireframe", False) and not getattr(settings, "cull_backfaces", False):
            vertex_bytes = scene_to_moderngl_vertex_bytes(
                scene,
                camera,
                settings,
                fast_color_cache=self._fast_color_cache,
            )
            frame = render_vertex_bytes(vertex_bytes, settings.width, settings.height, background=background)
            return frame_to_py3d_pixel_buffer(frame, target)
        batch = scene_to_raster_batch(
            scene,
            camera,
            settings,
            fast_materials=self.fast_materials,
            fast_color_cache=self._fast_color_cache if self.fast_materials else None,
        )
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
        camera_position = self.camera.position
        object.__setattr__(self, "right", right)
        object.__setattr__(self, "true_up", true_up)
        object.__setattr__(self, "forward", forward)
        object.__setattr__(self, "camera_x", camera_position.x)
        object.__setattr__(self, "camera_y", camera_position.y)
        object.__setattr__(self, "camera_z", camera_position.z)
        object.__setattr__(self, "aspect", self.width / self.height)
        object.__setattr__(self, "focal", 1.0 / tan(radians(self.camera.fov_degrees) / 2.0))
        object.__setattr__(self, "half_width", 0.5 * (self.width - 1))
        object.__setattr__(self, "half_height", 0.5 * (self.height - 1))

    def project(self, point) -> ScreenVertex | None:
        try:
            point_x = point.x
            point_y = point.y
            point_z = point.z
        except AttributeError:
            point_x, point_y, point_z = point
        relative_x = point_x - self.camera_x
        relative_y = point_y - self.camera_y
        relative_z = point_z - self.camera_z
        camera_x = relative_x * self.right.x + relative_y * self.right.y + relative_z * self.right.z
        camera_y = relative_x * self.true_up.x + relative_y * self.true_up.y + relative_z * self.true_up.z
        camera_z = relative_x * self.forward.x + relative_y * self.forward.y + relative_z * self.forward.z
        if camera_z < self.camera.near or camera_z > self.camera.far:
            return None
        ndc_x = (camera_x * self.focal / self.aspect) / camera_z
        ndc_y = (camera_y * self.focal) / camera_z
        return ScreenVertex(
            (ndc_x + 1.0) * self.half_width,
            (1.0 - ndc_y) * self.half_height,
            camera_z,
        )


@dataclass(frozen=True)
class _NDCProjector:
    camera: object
    width: int
    height: int

    def __post_init__(self) -> None:
        right, true_up, forward = self.camera.basis()
        position = self.camera.position
        object.__setattr__(self, "right", right)
        object.__setattr__(self, "true_up", true_up)
        object.__setattr__(self, "forward", forward)
        object.__setattr__(self, "camera_x", position.x)
        object.__setattr__(self, "camera_y", position.y)
        object.__setattr__(self, "camera_z", position.z)
        object.__setattr__(self, "aspect", self.width / self.height)
        object.__setattr__(self, "focal", 1.0 / tan(radians(self.camera.fov_degrees) / 2.0))
        object.__setattr__(self, "depth_span", max(1e-9, self.camera.far - self.camera.near))

    def project_xyz(self, point_x: float, point_y: float, point_z: float) -> tuple[float, float, float] | None:
        relative_x = point_x - self.camera_x
        relative_y = point_y - self.camera_y
        relative_z = point_z - self.camera_z
        camera_x = relative_x * self.right.x + relative_y * self.right.y + relative_z * self.right.z
        camera_y = relative_x * self.true_up.x + relative_y * self.true_up.y + relative_z * self.true_up.z
        camera_z = relative_x * self.forward.x + relative_y * self.forward.y + relative_z * self.forward.z
        if camera_z < self.camera.near or camera_z > self.camera.far:
            return None
        return (
            (camera_x * self.focal / self.aspect) / camera_z,
            (camera_y * self.focal) / camera_z,
            -1.0 + 2.0 * ((camera_z - self.camera.near) / self.depth_span),
        )


def _direct_geometry_api() -> dict[str, object]:
    try:
        from py_3d import Bowl, Line3, Sphere
        from py_3d.primitives import _bowl_template, _rotate_euler_precomputed, _rotation_terms, _sphere_template
    except Exception as exc:  # pragma: no cover - only reached without optional py_3d
        raise RuntimeError("py_3d must be importable to use py_gpu.adapters.py3d") from exc
    return {
        "Bowl": Bowl,
        "Line3": Line3,
        "Sphere": Sphere,
        "_bowl_template": _bowl_template,
        "_rotate_euler_precomputed": _rotate_euler_precomputed,
        "_rotation_terms": _rotation_terms,
        "_sphere_template": _sphere_template,
    }


def _append_sphere_vertices(payload, sphere, settings, projector: _NDCProjector, max_distance_squared: float | None, cache: dict | None, api: dict[str, object]) -> None:
    template = api["_sphere_template"](sphere.radius, sphere.perturbation, settings.sphere_segments, settings.sphere_rings)
    material = sphere.material
    if sphere.rotation.x == 0.0 and sphere.rotation.y == 0.0 and sphere.rotation.z == 0.0:
        for a, b, c, uv_a, uv_b, uv_c, _normal_a, _normal_b, _normal_c in template:
            color = _fast_material_color_data(material, uv_a, uv_b, uv_c, settings, cache)
            _append_triangle_vertices(
                payload,
                projector,
                sphere.center.x + a.x,
                sphere.center.y + a.y,
                sphere.center.z + a.z,
                sphere.center.x + b.x,
                sphere.center.y + b.y,
                sphere.center.z + b.z,
                sphere.center.x + c.x,
                sphere.center.y + c.y,
                sphere.center.z + c.z,
                color,
                max_distance_squared,
            )
        return

    rotation = api["_rotation_terms"](sphere.rotation)
    rotate = api["_rotate_euler_precomputed"]
    for a, b, c, uv_a, uv_b, uv_c, _normal_a, _normal_b, _normal_c in template:
        ra = rotate(a, rotation)
        rb = rotate(b, rotation)
        rc = rotate(c, rotation)
        color = _fast_material_color_data(material, uv_a, uv_b, uv_c, settings, cache)
        _append_triangle_vertices(
            payload,
            projector,
            sphere.center.x + ra.x,
            sphere.center.y + ra.y,
            sphere.center.z + ra.z,
            sphere.center.x + rb.x,
            sphere.center.y + rb.y,
            sphere.center.z + rb.z,
            sphere.center.x + rc.x,
            sphere.center.y + rc.y,
            sphere.center.z + rc.z,
            color,
            max_distance_squared,
        )


def _append_bowl_vertices(payload, bowl, settings, projector: _NDCProjector, max_distance_squared: float | None, cache: dict | None, api: dict[str, object]) -> None:
    template = api["_bowl_template"](bowl.radius, bowl.depth, bowl.perturbation, bowl.thickness, settings.sphere_segments, settings.sphere_rings)
    material = bowl.material
    for a, b, c, uv_a, uv_b, uv_c, _normal_a, _normal_b, _normal_c in template:
        color = _fast_material_color_data(material, uv_a, uv_b, uv_c, settings, cache)
        _append_triangle_vertices(
            payload,
            projector,
            bowl.center.x + a.x,
            bowl.center.y + a.y,
            bowl.center.z + a.z,
            bowl.center.x + b.x,
            bowl.center.y + b.y,
            bowl.center.z + b.z,
            bowl.center.x + c.x,
            bowl.center.y + c.y,
            bowl.center.z + c.z,
            color,
            max_distance_squared,
        )


def _append_triangle_vertices(
    payload,
    projector: _NDCProjector,
    ax: float,
    ay: float,
    az: float,
    bx: float,
    by: float,
    bz: float,
    cx: float,
    cy: float,
    cz: float,
    color: tuple[int, int, int],
    max_distance_squared: float | None,
) -> None:
    if max_distance_squared is not None:
        center_x = (ax + bx + cx) / 3.0
        center_y = (ay + by + cy) / 3.0
        center_z = (az + bz + cz) / 3.0
        dx = center_x - projector.camera_x
        dy = center_y - projector.camera_y
        dz = center_z - projector.camera_z
        if dx * dx + dy * dy + dz * dz > max_distance_squared:
            return
    projected_a = projector.project_xyz(ax, ay, az)
    projected_b = projector.project_xyz(bx, by, bz)
    projected_c = projector.project_xyz(cx, cy, cz)
    if projected_a is None or projected_b is None or projected_c is None:
        return
    red = color[0] / 255.0
    green = color[1] / 255.0
    blue = color[2] / 255.0
    payload.extend((projected_a[0], projected_a[1], projected_a[2], red, green, blue))
    payload.extend((projected_b[0], projected_b[1], projected_b[2], red, green, blue))
    payload.extend((projected_c[0], projected_c[1], projected_c[2], red, green, blue))


def _append_screen_triangle_vertices(payload, triangle: ScreenTriangle, width: int, height: int) -> None:
    x_scale = 2.0 / max(1.0, width - 1)
    y_scale = 2.0 / max(1.0, height - 1)
    red = triangle.color[0] / 255.0
    green = triangle.color[1] / 255.0
    blue = triangle.color[2] / 255.0
    for vertex in (triangle.a, triangle.b, triangle.c):
        payload.extend((vertex.x * x_scale - 1.0, 1.0 - vertex.y * y_scale, -0.5, red, green, blue))


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


def _culled_by_facing(center, normal, camera) -> bool:
    view_direction = (camera.position - center).normalized()
    return normal.dot(view_direction) <= 0.0


def _lit_material_color(scene, triangle, camera, settings, center, normal) -> tuple[int, int, int]:
    material = triangle.material
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


def _fast_material_color(triangle, settings, cache: dict | None = None) -> tuple[int, int, int]:
    uv_a = uv_b = uv_c = None
    if triangle.has_texture_coordinates():
        uv_a, uv_b, uv_c = triangle.uv_a, triangle.uv_b, triangle.uv_c
    return _fast_material_color_data(triangle.material, uv_a, uv_b, uv_c, settings, cache)


def _fast_material_color_data(material, uv_a, uv_b, uv_c, settings, cache: dict | None = None) -> tuple[int, int, int]:
    gamma = getattr(settings, "gamma", 1.0)
    ambient = max(0.0, min(1.0, float(getattr(settings, "ambient", 0.0))))
    if getattr(material, "texture", None) is not None and uv_a is not None and uv_b is not None and uv_c is not None:
        u = (uv_a[0] + uv_b[0] + uv_c[0]) / 3.0
        v = (uv_a[1] + uv_b[1] + uv_c[1]) / 3.0
        if cache is not None:
            key = (id(material), id(material.texture), round(u, 6), round(v, 6), gamma, ambient)
            cached = cache.get(key)
            if cached is not None:
                return cached
        base = material.color_at((u, v))
    else:
        if cache is not None:
            key = (id(material), None, gamma, ambient)
            cached = cache.get(key)
            if cached is not None:
                return cached
        base = material.color
    rgb = list(_color_to_rgb(base))
    emission = getattr(material, "emission", None)
    if emission is not None:
        ergb = _color_to_rgb(emission)
        rgb = [min(255, rgb[index] + ergb[index]) for index in range(3)]
    lift = 0.55 + ambient * 0.45
    color = tuple(max(0, min(255, int(channel * lift))) for channel in rgb)
    result = _apply_gamma(color, gamma)
    if cache is not None:
        cache[key] = result
    return result


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

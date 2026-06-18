"""Render a lit, textured 3D shape preview through ModernGL."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from math import cos, pi, radians, sin, tau, tan
from pathlib import Path
import shutil
import subprocess

import numpy as np

from py_gpu import FrameBuffer


OUTPUT_DIR = Path("renderings-tests")

VERTEX_SHADER = """
#version 330
in vec3 in_position;
in vec3 in_normal;
in vec2 in_uv;

uniform mat4 u_mvp;
uniform mat4 u_model;
uniform mat3 u_normal_matrix;

out vec3 v_world;
out vec3 v_normal;
out vec2 v_uv;

void main() {
    vec4 world = u_model * vec4(in_position, 1.0);
    v_world = world.xyz;
    v_normal = normalize(u_normal_matrix * in_normal);
    v_uv = in_uv;
    gl_Position = u_mvp * vec4(in_position, 1.0);
}
"""

FRAGMENT_SHADER = """
#version 330
uniform sampler2D u_texture;
uniform vec3 u_base_color;
uniform vec3 u_camera;
uniform vec3 u_light_positions[3];
uniform vec3 u_light_colors[3];
uniform float u_light_intensities[3];
uniform float u_diffuse;
uniform float u_specular;
uniform float u_shininess;
uniform float u_reflectivity;
uniform float u_use_texture;

in vec3 v_world;
in vec3 v_normal;
in vec2 v_uv;
out vec4 frag_color;

void main() {
    vec3 normal = normalize(v_normal);
    vec3 view_dir = normalize(u_camera - v_world);
    vec3 base = mix(u_base_color, texture(u_texture, v_uv).rgb, u_use_texture);
    vec3 color = base * 0.08;

    for (int i = 0; i < 3; i++) {
        vec3 light_vec = u_light_positions[i] - v_world;
        float distance_sq = max(dot(light_vec, light_vec), 0.04);
        vec3 light_dir = normalize(light_vec);
        float attenuation = u_light_intensities[i] / (1.0 + distance_sq * 0.18);
        float diffuse = max(dot(normal, light_dir), 0.0) * u_diffuse;
        vec3 half_dir = normalize(light_dir + view_dir);
        float highlight = pow(max(dot(normal, half_dir), 0.0), u_shininess) * u_specular;
        color += base * u_light_colors[i] * diffuse * attenuation;
        color += u_light_colors[i] * highlight * attenuation;
    }

    vec3 rim = vec3(pow(max(0.0, 1.0 - dot(normal, view_dir)), 3.0)) * u_reflectivity;
    frag_color = vec4(clamp(color + rim, 0.0, 1.0), 1.0);
}
"""


@dataclass
class MeshHandle:
    vao: object
    vertex_buffer: object
    index_buffer: object
    index_count: int


class LitShapeRenderer:
    def __init__(self, width: int, height: int) -> None:
        import moderngl

        self.moderngl = moderngl
        self.width = width
        self.height = height
        self.ctx = moderngl.create_standalone_context()
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.program = self.ctx.program(vertex_shader=VERTEX_SHADER, fragment_shader=FRAGMENT_SHADER)
        self.texture = self._make_texture()
        self.frame_texture = self.ctx.texture((width, height), 3)
        self.depth = self.ctx.depth_renderbuffer((width, height))
        self.framebuffer = self.ctx.framebuffer(color_attachments=[self.frame_texture], depth_attachment=self.depth)
        self.sphere = self._make_mesh(*sphere_mesh(42, 22))
        self.cube = self._make_mesh(*cube_mesh())
        self.floor = self._make_mesh(*floor_mesh())

    def _make_texture(self):
        size = 96
        pixels = bytearray(size * size * 3)
        for y in range(size):
            for x in range(size):
                cell = ((x // 8) + (y // 8)) % 2
                stripe = 0.5 + 0.5 * sin((x + y) * 0.18)
                if cell:
                    color = (46, int(130 + stripe * 70), 218)
                else:
                    color = (238, int(238 - stripe * 70), 74)
                offset = (y * size + x) * 3
                pixels[offset : offset + 3] = bytes(color)
        texture = self.ctx.texture((size, size), 3, bytes(pixels))
        texture.build_mipmaps()
        texture.filter = (self.moderngl.LINEAR_MIPMAP_LINEAR, self.moderngl.LINEAR)
        return texture

    def _make_mesh(self, vertices: np.ndarray, indices: np.ndarray) -> MeshHandle:
        vbo = self.ctx.buffer(vertices.astype("f4").tobytes())
        ibo = self.ctx.buffer(indices.astype("i4").tobytes())
        vao = self.ctx.vertex_array(self.program, [(vbo, "3f 3f 2f", "in_position", "in_normal", "in_uv")], ibo)
        return MeshHandle(vao, vbo, ibo, int(indices.size))

    def render(self, time: float) -> FrameBuffer:
        self.framebuffer.use()
        self.framebuffer.clear(0.02, 0.025, 0.04, 1.0, depth=1.0)
        aspect = self.width / self.height
        camera = np.array((3.1 * sin(time * 0.45), 1.55, 4.2 * cos(time * 0.45)), dtype=np.float32)
        view = look_at(camera, np.array((0.0, 0.2, 0.0), dtype=np.float32), np.array((0.0, 1.0, 0.0), dtype=np.float32))
        projection = perspective(radians(46.0), aspect, 0.05, 50.0)
        self.texture.use(0)
        self.program["u_texture"] = 0
        self.program["u_camera"].value = tuple(float(value) for value in camera)
        self.program["u_light_positions"].value = (
            (-2.3, 2.6, 2.1),
            (2.6, 1.6, 1.6),
            (0.0, 3.0, -2.4),
        )
        self.program["u_light_colors"].value = (
            (1.0, 0.9, 0.72),
            (0.42, 0.65, 1.0),
            (1.0, 0.36, 0.48),
        )
        self.program["u_light_intensities"].value = (4.4, 2.8, 1.8)

        sphere_model = translate(-0.65, 0.35, 0.0) @ rotate_y(time * 1.15) @ rotate_x(time * 0.35)
        self._draw(self.sphere, projection, view, sphere_model, (1.0, 1.0, 1.0), diffuse=0.95, specular=0.32, shininess=40.0, reflectivity=0.12, use_texture=1.0)

        cube_model = translate(0.9, 0.35, -0.05) @ rotate_y(-time * 0.85) @ rotate_x(0.55) @ scale(0.62, 0.62, 0.62)
        self._draw(self.cube, projection, view, cube_model, (0.86, 0.88, 0.92), diffuse=0.65, specular=0.9, shininess=76.0, reflectivity=0.35, use_texture=0.0)

        floor_model = np.eye(4, dtype=np.float32)
        self._draw(self.floor, projection, view, floor_model, (0.34, 0.28, 0.2), diffuse=0.82, specular=0.08, shininess=12.0, reflectivity=0.02, use_texture=0.0)

        data = self.framebuffer.read(components=3, alignment=1)
        return FrameBuffer(self.width, self.height, bytearray(flip_rows(data, self.width, self.height)))

    def _draw(
        self,
        mesh: MeshHandle,
        projection: np.ndarray,
        view: np.ndarray,
        model: np.ndarray,
        base_color: tuple[float, float, float],
        *,
        diffuse: float,
        specular: float,
        shininess: float,
        reflectivity: float,
        use_texture: float,
    ) -> None:
        mvp = projection @ view @ model
        normal_matrix = np.linalg.inv(model[:3, :3]).T.astype(np.float32)
        self.program["u_mvp"].write(mvp.T.astype("f4").tobytes())
        self.program["u_model"].write(model.T.astype("f4").tobytes())
        self.program["u_normal_matrix"].write(normal_matrix.T.astype("f4").tobytes())
        self.program["u_base_color"].value = base_color
        self.program["u_diffuse"].value = diffuse
        self.program["u_specular"].value = specular
        self.program["u_shininess"].value = shininess
        self.program["u_reflectivity"].value = reflectivity
        self.program["u_use_texture"].value = use_texture
        mesh.vao.render()


def sphere_mesh(segments: int, rings: int) -> tuple[np.ndarray, np.ndarray]:
    vertices = []
    indices = []
    for ring in range(rings + 1):
        phi = pi * ring / rings
        for segment in range(segments + 1):
            theta = tau * segment / segments
            normal = np.array((sin(phi) * cos(theta), cos(phi), sin(phi) * sin(theta)), dtype=np.float32)
            vertices.extend((*normal, *normal, segment / segments, ring / rings))
    stride = segments + 1
    for ring in range(rings):
        for segment in range(segments):
            a = ring * stride + segment
            b = a + stride
            indices.extend((a, b, a + 1, a + 1, b, b + 1))
    return np.array(vertices, dtype=np.float32), np.array(indices, dtype=np.int32)


def cube_mesh() -> tuple[np.ndarray, np.ndarray]:
    faces = [
        ((0, 0, 1), [(-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)]),
        ((0, 0, -1), [(1, -1, -1), (-1, -1, -1), (-1, 1, -1), (1, 1, -1)]),
        ((1, 0, 0), [(1, -1, 1), (1, -1, -1), (1, 1, -1), (1, 1, 1)]),
        ((-1, 0, 0), [(-1, -1, -1), (-1, -1, 1), (-1, 1, 1), (-1, 1, -1)]),
        ((0, 1, 0), [(-1, 1, 1), (1, 1, 1), (1, 1, -1), (-1, 1, -1)]),
        ((0, -1, 0), [(-1, -1, -1), (1, -1, -1), (1, -1, 1), (-1, -1, 1)]),
    ]
    uvs = ((0, 0), (1, 0), (1, 1), (0, 1))
    vertices = []
    indices = []
    for face_index, (normal, points) in enumerate(faces):
        base = face_index * 4
        for point, uv in zip(points, uvs):
            vertices.extend((*point, *normal, *uv))
        indices.extend((base, base + 1, base + 2, base, base + 2, base + 3))
    return np.array(vertices, dtype=np.float32), np.array(indices, dtype=np.int32)


def floor_mesh() -> tuple[np.ndarray, np.ndarray]:
    vertices = np.array(
        [
            -3.0, -0.32, -2.2, 0.0, 1.0, 0.0, 0.0, 0.0,
            3.0, -0.32, -2.2, 0.0, 1.0, 0.0, 1.0, 0.0,
            3.0, -0.32, 2.2, 0.0, 1.0, 0.0, 1.0, 1.0,
            -3.0, -0.32, 2.2, 0.0, 1.0, 0.0, 0.0, 1.0,
        ],
        dtype=np.float32,
    )
    return vertices, np.array((0, 1, 2, 0, 2, 3), dtype=np.int32)


def perspective(fov_radians: float, aspect: float, near: float, far: float) -> np.ndarray:
    f = 1.0 / tan(fov_radians * 0.5)
    matrix = np.zeros((4, 4), dtype=np.float32)
    matrix[0, 0] = f / aspect
    matrix[1, 1] = f
    matrix[2, 2] = (far + near) / (near - far)
    matrix[2, 3] = (2.0 * far * near) / (near - far)
    matrix[3, 2] = -1.0
    return matrix


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    forward = target - eye
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    true_up = np.cross(right, forward)
    matrix = np.eye(4, dtype=np.float32)
    matrix[0, :3] = right
    matrix[1, :3] = true_up
    matrix[2, :3] = -forward
    matrix[0, 3] = -np.dot(right, eye)
    matrix[1, 3] = -np.dot(true_up, eye)
    matrix[2, 3] = np.dot(forward, eye)
    return matrix


def translate(x: float, y: float, z: float) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, 3] = (x, y, z)
    return matrix


def scale(x: float, y: float, z: float) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float32)
    matrix[0, 0] = x
    matrix[1, 1] = y
    matrix[2, 2] = z
    return matrix


def rotate_x(angle: float) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float32)
    c, s = cos(angle), sin(angle)
    matrix[1, 1] = c
    matrix[1, 2] = -s
    matrix[2, 1] = s
    matrix[2, 2] = c
    return matrix


def rotate_y(angle: float) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float32)
    c, s = cos(angle), sin(angle)
    matrix[0, 0] = c
    matrix[0, 2] = s
    matrix[2, 0] = -s
    matrix[2, 2] = c
    return matrix


def flip_rows(data: bytes, width: int, height: int) -> bytes:
    row_size = width * 3
    payload = bytearray(len(data))
    for y in range(height):
        src_start = (height - 1 - y) * row_size
        dst_start = y * row_size
        payload[dst_start : dst_start + row_size] = data[src_start : src_start + row_size]
    return bytes(payload)


def find_ffmpeg() -> str | None:
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def render_image(args: argparse.Namespace) -> Path:
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    renderer = LitShapeRenderer(args.width, args.height)
    frame = renderer.render(0.35)
    frame.to_png(output)
    print(f"Wrote {output}")
    return output


def render_video(args: argparse.Namespace) -> Path:
    output = Path(args.video)
    output.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = find_ffmpeg()
    if ffmpeg is None:
        raise RuntimeError("ffmpeg executable not found")
    renderer = LitShapeRenderer(args.width, args.height)
    command = [
        ffmpeg,
        "-y",
        "-loglevel",
        "warning",
        "-f",
        "image2pipe",
        "-framerate",
        str(args.fps),
        "-vcodec",
        "ppm",
        "-i",
        "-",
        "-vf",
        "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-an",
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(output),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    if process.stdin is None:
        raise RuntimeError("could not open ffmpeg stdin")
    try:
        for frame_index in range(args.frames):
            frame = renderer.render(frame_index / max(1, args.fps))
            process.stdin.write(frame.to_ppm_bytes())
    finally:
        process.stdin.close()
    result = process.wait()
    if result != 0:
        raise RuntimeError(f"ffmpeg failed with exit code {result}")
    print(f"Wrote {output}")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render lit textured 3D GPU preview media.")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "gpu_lit_textured_shapes.png")
    parser.add_argument("--video", type=Path, default=OUTPUT_DIR / "gpu_lit_textured_shapes.mp4")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--frames", type=int, default=96)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--image-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    render_image(args)
    if not args.image_only:
        render_video(args)


if __name__ == "__main__":
    main()

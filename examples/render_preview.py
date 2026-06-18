"""Render py_gpu preview image and video artifacts."""

from __future__ import annotations

import argparse
from math import cos, sin, tau
from pathlib import Path
import shutil
import subprocess

from py_gpu import RasterBatch, ScreenTriangle, ScreenVertex, select_backend


OUTPUT_DIR = Path("renderings-tests")


def make_preview_batch(width: int, height: int, time: float = 0.0) -> RasterBatch:
    triangles: list[ScreenTriangle] = []
    center_x = width * 0.5
    center_y = height * 0.52
    ring = []
    for index in range(9):
        angle = tau * index / 9.0 + time * 0.8
        radius = min(width, height) * (0.26 + 0.04 * sin(time * 1.7 + index))
        ring.append(
            ScreenVertex(
                center_x + cos(angle) * radius,
                center_y + sin(angle) * radius,
                1.2 + 0.35 * sin(angle + time),
            )
        )
    hub = ScreenVertex(center_x, center_y, 0.7)
    colors = (
        (255, 88, 76),
        (255, 186, 76),
        (246, 240, 90),
        (86, 220, 120),
        (78, 210, 230),
        (94, 138, 255),
        (190, 105, 255),
        (255, 105, 188),
        (230, 230, 238),
    )
    for index, vertex in enumerate(ring):
        triangles.append(ScreenTriangle(hub, vertex, ring[(index + 1) % len(ring)], colors[index % len(colors)]))

    floor_y = height * 0.82
    triangles.extend(
        [
            ScreenTriangle(
                ScreenVertex(width * 0.12, floor_y, 2.2),
                ScreenVertex(width * 0.88, floor_y, 2.2),
                ScreenVertex(width * 0.76, height * 0.95, 2.4),
                (42, 78, 92),
            ),
            ScreenTriangle(
                ScreenVertex(width * 0.12, floor_y, 2.2),
                ScreenVertex(width * 0.76, height * 0.95, 2.4),
                ScreenVertex(width * 0.24, height * 0.95, 2.4),
                (36, 58, 70),
            ),
        ]
    )
    return RasterBatch(triangles)


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
    backend = select_backend(args.backend)
    frame = backend.render(make_preview_batch(args.width, args.height, 0.2), args.width, args.height, background=(7, 9, 14))
    frame.to_png(output)
    print(f"Wrote {output} with {backend.capabilities.name}")
    return output


def render_video(args: argparse.Namespace) -> Path:
    output = Path(args.video)
    output.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = find_ffmpeg()
    if ffmpeg is None:
        raise RuntimeError("ffmpeg executable not found")
    backend = select_backend(args.backend)
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
            time = frame_index / max(1, args.fps)
            frame = backend.render(make_preview_batch(args.width, args.height, time), args.width, args.height, background=(7, 9, 14))
            process.stdin.write(frame.to_ppm_bytes())
    finally:
        process.stdin.close()
    result = process.wait()
    if result != 0:
        raise RuntimeError(f"ffmpeg failed with exit code {result}")
    print(f"Wrote {output} with {backend.capabilities.name}")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render py_gpu preview artifacts.")
    parser.add_argument("--backend", choices=("auto", "gpu", "moderngl", "numpy", "cpu"), default="auto")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "py_gpu_preview.png")
    parser.add_argument("--video", type=Path, default=OUTPUT_DIR / "py_gpu_preview.mp4")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--frames", type=int, default=72)
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

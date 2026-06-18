"""Benchmark the py_gpu screen-space raster backend."""

from __future__ import annotations

import argparse
from random import Random
from statistics import mean
from time import perf_counter

from py_gpu import RasterBatch, ScreenTriangle, ScreenVertex, select_backend


def make_batch(width: int, height: int, triangles: int, seed: int) -> RasterBatch:
    rng = Random(seed)
    items = []
    for _ in range(triangles):
        center_x = rng.uniform(0.0, width)
        center_y = rng.uniform(0.0, height)
        size = rng.uniform(8.0, 48.0)
        z = rng.uniform(0.2, 8.0)
        color = (
            rng.randrange(40, 255),
            rng.randrange(40, 255),
            rng.randrange(40, 255),
        )
        items.append(
            ScreenTriangle(
                ScreenVertex(center_x, center_y - size, z),
                ScreenVertex(center_x - size, center_y + size, z + rng.uniform(-0.1, 0.1)),
                ScreenVertex(center_x + size, center_y + size, z + rng.uniform(-0.1, 0.1)),
                color,
            )
        )
    return RasterBatch(items)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark py_gpu's current raster backend.")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--triangles", type=int, default=1000)
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--backend", choices=("auto", "cpu", "numpy"), default="auto")
    parser.add_argument("--output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    backend = select_backend(args.backend)
    batch = make_batch(args.width, args.height, args.triangles, args.seed)
    timings = []
    frame = None
    for _ in range(args.frames):
        start = perf_counter()
        frame = backend.render(batch, args.width, args.height, background=(6, 8, 12))
        timings.append(perf_counter() - start)
    average = mean(timings)
    print(f"backend: {backend.capabilities.name}")
    print(f"accelerated: {backend.capabilities.accelerated}")
    print(f"triangles: {args.triangles}")
    print(f"size: {args.width}x{args.height}")
    print(f"avg frame: {average * 1000:.2f} ms")
    print(f"approx fps: {1.0 / average:.1f}")
    if args.output and frame is not None:
        frame.to_png(args.output)
        print(f"wrote: {args.output}")


if __name__ == "__main__":
    main()

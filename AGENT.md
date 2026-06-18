# Agent Guide

`py_gpu` is the renderer-core companion to scene packages such as `py_3d`.
Keep it focused on batching, backend selection, buffers, and acceleration
experiments.

## Rules

- Do not move scene, physics, or gameplay concepts into this package.
- Keep external scene support in `py_gpu.adapters`.
- Keep backend dependencies optional. The default test path must work without
  GPU packages installed.
- Keep the CPU raster backend as a correctness fallback.
- Add capability flags when a backend supports only part of the renderer
  surface.
- Benchmark performance claims with scripts under `examples/` or a future
  `benchmarks/` directory.
- Prefer persistent batches and explicit transfer points over hidden global
  renderer state.
- Keep APIs small enough that other Python libraries can adapt into them.

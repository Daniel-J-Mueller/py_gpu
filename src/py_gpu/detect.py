"""Optional backend discovery."""

from __future__ import annotations

from importlib.util import find_spec


def detect_gpu_packages() -> tuple[str, ...]:
    """Return optional GPU or acceleration packages visible to Python."""

    candidates = (
        ("wgpu", "WebGPU/wgpu"),
        ("moderngl", "OpenGL/ModernGL"),
        ("OpenGL", "OpenGL/PyOpenGL"),
        ("vulkan", "Vulkan"),
        ("cupy", "CUDA/CuPy"),
        ("numba", "CPU/GPU Numba"),
        ("torch", "PyTorch"),
    )
    return tuple(label for module_name, label in candidates if find_spec(module_name) is not None)

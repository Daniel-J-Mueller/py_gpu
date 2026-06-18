"""Backend-agnostic rendering foundations for Python scene packages."""

from .backends import BackendCapabilities, RasterBackend, select_backend
from .buffers import DepthBuffer, FrameBuffer
from .detect import detect_gpu_packages
from .geometry import RasterBatch, ScreenTriangle, ScreenVertex
from .raster import CPURasterBackend

try:
    from .moderngl_backend import ModernGLRasterBackend
except Exception:  # pragma: no cover - optional dependency
    ModernGLRasterBackend = None  # type: ignore[assignment]

try:
    from .numpy_raster import NumpyRasterBackend
except Exception:  # pragma: no cover - optional dependency
    NumpyRasterBackend = None  # type: ignore[assignment]

__all__ = [
    "BackendCapabilities",
    "CPURasterBackend",
    "DepthBuffer",
    "FrameBuffer",
    "ModernGLRasterBackend",
    "NumpyRasterBackend",
    "RasterBackend",
    "RasterBatch",
    "ScreenTriangle",
    "ScreenVertex",
    "detect_gpu_packages",
    "select_backend",
]

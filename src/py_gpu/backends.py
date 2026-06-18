"""Backend selection contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from .buffers import ColorLike, FrameBuffer
from .geometry import RasterBatch


@dataclass(frozen=True)
class BackendCapabilities:
    name: str
    accelerated: bool
    supports_depth: bool = True
    supports_textures: bool = False
    supports_vertex_attributes: bool = False


@runtime_checkable
class RasterBackend(Protocol):
    """Protocol for projected triangle raster backends."""

    capabilities: BackendCapabilities

    def render(
        self,
        batch: RasterBatch,
        width: int,
        height: int,
        *,
        background: ColorLike = (0, 0, 0),
        target: FrameBuffer | None = None,
    ) -> FrameBuffer:
        ...


def select_backend(prefer: str = "auto") -> RasterBackend:
    """Select the best currently available raster backend.

    The first committed implementation is the dependency-free CPU backend.
    GPU backends should keep this function's return contract stable.
    """

    if prefer in {"auto", "numpy"}:
        try:
            from .numpy_raster import NumpyRasterBackend

            return NumpyRasterBackend()
        except Exception as exc:
            if prefer == "numpy":
                raise RuntimeError("NumPy raster backend is not available") from exc

    from .raster import CPURasterBackend

    if prefer not in {"auto", "cpu"}:
        raise RuntimeError(f"backend preference is not implemented yet: {prefer}")
    return CPURasterBackend()

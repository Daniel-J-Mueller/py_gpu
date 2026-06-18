"""Dependency-free frame and depth buffers."""

from __future__ import annotations

from binascii import crc32
from dataclasses import dataclass, field
from pathlib import Path
from struct import pack
from zlib import compress

ColorLike = tuple[int, int, int]


def clamp_channel(value: int | float) -> int:
    return max(0, min(255, int(round(value))))


def as_rgb(value: ColorLike) -> ColorLike:
    if len(value) != 3:
        raise ValueError("RGB colors must have exactly three channels")
    return (clamp_channel(value[0]), clamp_channel(value[1]), clamp_channel(value[2]))


def _validate_dimensions(width: int, height: int) -> None:
    if width <= 0 or height <= 0:
        raise ValueError("buffer dimensions must be positive")


@dataclass
class FrameBuffer:
    """A row-major RGB byte buffer."""

    width: int
    height: int
    pixels: bytearray = field(repr=False)

    def __post_init__(self) -> None:
        _validate_dimensions(self.width, self.height)
        expected = self.width * self.height * 3
        if len(self.pixels) != expected:
            raise ValueError(f"frame buffer requires {expected} bytes")

    @classmethod
    def new(cls, width: int, height: int, fill: ColorLike = (0, 0, 0)) -> "FrameBuffer":
        color = as_rgb(fill)
        pixels = bytearray(width * height * 3)
        buffer = cls(width, height, pixels)
        buffer.clear(color)
        return buffer

    def clear(self, color: ColorLike = (0, 0, 0)) -> None:
        r, g, b = as_rgb(color)
        self.pixels[:] = bytes((r, g, b)) * (self.width * self.height)

    def index(self, x: int, y: int) -> int:
        if not (0 <= x < self.width and 0 <= y < self.height):
            raise IndexError(f"pixel coordinate out of bounds: {(x, y)}")
        return (y * self.width + x) * 3

    def set_pixel(self, x: int, y: int, color: ColorLike) -> None:
        if 0 <= x < self.width and 0 <= y < self.height:
            offset = (y * self.width + x) * 3
            r, g, b = as_rgb(color)
            self.pixels[offset] = r
            self.pixels[offset + 1] = g
            self.pixels[offset + 2] = b

    def get_pixel(self, x: int, y: int) -> ColorLike:
        offset = self.index(x, y)
        return (self.pixels[offset], self.pixels[offset + 1], self.pixels[offset + 2])

    def to_rgb_bytes(self) -> bytes:
        return bytes(self.pixels)

    def to_ppm_bytes(self) -> bytes:
        header = f"P6\n{self.width} {self.height}\n255\n".encode("ascii")
        return header + self.to_rgb_bytes()

    def to_ppm(self, path: str | Path) -> None:
        Path(path).write_bytes(self.to_ppm_bytes())

    def to_png(self, path: str | Path) -> None:
        rows = bytearray()
        row_stride = self.width * 3
        for y in range(self.height):
            rows.append(0)
            start = y * row_stride
            rows.extend(self.pixels[start : start + row_stride])
        payload = b"".join(
            [
                b"\x89PNG\r\n\x1a\n",
                _png_chunk(b"IHDR", pack(">IIBBBBB", self.width, self.height, 8, 2, 0, 0, 0)),
                _png_chunk(b"IDAT", compress(bytes(rows))),
                _png_chunk(b"IEND", b""),
            ]
        )
        Path(path).write_bytes(payload)


@dataclass
class DepthBuffer:
    """A row-major depth buffer where smaller values are closer."""

    width: int
    height: int
    values: list[float] = field(repr=False)

    def __post_init__(self) -> None:
        _validate_dimensions(self.width, self.height)
        expected = self.width * self.height
        if len(self.values) != expected:
            raise ValueError(f"depth buffer requires {expected} values")

    @classmethod
    def new(cls, width: int, height: int, fill: float = float("inf")) -> "DepthBuffer":
        return cls(width, height, [fill] * (width * height))

    def clear(self, value: float = float("inf")) -> None:
        self.values[:] = [value] * (self.width * self.height)


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    checksum = crc32(kind + data) & 0xFFFFFFFF
    return pack(">I", len(data)) + kind + data + pack(">I", checksum)

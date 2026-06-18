import pytest

from py_gpu import RasterBatch, ScreenTriangle, ScreenVertex


def test_moderngl_backend_draws_triangle_when_available():
    try:
        from py_gpu import ModernGLRasterBackend
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"ModernGL import unavailable: {exc}")
    if ModernGLRasterBackend is None:
        pytest.skip("ModernGL backend unavailable")

    backend = ModernGLRasterBackend()
    try:
        frame = backend.render(
            RasterBatch(
                [
                    ScreenTriangle(
                        ScreenVertex(12, 4, 1.0),
                        ScreenVertex(4, 28, 1.0),
                        ScreenVertex(28, 28, 1.0),
                        (220, 90, 40),
                    )
                ]
            ),
            32,
            32,
            background=(0, 0, 0),
        )
    except Exception as exc:
        pytest.skip(f"ModernGL context unavailable: {exc}")

    assert any(frame.pixels[index] == 220 for index in range(0, len(frame.pixels), 3))

from py_gpu import FrameBuffer, RasterBatch, ScreenTriangle, ScreenVertex, select_backend


def test_frame_buffer_sets_and_reads_pixels():
    frame = FrameBuffer.new(4, 3, (1, 2, 3))

    frame.set_pixel(2, 1, (20, 30, 40))

    assert frame.get_pixel(0, 0) == (1, 2, 3)
    assert frame.get_pixel(2, 1) == (20, 30, 40)


def test_cpu_raster_backend_draws_triangle():
    batch = RasterBatch(
        [
            ScreenTriangle(
                ScreenVertex(2, 0, 1.0),
                ScreenVertex(0, 4, 1.0),
                ScreenVertex(4, 4, 1.0),
                (200, 80, 40),
            )
        ]
    )

    frame = select_backend("cpu").render(batch, 5, 5)

    assert any(frame.pixels[index] == 200 for index in range(0, len(frame.pixels), 3))


def test_depth_keeps_nearest_triangle():
    far = ScreenTriangle(
        ScreenVertex(2, 0, 3.0),
        ScreenVertex(0, 4, 3.0),
        ScreenVertex(4, 4, 3.0),
        (0, 0, 255),
    )
    near = ScreenTriangle(
        ScreenVertex(2, 0, 1.0),
        ScreenVertex(0, 4, 1.0),
        ScreenVertex(4, 4, 1.0),
        (255, 0, 0),
    )

    frame = select_backend("cpu").render(RasterBatch([far, near]), 5, 5)

    assert frame.get_pixel(2, 2) == (255, 0, 0)

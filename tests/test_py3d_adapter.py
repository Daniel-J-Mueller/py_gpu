def test_py3d_adapter_defaults_to_reference_compatible_rendering():
    try:
        from py_3d import Camera, Material, RenderEngine, RenderSettings, Scene, Sphere, TextBulletin
        from py_gpu.adapters.py3d import Py3DRasterRenderer
    except Exception as exc:  # pragma: no cover
        import pytest

        pytest.skip(f"py_3d adapter dependencies unavailable: {exc}")

    scene = Scene()
    scene.add(Sphere((0, 0, 0), 0.6, Material(color=(80, 130, 210), roughness=0.5, fuzziness=0.2)))
    scene.add_bulletin(TextBulletin("PARITY", position=(0, 0), color=(255, 255, 255), background=(0, 0, 0)))
    camera = Camera(position=(0, 0, -4), target=(0, 0, 0))
    settings = RenderSettings(width=64, height=48, smooth_shading=True)

    reference = RenderEngine().render(scene, camera, settings)
    adapted = Py3DRasterRenderer().render(scene, camera, settings)

    assert adapted.pixels == reference.pixels


def test_py3d_adapter_fast_path_renders_with_cpu_backend():
    try:
        from py_3d import Camera, Lamp, Material, RenderSettings, Scene, Triangle
        from py_gpu import select_backend
        from py_gpu.adapters.py3d import Py3DRasterRenderer
    except Exception as exc:  # pragma: no cover
        import pytest

        pytest.skip(f"py_3d adapter dependencies unavailable: {exc}")

    scene = Scene()
    scene.add(Triangle((-0.8, -0.6, 0), (0.8, -0.6, 0), (0.0, 0.7, 0), Material(color=(220, 90, 40))))
    scene.add_light(Lamp(position=(0, 0, -2), intensity=4.0))
    camera = Camera(position=(0, 0, -4), target=(0, 0, 0))
    settings = RenderSettings(width=48, height=36, ambient=0.1)

    adapted = Py3DRasterRenderer(backend_impl=select_backend("cpu"), reference_compatible=False).render(scene, camera, settings)

    assert any(pixel.r > 0 for pixel in adapted.pixels)


def test_py3d_adapter_fast_path_expands_wireframe_edges():
    try:
        from py_3d import Camera, Material, RenderSettings, Scene, Triangle
        from py_gpu.adapters.py3d import scene_to_raster_batch
    except Exception as exc:  # pragma: no cover
        import pytest

        pytest.skip(f"py_3d adapter dependencies unavailable: {exc}")

    scene = Scene()
    scene.add(Triangle((-0.8, -0.6, 0), (0.8, -0.6, 0), (0.0, 0.7, 0), Material(color=(220, 90, 40), emission=(10, 10, 10))))
    camera = Camera(position=(0, 0, -4), target=(0, 0, 0))
    settings = RenderSettings(width=48, height=36, wireframe=True)

    batch = scene_to_raster_batch(scene, camera, settings)

    assert len(batch.triangles) == 6


def test_py3d_adapter_fast_path_renders_line_primitives():
    try:
        from py_3d import Camera, Line3, Material, RenderSettings, Scene
        from py_gpu.adapters.py3d import scene_to_raster_batch
    except Exception as exc:  # pragma: no cover
        import pytest

        pytest.skip(f"py_3d adapter dependencies unavailable: {exc}")

    scene = Scene()
    scene.add(Line3((-0.4, 0.0, 0.0), (0.4, 0.0, 0.0), Material(color=(40, 220, 255))))
    camera = Camera(position=(0, 0, -4), target=(0, 0, 0))
    settings = RenderSettings(width=48, height=36)

    batch = scene_to_raster_batch(scene, camera, settings)

    assert len(batch.triangles) == 2

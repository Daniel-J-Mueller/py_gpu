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

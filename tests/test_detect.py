from py_gpu import detect_gpu_packages


def test_detect_gpu_packages_returns_tuple():
    assert isinstance(detect_gpu_packages(), tuple)

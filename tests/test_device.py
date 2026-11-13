"""Device selection and pipeline caching (torch and Kokoro are faked)."""

from types import SimpleNamespace

import pytest

from bts import device, errors
from bts.settings import use_settings


def fake_torch(available=True, free_mib=4096, total_mib=8192, mem_error=None):
    def mem_get_info():
        if mem_error:
            raise mem_error
        return free_mib * 2**20, total_mib * 2**20

    class OOM(RuntimeError):
        pass

    cuda = SimpleNamespace(
        is_available=lambda: available,
        mem_get_info=mem_get_info,
        OutOfMemoryError=OOM,
        empty_cache=lambda: None,
    )
    return SimpleNamespace(cuda=cuda)


def patch_torch(monkeypatch, **kwargs):
    torch = fake_torch(**kwargs)
    monkeypatch.setattr(device, "get_torch", lambda: torch)
    return torch


def test_cpu_never_touches_torch(monkeypatch):
    def boom():
        raise AssertionError("torch must not be imported")

    monkeypatch.setattr(device, "get_torch", boom)
    assert device.resolve_device("cpu") == "cpu"


def test_auto_uses_gpu_with_room(monkeypatch):
    patch_torch(monkeypatch)
    assert device.resolve_device("auto") == "cuda"
    assert device.resolve_device("cuda") == "cuda"


def test_auto_falls_back_without_gpu(monkeypatch):
    patch_torch(monkeypatch, available=False)
    assert device.resolve_device("auto") == "cpu"


def test_cuda_without_gpu_is_error(monkeypatch):
    patch_torch(monkeypatch, available=False)
    with pytest.raises(errors.TTSError, match="no usable GPU"):
        device.resolve_device("cuda")


def test_gpu_full(monkeypatch, capsys):
    patch_torch(monkeypatch, free_mib=100)
    assert device.resolve_device("auto") == "cpu"
    assert "Falling back" in capsys.readouterr().err
    with pytest.raises(errors.GpuMemoryError, match="Only 100 MiB"):
        device.resolve_device("cuda")


def test_gpu_has_room_unknown_error_assumes_usable(monkeypatch):
    patch_torch(monkeypatch, mem_error=ValueError("weird"))
    assert device.gpu_has_room() == (True, "")


def test_gpu_has_room_oom_while_measuring(monkeypatch):
    torch = patch_torch(monkeypatch)
    torch.cuda.mem_get_info = lambda: (_ for _ in ()).throw(
        torch.cuda.OutOfMemoryError("out of memory")
    )
    has_room, message = device.gpu_has_room()
    assert not has_room and "not have enough free memory" in message


def test_is_cuda_oom(monkeypatch):
    torch = patch_torch(monkeypatch)
    assert device.is_cuda_oom(torch.cuda.OutOfMemoryError())
    assert device.is_cuda_oom(RuntimeError("CUDA out of memory"))
    assert not device.is_cuda_oom(RuntimeError("other"))
    assert not device.is_cuda_oom(ValueError("out of memory"))


def test_gpu_oom_message_mentions_numbers():
    text = device.gpu_oom_message(10, 20)
    assert "10 MiB" in text and "20 MiB" in text
    assert "--device cpu" in device.gpu_oom_message()


def test_get_pipeline_is_cached(monkeypatch):
    created = []

    class Pipeline:
        def __init__(self, **kwargs):
            created.append(kwargs)

    monkeypatch.setattr(device, "get_kpipeline_class", lambda: Pipeline)
    device.get_pipeline.cache_clear()
    first = device.get_pipeline("e", "cpu")
    assert device.get_pipeline("e", "cpu") is first
    assert len(created) == 1
    assert created[0]["lang_code"] == "e"
    device.get_pipeline.cache_clear()


def test_configured_repo_is_part_of_both_pipeline_cache_keys(monkeypatch):
    created = []

    class Pipeline:
        def __init__(self, **kwargs):
            created.append(kwargs)

    monkeypatch.setattr(device, "get_kpipeline_class", lambda: Pipeline)
    device.get_pipeline.cache_clear()
    device.get_g2p_pipeline.cache_clear()
    try:
        with use_settings({"repo_id": "first/model"}):
            first = device.get_pipeline("e", "cpu")
            first_g2p = device.get_g2p_pipeline("e")
        with use_settings({"repo_id": "second/model"}):
            assert device.get_pipeline("e", "cpu") is not first
            assert device.get_g2p_pipeline("e") is not first_g2p
        assert [item["repo_id"] for item in created] == [
            "first/model", "first/model", "second/model", "second/model"
        ]
    finally:
        device.get_pipeline.cache_clear()
        device.get_g2p_pipeline.cache_clear()


def test_configured_gpu_memory_threshold(monkeypatch):
    patch_torch(monkeypatch, free_mib=500)
    with use_settings({"gpu_min_free_mib": 400}):
        assert device.resolve_device("auto") == "cuda"
    with use_settings({"gpu_min_free_mib": 600}):
        assert device.gpu_has_room()[0] is False
        assert "600 MiB" in device.gpu_has_room()[1]


def test_get_pipeline_translates_oom(monkeypatch):
    patch_torch(monkeypatch)

    class Pipeline:
        def __init__(self, **kwargs):
            raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(device, "get_kpipeline_class", lambda: Pipeline)
    device.get_pipeline.cache_clear()
    with pytest.raises(errors.GpuMemoryError):
        device.get_pipeline("e", "cuda")
    device.get_pipeline.cache_clear()


def test_get_g2p_pipeline_wraps_errors(monkeypatch):
    class Pipeline:
        def __init__(self, **kwargs):
            raise OSError("no network")

    monkeypatch.setattr(device, "get_kpipeline_class", lambda: Pipeline)
    device.get_g2p_pipeline.cache_clear()
    with pytest.raises(errors.TTSError, match="phonemizer"):
        device.get_g2p_pipeline("e")
    device.get_g2p_pipeline.cache_clear()

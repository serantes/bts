"""Device selection (CPU / GPU) and loading of the Kokoro pipelines.

``torch`` and ``kokoro`` are heavy, so they are imported lazily."""

from __future__ import annotations

import functools
import sys
import warnings
from typing import TYPE_CHECKING

from bts.errors import GpuMemoryError, TTSError
from bts.settings import current_settings

if TYPE_CHECKING:
    from kokoro import KPipeline


warnings.filterwarnings(
    "ignore",
    message=r".*torch\.jit\.script",
    category=FutureWarning,
)


warnings.filterwarnings(
    "ignore",
    message=".*weight_norm",
    category=FutureWarning,
)


# Hide only the two known, harmless warnings emitted by PyTorch/Kokoro.
warnings.filterwarnings(
    "ignore",
    message="dropout option adds dropout",
    category=UserWarning,
)


def _import_error(exc: ImportError) -> TTSError:
    """Build the error shown when torch or kokoro cannot be imported."""
    return TTSError(
        f"a required package is missing ({exc}). "
        "Run 'pip install kokoro torch'"
    )


def get_torch():
    """Import ``torch`` on first use; it takes seconds to load."""
    try:
        import torch
    except ImportError as exc:
        raise _import_error(exc) from exc
    return torch


def get_kpipeline_class():
    """Import Kokoro's ``KPipeline`` on first use."""
    try:
        from kokoro import KPipeline
    except ImportError as exc:
        raise _import_error(exc) from exc
    return KPipeline


def is_cuda_oom(exc: BaseException) -> bool:
    """Return True if an exception is a CUDA out-of-memory error."""
    oom_class = getattr(get_torch().cuda, "OutOfMemoryError", None)
    if oom_class is not None and isinstance(exc, oom_class):
        return True
    return isinstance(exc, RuntimeError) and (
        "out of memory" in str(exc).lower()
    )


def gpu_oom_message(
    free_mib: int | None = None, total_mib: int | None = None
) -> str:
    """Build the user-facing message for a GPU without free memory.

    Args:
        free_mib: Free GPU memory in MiB, if known.
        total_mib: Total GPU memory in MiB, if known.

    Returns:
        A message explaining the problem and how to solve it.
    """
    detail = ""
    if free_mib is not None and total_mib is not None:
        detail = (
            f" Only {free_mib} MiB are free out of {total_mib} MiB "
            f"(at least {current_settings().gpu_min_free_mib} MiB are needed)."
        )
    return (
        "The GPU does not have enough free memory to generate audio."
        f"{detail} Close other programs that use the GPU (see nvidia-smi) "
        "and try again, or use --device cpu."
    )


def gpu_has_room() -> tuple[bool, str]:
    """Check whether the GPU has enough free memory for the model.

    Returns:
        A tuple ``(has_room, message)``. ``message`` is empty when there is
        room. If the free memory cannot be measured, the GPU is assumed to
        be usable.
    """
    try:
        free, total = get_torch().cuda.mem_get_info()
    except Exception as exc:  # CUDA may fail even to create its context
        if is_cuda_oom(exc):
            return False, gpu_oom_message()
        return True, ""
    free_mib, total_mib = free // 2**20, total // 2**20
    if free_mib < current_settings().gpu_min_free_mib:
        return False, gpu_oom_message(free_mib, total_mib)
    return True, ""


def resolve_device(requested: str) -> str:
    """Choose the device on which the model will run.

    Args:
        requested: ``"cpu"``, ``"cuda"`` or ``"auto"``. With ``"auto"`` the
            GPU is used only if it is available and has free memory.

    Returns:
        ``"cpu"`` or ``"cuda"``.

    Raises:
        GpuMemoryError: If ``"cuda"`` was requested but there is no room.
        TTSError: If ``"cuda"`` was requested but no GPU is available.
    """
    if requested == "cpu":
        return "cpu"
    if not get_torch().cuda.is_available():
        if requested == "cuda":
            raise TTSError(
                "CUDA was requested but no usable GPU was found. "
                "Use --device cpu or --device auto."
            )
        return "cpu"

    has_room, message = gpu_has_room()
    if has_room:
        return "cuda"
    if requested == "auto":
        print(f"[INFO] {message} Falling back to the CPU.", file=sys.stderr)
        return "cpu"
    raise GpuMemoryError(message)


def get_pipeline(internal_lang_code: str, device: str) -> KPipeline:
    """Return a Kokoro pipeline, loading the model only once per language.

    Loading the model takes seconds, so the pipelines are cached for the
    whole run (EPUB books and ``--test`` call the synthesis many times).

    Args:
        internal_lang_code: Kokoro internal language code (``"e"``, ...).
        device: ``"cpu"`` or ``"cuda"``.

    Returns:
        The loaded pipeline.

    Raises:
        GpuMemoryError: If the GPU runs out of memory while loading.
    """
    return _get_pipeline(
        internal_lang_code, device, current_settings().repo_id
    )


@functools.lru_cache(maxsize=None)
def _get_pipeline(
    internal_lang_code: str, device: str, repo_id: str
) -> KPipeline:
    try:
        return get_kpipeline_class()(
            lang_code=internal_lang_code,
            repo_id=repo_id,
            device=device,
        )
    except Exception as exc:
        if is_cuda_oom(exc):
            raise GpuMemoryError(gpu_oom_message()) from exc
        raise


def get_g2p_pipeline(internal_lang_code: str) -> KPipeline:
    """Return a Kokoro pipeline that only converts text to phonemes.

    The voice model is not loaded, so it starts much faster and needs no
    GPU.

    Raises:
        TTSError: If the phonemizer of the language cannot be created.
    """
    return _get_g2p_pipeline(internal_lang_code, current_settings().repo_id)


@functools.lru_cache(maxsize=None)
def _get_g2p_pipeline(internal_lang_code: str, repo_id: str) -> KPipeline:
    try:
        return get_kpipeline_class()(
            lang_code=internal_lang_code, repo_id=repo_id, model=False
        )
    except Exception as exc:
        raise TTSError(
            f"Cannot create the Kokoro phonemizer: {exc}"
        ) from exc


get_pipeline.cache_clear = _get_pipeline.cache_clear
get_g2p_pipeline.cache_clear = _get_g2p_pipeline.cache_clear

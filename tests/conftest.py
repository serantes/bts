"""Shared fixtures and fakes. No test loads torch or the Kokoro model."""

import numpy as np
import pytest

from bts import errors


class FakePipeline:
    """Stands in for ``kokoro.KPipeline``: one fixed-size chunk per call."""

    def __init__(self, samples: int = 240):
        self.samples = samples
        self.calls = []
        self.token_calls = []

    def __call__(self, text, voice=None, speed=1.0, split_pattern=None):
        self.calls.append((text, voice, speed))
        if text.strip():
            yield text, "ph", np.full(self.samples, 0.1, dtype=np.float32)

    def g2p(self, text):
        return text.strip().upper(), None

    def generate_from_tokens(self, phonemes, voice=None, speed=1.0):
        self.token_calls.append((phonemes, voice, speed))
        yield None, phonemes, np.full(self.samples, 0.1, dtype=np.float32)


class FakePlayer:
    """Replacement for ``AudioPlayer`` that records the written audio."""

    instances = []

    def __init__(self):
        self.writes = []
        FakePlayer.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return None

    def write(self, samples):
        self.writes.append(samples)


@pytest.fixture
def fake_pipeline():
    return FakePipeline()


@pytest.fixture
def fake_player():
    FakePlayer.instances = []
    return FakePlayer


@pytest.fixture(autouse=True)
def reset_warnings():
    errors._WARNED.clear()
    yield
    errors._WARNED.clear()

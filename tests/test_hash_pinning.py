# SPDX-License-Identifier: MIT

"""Tests for hash-pinning on TracedFunctional.load()."""

import hashlib
import io
import json
import os
import tempfile

import pytest
import torch

from skala.functional.load import TracedFunctional


def _make_dummy_fun_bytes() -> bytes:
    """Create a minimal TorchScript archive that TracedFunctional.load can open."""

    class Dummy(torch.nn.Module):
        features: list[str] = []

        def get_exc_density(self, data: dict[str, torch.Tensor]) -> torch.Tensor:
            return torch.tensor(0.0)

        def get_exc(self, data: dict[str, torch.Tensor]) -> torch.Tensor:
            return torch.tensor(0.0)

    scripted = torch.jit.script(Dummy())
    extra_files = {
        "metadata": json.dumps({}).encode(),
        "features": json.dumps([]).encode(),
        "expected_d3_settings": json.dumps(None).encode(),
        "protocol_version": json.dumps(2).encode(),
    }
    buf = io.BytesIO()
    torch.jit.save(scripted, buf, _extra_files=extra_files)
    return buf.getvalue()


@pytest.fixture(scope="module")
def dummy_fun_bytes() -> bytes:
    return _make_dummy_fun_bytes()


def test_load_with_correct_hash(dummy_fun_bytes: bytes) -> None:
    """Loading succeeds when the expected hash matches."""
    correct_hash = hashlib.sha256(dummy_fun_bytes).hexdigest()
    func = TracedFunctional.load(
        io.BytesIO(dummy_fun_bytes),
        expected_hash=correct_hash,
    )
    assert isinstance(func, TracedFunctional)


def test_load_with_wrong_hash(dummy_fun_bytes: bytes) -> None:
    """Loading raises ValueError when the hash does not match."""
    wrong_hash = "0" * 64
    with pytest.raises(ValueError, match="Hash mismatch"):
        TracedFunctional.load(
            io.BytesIO(dummy_fun_bytes),
            expected_hash=wrong_hash,
        )


def test_load_without_hash(dummy_fun_bytes: bytes) -> None:
    """Loading without expected_hash still works (opt-out)."""
    func = TracedFunctional.load(
        io.BytesIO(dummy_fun_bytes),
    )
    assert isinstance(func, TracedFunctional)


def test_load_from_path_with_correct_hash(dummy_fun_bytes: bytes) -> None:
    """Hash verification works when loading from a file path."""
    correct_hash = hashlib.sha256(dummy_fun_bytes).hexdigest()
    with tempfile.NamedTemporaryFile(suffix=".fun", delete=False) as f:
        f.write(dummy_fun_bytes)
        f.flush()
        path = f.name
    try:
        func = TracedFunctional.load(path, expected_hash=correct_hash)
        assert isinstance(func, TracedFunctional)
    finally:
        os.unlink(path)


def test_load_from_path_with_wrong_hash(dummy_fun_bytes: bytes) -> None:
    """Hash verification raises ValueError for file path loading too."""
    wrong_hash = "0" * 64
    with tempfile.NamedTemporaryFile(suffix=".fun", delete=False) as f:
        f.write(dummy_fun_bytes)
        f.flush()
        path = f.name
    try:
        with pytest.raises(ValueError, match="Hash mismatch"):
            TracedFunctional.load(path, expected_hash=wrong_hash)
    finally:
        os.unlink(path)

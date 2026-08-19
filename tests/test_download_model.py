# SPDX-License-Identifier: MIT

"""Tests for the C++ integration ``download_model.py`` example script.

The script keeps hand-maintained ``feature_shapes`` / ``feature_dtypes`` /
``feature_labels`` dictionaries that document the inputs each Skala functional
expects. These dictionaries are not exercised anywhere else, so they silently
went stale when new features (e.g. the ``atomic_grid_*`` features used by Skala
1.1) were added to the model. These tests run the script end to end and assert
that every feature requested by each published functional is documented, so the
script cannot drift out of sync again.
"""

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import cast

import pytest
import torch

import skala.functional as functional_module
from skala.functional import FunctionalArtifact, resolve_functional_artifact
from skala.functional._hashes import KNOWN_HASHES
from skala.functional.load import TracedFunctional

_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "cpp"
    / "cpp_integration"
    / "download_model.py"
)

# Functionals downloaded and documented by the script's ``main`` entry point.
_PUBLISHED_FUNCTIONALS = [
    ("microsoft/skala-1.1", "skala-1.1-rev1.fun"),
    ("microsoft/skala-baselines", "ldax.fun"),
]


def test_resolve_functional_artifact_uses_device_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Resolve the device-specific file once without exposing private tables."""
    resolved_path = tmp_path / "skala.fun"
    calls: list[tuple[str, str]] = []

    def fake_hf_hub_download(*, repo_id: str, filename: str) -> str:
        calls.append((repo_id, filename))
        return str(resolved_path)

    monkeypatch.delenv("SKALA_LOCAL_MODEL_PATH", raising=False)
    monkeypatch.setattr(functional_module, "hf_hub_download", fake_hf_hub_download)

    artifact = resolve_functional_artifact("skala-1.1", torch.device("cuda"))
    loaded_functional = cast(TracedFunctional, object())
    load_calls: list[tuple[Path, torch.device | None, str | None]] = []

    def fake_load(
        path: Path,
        device: torch.device | None = None,
        *,
        expected_hash: str | None = None,
    ) -> TracedFunctional:
        load_calls.append((path, device, expected_hash))
        return loaded_functional

    monkeypatch.setattr(TracedFunctional, "load", fake_load)

    assert artifact == FunctionalArtifact(
        resolved_path,
        KNOWN_HASHES[("microsoft/skala-1.1", "skala-1.1-rev1-cuda.fun")],
    )
    assert artifact.load(torch.device("cuda")) is loaded_functional
    assert artifact.load(torch.device("cuda")) is loaded_functional
    assert calls == [("microsoft/skala-1.1", "skala-1.1-rev1-cuda.fun")]
    assert load_calls == [
        (resolved_path, torch.device("cuda"), artifact.expected_hash),
        (resolved_path, torch.device("cuda"), artifact.expected_hash),
    ]


@pytest.fixture(scope="module")
def script_module() -> ModuleType:
    """Import ``download_model.py`` as a module from its file path."""
    spec = importlib.util.spec_from_file_location("download_model", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_download_script_runs(
    script_module: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The script runs end to end without raising.

    ``main`` writes the downloaded ``.fun`` files into the current working
    directory, so run it inside ``tmp_path``. A missing feature in any of the
    documentation dictionaries surfaces here as a ``KeyError``.
    """
    monkeypatch.chdir(tmp_path)
    script_module.main()


@pytest.mark.parametrize("repo_id, filename", _PUBLISHED_FUNCTIONALS)
def test_all_expected_features_are_documented(
    script_module: ModuleType, repo_id: str, filename: str
) -> None:
    """Every feature a published functional requests must be documented."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo_id=repo_id, filename=filename)
    fun = TracedFunctional.load(
        path, expected_hash=KNOWN_HASHES.get((repo_id, filename))
    )

    for feature in fun.features:
        assert feature in script_module.feature_shapes, (
            f"{feature!r} (required by {filename}) is missing from feature_shapes"
        )
        assert feature in script_module.feature_dtypes, (
            f"{feature!r} (required by {filename}) is missing from feature_dtypes"
        )
        assert feature in script_module.feature_labels, (
            f"{feature!r} (required by {filename}) is missing from feature_labels"
        )

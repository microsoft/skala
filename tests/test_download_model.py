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

import pytest

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
    ("microsoft/skala-1.1", "skala-1.1.fun"),
    ("microsoft/skala-baselines", "ldax.fun"),
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

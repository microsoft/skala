# SPDX-License-Identifier: MIT

"""Tests for Hugging Face Hub download helpers."""

import sys
import types
import warnings

from skala._huggingface import hf_hub_download


def test_hf_hub_download_suppresses_hf_xet_deprecation(monkeypatch) -> None:
    """The hf_xet implementation detail warning should not escape downloads."""

    calls = {}

    def fake_hf_hub_download(**kwargs) -> str:
        calls.update(kwargs)
        warnings.warn(
            "hf_xet.download_files() is deprecated. Use "
            "XetSession().new_file_download_group().start_download_file() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return "/tmp/skala.fun"

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(hf_hub_download=fake_hf_hub_download),
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        path = hf_hub_download(
            repo_id="microsoft/skala-1.1",
            filename="skala-1.1.fun",
        )

    assert path == "/tmp/skala.fun"
    assert calls == {"repo_id": "microsoft/skala-1.1", "filename": "skala-1.1.fun"}
    assert not caught

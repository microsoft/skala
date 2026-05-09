# SPDX-License-Identifier: MIT

"""Helpers for interacting with Hugging Face Hub."""

import warnings
from typing import Any

_HF_XET_DOWNLOAD_FILES_DEPRECATION = (
    r"hf_xet\.download_files\(\) is deprecated\. Use "
    r"XetSession\(\)\.new_file_download_group\(\)\.start_download_file\(\) instead\."
)


def hf_hub_download(*args: Any, **kwargs: Any) -> str:
    """Download a file from Hugging Face Hub without surfacing hf_xet internals.

    Args:
        *args: Positional arguments passed through to Hugging Face Hub.
        **kwargs: Keyword arguments passed through to Hugging Face Hub.

    Returns:
        The local path to the downloaded file.
    """
    from huggingface_hub import hf_hub_download as _hf_hub_download

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=_HF_XET_DOWNLOAD_FILES_DEPRECATION,
            category=DeprecationWarning,
        )
        return _hf_hub_download(*args, **kwargs)

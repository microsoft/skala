# SPDX-License-Identifier: MIT

"""
SHA-256 hash digests for known traced functional files.

These hashes are used to verify file integrity before loading with
``torch.jit.load``, which deserializes arbitrary code and is therefore
security-sensitive.
"""

# Maps (repo_id, filename) -> expected SHA-256 hex digest.
KNOWN_HASHES: dict[tuple[str, str], str] = {
    ("microsoft/skala", "skala-1.0.fun"): (
        "08d94436995937eb57c451af7c92e2c7f9e1bff6b7da029a3887e9f9dd4581c0"
    ),
    ("microsoft/skala", "skala-1.0-cuda.fun"): (
        "0b38e13237cec771fed331664aace42f8c0db8f15caca6a5c563085e61e2b1fd"
    ),
}

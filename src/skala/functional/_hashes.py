# SPDX-License-Identifier: MIT

"""
SHA-256 hash digests for known traced functional files.

These hashes are used to verify file integrity before loading with
``torch.jit.load``, which deserializes arbitrary code and is therefore
security-sensitive.
"""

# Maps (repo_id, filename) -> expected SHA-256 hex digest.
KNOWN_HASHES: dict[tuple[str, str], str] = {
    ("microsoft/skala-1.0", "skala-1.0.fun"): (
        "08d94436995937eb57c451af7c92e2c7f9e1bff6b7da029a3887e9f9dd4581c0"
    ),
    ("microsoft/skala-1.0", "skala-1.0-cuda.fun"): (
        "0b38e13237cec771fed331664aace42f8c0db8f15caca6a5c563085e61e2b1fd"
    ),
    ("microsoft/skala-1.1", "skala-1.1.fun"): (
        "0c8432ac3f03c8f1276372df9aca5b7ee7f8939d47a8789eb158976e89aa0606"
    ),
    (
        "microsoft/skala-1.1",
        "skala-1.1-cuda.fun",
    ): "f77be6002d873c0a2384b6df7850d32bbec519036344ff5fdde9730c6f9a4326",
}

# SPDX-License-Identifier: MIT

"""Check repository source links against the commit being built in CI."""

from __future__ import annotations

import os
from typing import Any


def _check_source_uri(app: Any, uri: str) -> str | None:
    revision = os.environ.get("GITHUB_SHA")
    if not revision:
        return None

    for kind in ("tree", "blob"):
        prefix = f"https://github.com/microsoft/skala/{kind}/main/"
        if uri.startswith(prefix):
            return (
                f"https://github.com/microsoft/skala/{kind}/{revision}/"
                + uri.removeprefix(prefix)
            )
    return None


def setup(app: Any) -> dict[str, object]:
    """Keep published links on main while checking new files in pull requests."""
    app.connect("linkcheck-process-uri", _check_source_uri)
    return {"parallel_read_safe": True, "parallel_write_safe": True}

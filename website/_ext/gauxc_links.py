# SPDX-License-Identifier: MIT

"""Point links to the GauXC guide at its build output next to this site.

The GauXC guide is a separate Sphinx project that is built into the ``gauxc/`` subdirectory of
this site's HTML output. The sources link to its published URL, which also works on GitHub and
in the toctree, and this extension rewrites those links to relative paths in the generated HTML,
so that they resolve in local builds and on the deployed site alike.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

_GAUXC_LINK = re.compile(
    r'href="https://microsoft\.github\.io/skala/gauxc/([^"#]*)(#[^"]*)?"'
)


def _relink(app: Any, exception: Exception | None) -> None:
    if exception is not None or app.builder.format != "html":
        return

    outdir = Path(app.outdir)
    for page in outdir.rglob("*.html"):
        if page.relative_to(outdir).parts[0] == "gauxc":
            continue
        root = Path(os.path.relpath(outdir, page.parent)).as_posix()

        def relative(match: re.Match[str], root: str = root) -> str:
            path = match.group(1)
            if not path or path.endswith("/"):
                path += "index.html"
            elif not path.endswith(".html"):
                path += ".html"
            return f'href="{root}/gauxc/{path}{match.group(2) or ""}"'

        html = page.read_text(encoding="utf-8")
        updated = _GAUXC_LINK.sub(relative, html)
        if updated != html:
            page.write_text(updated, encoding="utf-8")


def setup(app: Any) -> dict[str, object]:
    app.connect("build-finished", _relink)
    return {"parallel_read_safe": True, "parallel_write_safe": True}

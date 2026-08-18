# SPDX-License-Identifier: MIT

"""Optional prose configuration and Markdown rendering."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import yaml
from markdown_it import MarkdownIt
from markdown_it.rules_inline import StateInline

from ._normalize import as_mapping

DEFAULT_PROSE: dict[str, Any] = {
    "title": "Skala benchmark comparison",
    "author": "",
    "date": "",
    "abstract": (
        "This offline report compares benchmark measurements across the collected "
        "compute environments and basis sets."
    ),
    "intro": (
        "Use the controls to compare environments and basis sets. No study-specific "
        "interpretation was supplied for this local report."
    ),
    "comparison": {
        "intro": (
            "The cards show the hardware and software metadata recorded with each "
            "benchmark run."
        ),
        "environments": {},
        "notes": [],
    },
    "plots": {
        "xc_eval": "Measured exchange-correlation functional evaluation time.",
        "numint": "Measured numerical-integration time.",
        "jk": "Measured Coulomb and exchange build time.",
        "cycle": "Measured steady-state SCF iteration time.",
        "iterations": "Recorded SCF iteration counts.",
        "total": "Measured total SCF kernel time.",
        "setup": "Measured setup time before the SCF loop.",
        "composition": "Measured composition of one steady-state SCF iteration.",
        "run_composition": "Measured composition of the complete SCF kernel.",
        "startup": (
            "Start-up and first-use costs are outside the plotted SCF timings."
        ),
    },
    "closing": (
        "No study-specific conclusions were supplied. Interpret the measured points "
        "and fitted lines in the context of the recorded environments."
    ),
}


def _math_inline(state: StateInline, silent: bool) -> bool:
    if state.src[state.pos] != "$":
        return False
    end = state.src.find("$", state.pos + 1)
    if end < 0 or end == state.pos + 1:
        return False
    if not silent:
        token = state.push("math_inline", "span", 0)
        token.content = state.src[state.pos + 1 : end]
    state.pos = end + 1
    return True


def _render_math(tokens: list[Any], index: int, *_: Any) -> str:
    import html

    tex = html.escape(tokens[index].content, quote=True)
    return f'<span class="tex" data-tex="{tex}">{tex}</span>'


_MARKDOWN = MarkdownIt("commonmark", {"html": False})
_MARKDOWN.inline.ruler.before("escape", "math_inline", _math_inline)
cast(Any, _MARKDOWN.renderer).rules["math_inline"] = _render_math


def load_prose(path: str | Path | None) -> dict[str, Any]:
    """Load optional prose YAML with defaults.

    Args:
        path: YAML path, or ``None`` for defaults.

    Returns:
        Normalized prose configuration.

    Raises:
        ValueError: If the parsed YAML root is not a mapping.
    """
    loaded: Mapping[str, Any] = {}
    if path is not None:
        prose_path = Path(path)
        parsed = yaml.safe_load(prose_path.read_text(encoding="utf-8")) or {}
        if not isinstance(parsed, Mapping):
            raise ValueError("prose YAML root must be a mapping")
        loaded = parsed

    result = _deep_merge(DEFAULT_PROSE, loaded)
    result["comparison"] = as_mapping(result.get("comparison"))
    result["comparison"]["environments"] = as_mapping(
        result["comparison"].get("environments")
    )
    notes = result["comparison"].get("notes")
    result["comparison"]["notes"] = list(notes) if isinstance(notes, list) else []
    result["plots"] = as_mapping(result.get("plots"))
    return result


def markdown_to_html(markdown: Any) -> str:
    """Render Markdown with raw HTML disabled.

    Args:
        markdown: Text-like value to render.

    Returns:
        Sanitized HTML.
    """
    text = str(markdown or "").strip()
    return _MARKDOWN.render(text) if text else ""


def _deep_merge(
    defaults: Mapping[str, Any], overrides: Mapping[str, Any]
) -> dict[str, Any]:
    result = dict(defaults)
    for key, value in overrides.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _deep_merge(as_mapping(result[key]), value)
        else:
            result[key] = value
    return result

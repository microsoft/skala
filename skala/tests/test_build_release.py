from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

import pytest
from tools import build_release

REPOSITORY = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    ("package", "module_name", "dependencies", "supports_macos"),
    [
        ("microsoft-skala", "microsoft_skala", [], True),
        ("skala-cuda12x", "skala_cuda", ["gpu4pyscf-cuda12x==1.8.1"], False),
        ("skala-cuda13x", "skala_cuda", ["gpu4pyscf-cuda13x==1.8.1"], False),
    ],
)
def test_render_compatibility_pyproject(
    package: str,
    module_name: str,
    dependencies: list[str],
    supports_macos: bool,
) -> None:
    module, rendered = build_release.render_compatibility_pyproject(REPOSITORY, package)
    generated: dict[str, Any] = tomllib.loads(rendered)
    project: dict[str, Any] = generated["project"]

    assert module == module_name
    assert project["name"] == package
    assert project["version"] == build_release.project_version(REPOSITORY / "skala")
    assert project["dependencies"] == [
        f"skala=={project['version']}",
        *dependencies,
    ]
    assert ("Operating System :: MacOS" in project["classifiers"]) is supports_macos
    assert build_release.PLACEHOLDER_PATTERN.search(rendered) is None

    module_path = f"src/{module_name}"
    assert generated["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"] == [
        module_path
    ]


def test_render_template_rejects_unresolved_placeholders(tmp_path: Path) -> None:
    template = tmp_path / "pyproject.toml"
    template.write_text('[project]\nname = "@NAME@"\nversion = "@VERSION@"\n')

    with pytest.raises(ValueError, match="@VERSION@"):
        build_release.render_template(template, {"@NAME@": "example"})


def test_render_template_rejects_invalid_toml(tmp_path: Path) -> None:
    template = tmp_path / "pyproject.toml"
    template.write_text("[project\n")

    with pytest.raises(tomllib.TOMLDecodeError):
        build_release.render_template(template, {})

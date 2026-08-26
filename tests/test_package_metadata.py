# SPDX-License-Identifier: MIT

"""Tests for consistent package and Skala dependency versions."""

import tomllib
from pathlib import Path
from typing import Any

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version

REPOSITORY = Path(__file__).resolve().parents[1]
PACKAGE_DIRECTORIES = ("model", "benchmark", "skala")


def _load_project(package_directory: str) -> dict[str, Any]:
    with (REPOSITORY / package_directory / "pyproject.toml").open("rb") as stream:
        metadata = tomllib.load(stream)
    return dict(metadata["project"])


def _dependencies(project: dict[str, Any]) -> dict[str, Requirement]:
    requirements = (Requirement(value) for value in project["dependencies"])
    return {
        canonicalize_name(requirement.name): requirement for requirement in requirements
    }


PROJECTS = {directory: _load_project(directory) for directory in PACKAGE_DIRECTORIES}
DEPENDENCIES = {
    directory: _dependencies(project) for directory, project in PROJECTS.items()
}


def test_package_versions_match() -> None:
    """All published Python packages use the same release version."""
    versions = {
        directory: project["version"] for directory, project in PROJECTS.items()
    }

    assert len(set(versions.values())) == 1, f"Package versions differ: {versions}"


def test_skala_dependency_versions_match_runtime_version() -> None:
    """Every Skala dependency uses the version released alongside it."""
    runtime_version = Version(PROJECTS["skala"]["version"])

    for package_directory, dependencies in DEPENDENCIES.items():
        requirement = dependencies.get("skala")
        if requirement is None:
            continue

        declared_versions = {
            Version(specifier.version)
            for specifier in requirement.specifier
            if specifier.operator in {"==", ">=", "~="}
        }
        assert runtime_version in requirement.specifier, (
            f"{package_directory} requires {requirement}, which excludes Skala {runtime_version}"
        )
        assert declared_versions == {runtime_version}, (
            f"{package_directory} requires {requirement}; its Skala dependency version "
            f"must match {runtime_version}"
        )

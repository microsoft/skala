"""Validate built release artifacts for expected and forbidden package contents."""

from __future__ import annotations

import argparse
import tarfile
import zipfile
from collections.abc import Iterable
from pathlib import Path, PurePosixPath

PACKAGES = ("skala", "microsoft-skala", "skala-cuda12x", "skala-cuda13x")

FORBIDDEN_PARTS = {"tests", "model", "gauxc", "docs", "examples", "benchmark"}
FORBIDDEN_RUNTIME_PATHS = {
    PurePosixPath("skala/functional/layers.py"),
    PurePosixPath("skala/functional/model.py"),
}
REQUIRED_RUNTIME_PATHS = {
    PurePosixPath("skala/__init__.py"),
    PurePosixPath("skala/ase/__init__.py"),
    PurePosixPath("skala/functional/load.py"),
    PurePosixPath("skala/gpu4pyscf/__init__.py"),
    PurePosixPath("skala/py.typed"),
    PurePosixPath("skala/pyscf/__init__.py"),
}


def archive_paths(artifact: Path) -> set[PurePosixPath]:
    """Return normalized member paths from one wheel or source archive."""
    if artifact.suffix == ".whl":
        with zipfile.ZipFile(artifact) as archive:
            return {
                PurePosixPath(name)
                for name in archive.namelist()
                if not name.endswith("/")
            }

    if artifact.name.endswith(".tar.gz"):
        with tarfile.open(artifact, "r:gz") as archive:
            members = {
                PurePosixPath(member.name)
                for member in archive.getmembers()
                if member.isfile()
            }
        return {
            PurePosixPath(*member.parts[1:])
            for member in members
            if len(member.parts) > 1
        }

    raise ValueError(f"Unsupported release artifact: {artifact}")


def assert_no_forbidden_paths(paths: Iterable[PurePosixPath], artifact: Path) -> None:
    """Reject component source that is not part of the runtime release."""
    forbidden = {
        path
        for path in paths
        if FORBIDDEN_PARTS.intersection(path.parts)
        or PurePosixPath(*path.parts[1:] if path.parts[:1] == ("src",) else path.parts)
        in FORBIDDEN_RUNTIME_PATHS
    }
    if forbidden:
        members = "\n".join(f"  - {path}" for path in sorted(forbidden))
        raise ValueError(f"Forbidden members in {artifact}:\n{members}")


def validate_artifact(artifact: Path, package: str) -> None:
    """Validate one built artifact for the selected distribution."""
    paths = archive_paths(artifact)
    assert_no_forbidden_paths(paths, artifact)

    if package == "skala":
        source_prefix = (
            PurePosixPath("src") if artifact.name.endswith(".tar.gz") else None
        )
        required = {
            source_prefix / path if source_prefix is not None else path
            for path in REQUIRED_RUNTIME_PATHS
        }
    else:
        module = "microsoft_skala" if package == "microsoft-skala" else "skala_cuda"
        prefix = (
            PurePosixPath("src")
            if artifact.name.endswith(".tar.gz")
            else PurePosixPath()
        )
        required = {prefix / module / "__init__.py", prefix / module / "py.typed"}

    missing = required - paths
    if missing:
        members = "\n".join(f"  - {path}" for path in sorted(missing))
        raise ValueError(f"Required members missing from {artifact}:\n{members}")


def matching_artifacts(directory: Path, package: str) -> list[Path]:
    """Find the wheel and sdist produced for one distribution."""
    normalized = package.replace("-", "_")
    prefixes = (f"{package}-", f"{normalized}-")
    artifacts = sorted(
        path
        for path in directory.iterdir()
        if path.name.startswith(prefixes)
        and (path.suffix == ".whl" or path.name.endswith(".tar.gz"))
    )
    if not any(path.suffix == ".whl" for path in artifacts):
        raise ValueError(f"No wheel found for {package} in {directory}")
    if not any(path.name.endswith(".tar.gz") for path in artifacts):
        raise ValueError(f"No sdist found for {package} in {directory}")
    return artifacts


def main() -> None:
    """Validate release artifacts from the command line."""
    parser = argparse.ArgumentParser()
    parser.add_argument("package", choices=PACKAGES)
    parser.add_argument("--dist-dir", type=Path, default=Path("dist"))
    args = parser.parse_args()

    for artifact in matching_artifacts(args.dist_dir.resolve(), args.package):
        validate_artifact(artifact, args.package)
        print(f"validated {artifact}")


if __name__ == "__main__":
    main()

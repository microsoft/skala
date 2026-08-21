from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

PACKAGES = ("skala", "microsoft-skala", "skala-cuda12x", "skala-cuda13x")


def project_version(repository: Path) -> str:
    """Read the release version from the canonical project metadata."""
    with (repository / "pyproject.toml").open("rb") as stream:
        metadata = tomllib.load(stream)
    return str(metadata["project"]["version"])


def prepare_source(repository: Path, staging: Path, package: str) -> None:
    """Create an isolated source tree for one distribution package."""
    shutil.copy2(repository / "README.md", staging / "README.md")
    shutil.copy2(repository / "LICENSE.txt", staging / "LICENSE.txt")
    source_root = staging / "src"
    source_root.mkdir()

    if package == "skala":
        shutil.copy2(repository / "pyproject.toml", staging / "pyproject.toml")
        shutil.copytree(repository / "src" / "skala", source_root / "skala")
        return

    version = project_version(repository)
    if package == "microsoft-skala":
        template = (
            repository / ".github" / "workflows" / "pypi" / "microsoft-skala.toml"
        )
        module_name = "microsoft_skala"
        replacements = {"@VERSION@": version}
    else:
        template = repository / ".github" / "workflows" / "pypi" / "skala-cuda.toml"
        module_name = "skala_cuda"
        replacements = {
            "@VERSION@": version,
            "@CUDA@": package.removeprefix("skala-cuda"),
        }

    rendered = template.read_text(encoding="utf-8")
    for placeholder, value in replacements.items():
        rendered = rendered.replace(placeholder, value)
    (staging / "pyproject.toml").write_text(rendered, encoding="utf-8")

    module = source_root / module_name
    module.mkdir()
    (module / "__init__.py").write_text("from skala import *\n", encoding="utf-8")
    (module / "py.typed").touch()


def build_package(repository: Path, output: Path, package: str) -> None:
    """Build a wheel and source distribution without modifying the checkout."""
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"{package}-") as temporary_directory:
        staging = Path(temporary_directory)
        prepare_source(repository, staging, package)
        subprocess.run(
            [sys.executable, "-m", "build", "--outdir", str(output), str(staging)],
            check=True,
        )


def main() -> None:
    """Build the requested compatibility distribution."""
    parser = argparse.ArgumentParser()
    parser.add_argument("package", choices=PACKAGES)
    parser.add_argument("--output-dir", type=Path, default=Path("dist"))
    args = parser.parse_args()

    repository = Path(__file__).resolve().parents[1]
    output = args.output_dir.resolve()
    build_package(repository, output, args.package)


if __name__ == "__main__":
    main()

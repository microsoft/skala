# SPDX-License-Identifier: MIT

"""
Functional module for exchange-correlation functionals.

This module provides the main interface for loading and using various
exchange-correlation functionals, including traditional functionals
(LDA, PBE, TPSS) and the Skala neural functional.
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download

from skala.functional._hashes import KNOWN_HASHES
from skala.functional.base import ExcFunctionalBase
from skala.functional.load import TracedFunctional
from skala.functional.model import SkalaFunctional
from skala.functional.traditional import (
    LDA,
    PBE,
    R2SCAN,
    RSCAN,
    SCAN,
    SPW92,
    TPSS,
    XC_FUNCTIONAL_MAP,
)

__all__ = [
    "LDA",
    "PBE",
    "R2SCAN",
    "RSCAN",
    "SCAN",
    "SPW92",
    "TPSS",
    "ExcFunctionalBase",
    "FunctionalArtifact",
    "SkalaFunctional",
    "TracedFunctional",
    "load_functional",
    "resolve_functional_artifact",
]

_SKALA_VERSIONS = {
    "skala-1.0": ("skala-1.0.fun", "skala-1.0-cuda.fun"),
    "skala-1.1": ("skala-1.1-rev1.fun", "skala-1.1-rev1-cuda.fun"),
    "skala-1.1-rev0": ("skala-1.1.fun", "skala-1.1-cuda.fun"),
    "skala-1.1-rev1": ("skala-1.1-rev1.fun", "skala-1.1-rev1-cuda.fun"),
}


@dataclass(frozen=True)
class FunctionalArtifact:
    """Resolved serialized functional with its integrity metadata."""

    path: Path
    expected_hash: str | None

    def load(self, device: torch.device | None = None) -> TracedFunctional:
        """Load and verify the resolved functional on one device."""
        return TracedFunctional.load(
            self.path,
            device=device,
            expected_hash=self.expected_hash,
        )


def resolve_functional_artifact(
    name: str, device: torch.device | None = None
) -> FunctionalArtifact:
    """Resolve one published Skala functional to a verified local artifact.

    Args:
        name: Published Skala functional name.
        device: Device whose CPU or CUDA artifact should be resolved.

    Returns:
        Local artifact path and its expected SHA-256 hash.

    Raises:
        ValueError: If ``name`` is not a published Skala functional.
    """
    func_name = name.lower()
    if func_name not in _SKALA_VERSIONS:
        raise ValueError(f"Cannot resolve non-Skala functional {name!r}")

    env_path = os.environ.get("SKALA_LOCAL_MODEL_PATH")
    if env_path is not None:
        logging.getLogger(__name__).warning(
            "Loading model from SKALA_LOCAL_MODEL_PATH; "
            "SHA-256 hash verification is disabled."
        )
        return FunctionalArtifact(Path(env_path), None)

    device_type = torch.get_default_device().type if device is None else device.type
    repo_id = f"microsoft/{func_name.partition('-rev')[0]}"
    cpu_file, cuda_file = _SKALA_VERSIONS[func_name]
    filename = cpu_file if device_type == "cpu" else cuda_file
    path = hf_hub_download(repo_id=repo_id, filename=filename)
    return FunctionalArtifact(
        Path(path),
        KNOWN_HASHES.get((repo_id, filename)),
    )


def load_functional(
    name: str, device: torch.device | None = None
) -> ExcFunctionalBase | str:
    """Load an exchange-correlation functional by name.

    Args:
        name: Name of the functional. Skala-native values:

            - ``"skala-1.1"``: Skala 1.1 neural functional (recommended).
            - ``"skala-1.1-rev1"``: Skala 1.1 pinned to revision 1.
            - ``"skala-1.1-rev0"``: Skala 1.1 pinned to the original revision.
            - ``"skala-1.0"``: Skala 1.0 neural functional (legacy, traced only).
            - ``"lda"``: Local Density Approximation.
            - ``"spw92"``: SPW92 (LDA with PW92 correlation).
            - ``"pbe"``: Perdew-Burke-Ernzerhof functional.
            - ``"tpss"``: Tao-Perdew-Staroverov-Scuseria meta-GGA.
            - ``"scan"``: Strongly Constrained and Appropriately Normed.
            - ``"rscan"``: Regularized SCAN.
            - ``"r2scan"``: Regularized-restored SCAN.

            Any other string is returned as-is for native PySCF/gpu4pyscf evaluation.

        device: Device to load the functional onto.

    Returns:
        An ``ExcFunctionalBase`` instance for Skala-native functionals, or the
        name string for PySCF-native functionals.

    Example:
        >>> from skala.features import Feature
        >>> func = load_functional("skala-1.1")
        >>> func.features[:3] == [Feature.DENSITY, Feature.KIN, Feature.GRAD]
        True
        >>> func = load_functional("lda")
        >>> func.features == [Feature.DENSITY, Feature.GRID_WEIGHTS]
        True
        >>> load_functional("b3lyp")
        'b3lyp'
    """
    func_name = name.lower()

    if func_name == "skala":
        raise ValueError(
            'The generic functional name "skala" is no longer supported. '
            'Please use "skala-1.0" or "skala-1.1".'
        )

    if func_name in _SKALA_VERSIONS:
        return resolve_functional_artifact(func_name, device).load(device)

    elif func_name in XC_FUNCTIONAL_MAP:
        func = XC_FUNCTIONAL_MAP[func_name]()
        if device is not None:
            func = func.to(device=device)
        return func
    else:
        return name

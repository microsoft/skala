# SPDX-License-Identifier: MIT

"""
Functional module for exchange-correlation functionals.

This module provides the main interface for loading and using various
exchange-correlation functionals, including traditional functionals
(LDA, PBE, TPSS) and the Skala neural functional.
"""

import logging
import os

import torch
from huggingface_hub import hf_hub_download

from skala.functional._hashes import KNOWN_HASHES
from skala.functional.base import ExcFunctionalBase
from skala.functional.load import TracedFunctional
from skala.functional.model import SkalaFunctional
from skala.functional.traditional import LDA, PBE, SPW92, TPSS, SpinScaledXCFunctional

__all__ = [
    "ExcFunctionalBase",
    "SkalaFunctional",
    "TracedFunctional",
    "LDA",
    "PBE",
    "SPW92",
    "TPSS",
    "load_functional",
]

_SKALA_VERSIONS = {
    "skala-1.0": ("skala-1.0.fun", "skala-1.0-cuda.fun"),
    "skala-1.1": ("skala-1.1.fun", "skala-1.1-cuda.fun"),
}


def load_functional(
    name: str, device: torch.device | None = None
) -> ExcFunctionalBase | str:
    """Load an exchange-correlation functional by name.

    Args:
        name: Name of the functional. Skala-native values:

            - ``"skala-1.1"``: Skala 1.1 neural functional (recommended).
            - ``"skala-1.0"``: Skala 1.0 neural functional (legacy, traced only).
            - ``"lda"``: Local Density Approximation.
            - ``"spw92"``: SPW92 (LDA with PW92 correlation).
            - ``"pbe"``: Perdew-Burke-Ernzerhof functional.
            - ``"tpss"``: Tao-Perdew-Staroverov-Scuseria meta-GGA.

            Any other string is returned as-is for native PySCF/gpu4pyscf evaluation.

        device: Device to load the functional onto.

    Returns:
        An ``ExcFunctionalBase`` instance for Skala-native functionals, or the
        name string for PySCF-native functionals.

    Example:
        >>> func = load_functional("skala-1.1")
        >>> func.features
        ['density', 'kin', 'grad', 'grid_coords', 'grid_weights', ...
        >>> func = load_functional("lda")
        >>> func.features
        ['density', 'grid_weights']
        >>> load_functional("b3lyp")
        'b3lyp'
    """
    func_name = name.lower()

    if func_name == "skala":
        raise ValueError(
            'The generic functional name "skala" is no longer supported. '
            'Please use "skala-1.0" or "skala-1.1".'
        )

    func: SpinScaledXCFunctional
    if func_name in _SKALA_VERSIONS:
        env_path = os.environ.get("SKALA_LOCAL_MODEL_PATH")
        if env_path is not None:
            logging.getLogger(__name__).warning(
                "Loading model from SKALA_LOCAL_MODEL_PATH; "
                "SHA-256 hash verification is disabled."
            )
            path = env_path
            expected_hash = None
        else:
            device_type = (
                torch.get_default_device().type if device is None else device.type
            )
            repo_id = f"microsoft/{func_name}"
            cpu_file, cuda_file = _SKALA_VERSIONS[func_name]
            filename = cpu_file if device_type == "cpu" else cuda_file
            path = hf_hub_download(repo_id=repo_id, filename=filename)
            expected_hash = KNOWN_HASHES.get((repo_id, filename))

        return TracedFunctional.load(path, device=device, expected_hash=expected_hash)

    elif func_name == "lda":
        func = LDA()
    elif func_name == "spw92":
        func = SPW92()
    elif func_name == "pbe":
        func = PBE()
    elif func_name == "tpss":
        func = TPSS()
    else:
        return name

    if device is not None:
        func = func.to(device=device)

    return func

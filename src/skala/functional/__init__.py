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
from skala.functional.traditional import LDA, PBE, SPW92, TPSS

__all__ = [
    "ExcFunctionalBase",
    "TracedFunctional",
    "LDA",
    "PBE",
    "SPW92",
    "TPSS",
    "load_functional",
]


def load_functional(name: str, device: torch.device | None = None) -> ExcFunctionalBase:
    """
    Load an exchange-correlation functional by name.

    Parameters
    ----------
    name : str
        Name of the functional. Supported values:

        - "skala": The Skala neural functional
        - "lda": Local Density Approximation
        - "spw92": SPW92 (LDA with PW92 correlation)
        - "pbe": Perdew-Burke-Ernzerhof functional
        - "tpss": Tao-Perdew-Staroverov-Scuseria meta-GGA

    Returns
    -------
    ExcFunctionalBase
        The loaded functional instance.

    Raises
    ------
    ValueError
        If the functional name is not recognized.

    Example
    -------
    >>> func = load_functional("skala")
    >>> func.features
    ['density', 'kin', 'grad', 'grid_coords', 'grid_weights', 'coarse_0_atomic_coords']
    >>> func = load_functional("lda")
    >>> func.features
    ['density', 'grid_weights']
    """
    func_name = name.lower()

    if func_name == "skala":
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
            repo_id = "microsoft/skala"
            filename = "skala-1.0.fun" if device_type == "cpu" else "skala-1.0-cuda.fun"
            path = hf_hub_download(repo_id=repo_id, filename=filename)
            expected_hash = KNOWN_HASHES.get((repo_id, filename))
        with open(path, "rb") as fd:
            return TracedFunctional.load(fd, device=device, expected_hash=expected_hash)
    elif func_name == "lda":
        func = LDA()
    elif func_name == "spw92":
        func = SPW92()
    elif func_name == "pbe":
        func = PBE()
    elif func_name == "tpss":
        func = TPSS()
    else:
        raise ValueError(
            f"Unknown functional: {name}. Please provide a valid functional name or path to a traced functional file."
        )
    if device is not None:
        func = func.to(device=device)
    return func

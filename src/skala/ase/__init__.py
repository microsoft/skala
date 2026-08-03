# SPDX-License-Identifier: MIT

try:
    import ase  # noqa: F401
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "ASE is not installed. Please install it with `pip install ase` or `conda install ase`."
    ) from e


from skala.ase.calculator import Skala

__all__ = ["Skala"]

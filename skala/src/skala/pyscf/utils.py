# SPDX-License-Identifier: MIT

"""Utilities for pyscf version checking."""

from pyscf import __version__ as pyscf_version


def pyscf_version_newer_than_2_10() -> bool:
    """Check if the installed PySCF version is newer than 2.10.0."""
    major, minor, _ = map(int, pyscf_version.split("."))
    return (major == 2 and minor >= 10) or (major > 2)

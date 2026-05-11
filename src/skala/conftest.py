# SPDX-License-Identifier: MIT

"""
Pytest configuration for the ``skala`` package.

The ``skala.gpu4pyscf`` subpackage requires a CUDA-capable device. On hosts
without CUDA, importing any module under ``skala.gpu4pyscf`` raises
``ImportError`` (see ``skala/gpu4pyscf/__init__.py``), which makes
``pytest --doctest-modules`` fail at collection time. To keep CPU-only CI
green, we skip doctest collection of that subpackage when CUDA is
unavailable. The dedicated tests in ``tests/test_gpu4pyscf_*.py`` apply
their own ``pytest.skip`` guard.

This file lives in ``src/skala/`` rather than ``src/skala/gpu4pyscf/`` so
pytest can load it without first importing ``skala.gpu4pyscf``.
"""

import torch

collect_ignore_glob: list[str] = []

if not torch.cuda.is_available():
    collect_ignore_glob.append("gpu4pyscf/*.py")

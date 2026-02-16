# SPDX-License-Identifier: MIT

"""
Tools to load functionals from serialized torchscript checkpoints.
"""

import hashlib
import io
import json
import os
from collections.abc import Iterable, Mapping
from typing import IO, Any, cast

import torch

from skala.functional.base import ExcFunctionalBase

PROTOCOL_VERSION = 2


class TracedFunctional(ExcFunctionalBase):
    """
    A TracedFunctional is a functional represented by a torch.jit graph.
    It can be saved to disk and loaded again with no dependencies on our code base (only libtorch).

    You can construct a TracedFunctional from any functional that implements `ExcFunctionalBase`.
    When you do that, the execution is traced with a dummy input, and the resulting graph is saved.
    The main thing to keep in mind, is that, if the functional uses branches or loops that are
    runtime-dependent, the traced graph will only take the path that's taken by the dummy input.
    This behavior can sometimes be avoided by using the `@torch.jit.script_if_tracing` decorator.

    To learn about tracing and torchscript, see
        https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html
    and for a few more advanced topics
        https://paulbridger.com/posts/mastering-torchscript/
    """

    def __init__(
        self,
        traced_model: torch.jit.ScriptModule,
        *,
        features: Iterable[str],
        metadata: Mapping[str, Any],
        expected_d3_settings: str | None,
    ):
        super().__init__()
        self._traced_model = traced_model
        self.metadata = dict(metadata)
        self.features = list(features)
        self.expected_d3_settings = expected_d3_settings

    def get_d3_settings(self) -> str | None:
        """
        Returns the D3 settings that this functional expects.
        If the functional does not use D3, it returns None.
        """
        return self.expected_d3_settings

    def get_exc_density(self, data: dict[str, torch.Tensor]) -> torch.FloatTensor:
        return self._traced_model.get_exc_density(data)

    def get_exc(self, data: dict[str, torch.Tensor]) -> torch.FloatTensor:
        return self._traced_model.get_exc(data)

    @property
    def original_name(self) -> str:
        if not hasattr(self._traced_model, "original_name"):
            raise RuntimeError(
                "The traced model does not have an 'original_name' attribute."
            )
        if not isinstance(original_name := self._traced_model.original_name, str):
            raise RuntimeError(
                "The 'original_name' attribute of the traced model is not a string."
            )
        return original_name

    @classmethod
    def load(
        cls,
        fp: str | bytes | os.PathLike[str] | IO[bytes],
        device: torch.device | None = None,
        *,
        expected_hash: str | None = None,
    ) -> "TracedFunctional":
        """Load a traced functional from a file.

        Args:
            fp: File path or readable binary stream.
            device: Target device for the loaded model.
            expected_hash: If provided, the SHA-256 hex digest that the file
                content must match. A ``ValueError`` is raised on mismatch.
        """
        extra_files = {
            "metadata": b"",
            "features": b"",
            "expected_d3_settings": b"",
            "protocol_version": b"",
        }

        if device is None:
            device = torch.get_default_device()

        if expected_hash is not None:
            # Read the whole file into memory so we can hash it before
            # passing it to the unsafe torch.jit.load deserializer.
            if isinstance(fp, (str, bytes, os.PathLike)):
                with open(fp, "rb") as f:
                    data = f.read()
            else:
                data = fp.read()
            actual_hash = hashlib.sha256(data).hexdigest()
            if actual_hash != expected_hash:
                raise ValueError(
                    f"Hash mismatch for functional file: "
                    f"expected {expected_hash}, got {actual_hash}. "
                    f"The file may have been tampered with."
                )
            fp = io.BytesIO(data)

        traced_model = torch.jit.load(fp, _extra_files=extra_files, map_location=device)

        _metadata = json.loads(extra_files["metadata"].decode("utf-8"))
        if not isinstance(_metadata, dict):
            raise RuntimeError(
                "metadata in traced functional extra_files does not have the correct format (dict)."
            )
        if not all([isinstance(key, str) for key in _metadata]):
            raise RuntimeError("metadata keys in traced functional must be strings.")
        metadata = cast(dict[str, Any], _metadata)

        _features = json.loads(extra_files["features"].decode("utf-8"))
        if not isinstance(_features, list):
            raise RuntimeError(
                "features in traced functional extra_files does not have the correct format (list)."
            )
        if not all([isinstance(feat, str) for feat in _features]):
            raise RuntimeError(
                "features in traced functional must be a list of strings."
            )
        features = cast(list[str], _features)

        if not isinstance(
            (
                protocol_version := json.loads(
                    extra_files["protocol_version"].decode("utf-8")
                )
            ),
            int,
        ):
            raise RuntimeError(
                "protocol_version in traced functional extra_files must be an integer."
            )

        if not isinstance(
            (
                expected_d3_settings := json.loads(
                    extra_files["expected_d3_settings"].decode("utf-8")
                )
            ),
            (str, type(None)),
        ):
            raise RuntimeError(
                "expected_d3_settings in traced functional extra_files must be a string or None."
            )

        if protocol_version != PROTOCOL_VERSION:
            raise RuntimeError(
                f"Cannot load model with protocol version {protocol_version} "
                f"with this version of the library (supports {PROTOCOL_VERSION})"
            )

        return cls(
            traced_model,
            features=features,
            metadata=metadata,
            expected_d3_settings=expected_d3_settings,
        )

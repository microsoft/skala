from __future__ import annotations

import skalaxc
from mpi4py import MPI
from typing_extensions import assert_type  # noqa: UP035

settings = skalaxc.DeviceRuntimeSettings()
assert_type(
    skalaxc.RuntimeEnvironment(MPI.COMM_SELF),
    skalaxc.RuntimeEnvironment,
)
assert_type(
    skalaxc.RuntimeEnvironment(MPI.COMM_SELF, settings),
    skalaxc.RuntimeEnvironment,
)

skalaxc.RuntimeEnvironment()  # type: ignore[call-overload]
skalaxc.RuntimeEnvironment(None)  # type: ignore[call-overload]
skalaxc.RuntimeEnvironment(settings)  # type: ignore[call-overload]

from __future__ import annotations

import skalaxc
from mpi4py import MPI
from typing_extensions import assert_type  # noqa: UP035

assert_type(
    skalaxc.RuntimeEnvironment(MPI.COMM_SELF),
    skalaxc.RuntimeEnvironment,
)
skalaxc.RuntimeEnvironment()  # type: ignore[call-arg]
skalaxc.RuntimeEnvironment(None)  # type: ignore[arg-type]
skalaxc.DeviceRuntimeSettings()  # type: ignore[attr-defined]

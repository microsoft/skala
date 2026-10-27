from __future__ import annotations

import skalaxc
from typing_extensions import assert_type  # noqa: UP035

assert_type(skalaxc.RuntimeEnvironment(), skalaxc.RuntimeEnvironment)
skalaxc.RuntimeEnvironment(None)  # type: ignore[call-arg]
skalaxc.DeviceRuntimeSettings()  # type: ignore[attr-defined]

from __future__ import annotations

import skalaxc
from typing_extensions import assert_type  # noqa: UP035

settings = skalaxc.DeviceRuntimeSettings()
assert_type(skalaxc.RuntimeEnvironment(), skalaxc.RuntimeEnvironment)
assert_type(skalaxc.RuntimeEnvironment(settings), skalaxc.RuntimeEnvironment)
skalaxc.RuntimeEnvironment(None)  # type: ignore[call-overload]

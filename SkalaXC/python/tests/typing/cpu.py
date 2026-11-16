from __future__ import annotations

from typing_extensions import assert_type  # noqa: UP035

import skalaxc

assert_type(skalaxc.RuntimeEnvironment(), skalaxc.RuntimeEnvironment)
skalaxc.RuntimeEnvironment(None)  # type: ignore[call-arg]
skalaxc.DeviceRuntimeSettings()  # type: ignore[attr-defined]

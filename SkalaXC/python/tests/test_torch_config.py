from __future__ import annotations

import importlib.util
from pathlib import Path


def test_torch_configuration_matches_imported_torch() -> None:
    script = Path(__file__).parents[1] / "tools" / "torch_config.py"
    spec = importlib.util.spec_from_file_location("torch_config", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    configuration = module.torch_configuration()
    assert Path(configuration["torch_dir"], "TorchConfig.cmake").is_file()
    assert configuration["version"]
    assert configuration["cxx11_abi"] in (0, 1)
    assert configuration["cuda"] is None or configuration["cuda"]

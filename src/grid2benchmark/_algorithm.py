from __future__ import annotations

import importlib.util
import tempfile
from pathlib import Path
from types import ModuleType

from ._config import REQUIRED_ALGORITHM_FUNCTION


def load_algorithm(source_code: str) -> ModuleType:
    """Load a Grid2Op algorithm module from a source code string."""
    with tempfile.TemporaryDirectory(prefix="benchmark_algo_") as td:
        module_path = Path(td) / "submitted_algorithm.py"
        module_path.write_text(source_code, encoding="utf-8")

        spec = importlib.util.spec_from_file_location(
            "submitted_algorithm", module_path
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("Failed to create module spec for algorithm source")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


def load_algorithm_from_file(path: Path) -> ModuleType:
    """Load a Grid2Op algorithm module from a .py file."""
    return load_algorithm(Path(path).read_text(encoding="utf-8"))


def validate_algorithm(module: ModuleType) -> None:
    """Raise ValueError if the module does not expose build_agent(env, context)."""
    build_agent = getattr(module, REQUIRED_ALGORITHM_FUNCTION, None)
    if build_agent is None or not callable(build_agent):
        raise ValueError(
            f"Algorithm module must expose callable {REQUIRED_ALGORITHM_FUNCTION}(env, context)"
        )

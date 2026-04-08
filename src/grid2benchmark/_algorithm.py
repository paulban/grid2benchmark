"""Algorithm loading and validation utilities.

The benchmark accepts user-submitted agents as Python modules exposing a
``build_agent(env, context)`` function. This module provides helpers to:

- load source code into an importable module object,
- load source from a file,
- validate that the expected function exists and is callable.
"""

from __future__ import annotations

import importlib.util
import tempfile
from pathlib import Path
from types import ModuleType

from ._config import REQUIRED_ALGORITHM_FUNCTION


def load_algorithm(source_code: str) -> ModuleType:
    """Load a benchmark algorithm module from a source-code string.

    Args:
        source_code: Python source code containing ``build_agent``.

    Returns:
        Dynamically imported module object.

    Raises:
        RuntimeError: If import machinery cannot create a valid module spec.
        Exception: Any syntax/import/runtime error raised while executing the
            submitted module.
    """
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
    """Load a benchmark algorithm module from a Python file.

    Args:
        path: Filesystem path to the algorithm source file.

    Returns:
        Dynamically imported module object.
    """
    return load_algorithm(Path(path).read_text(encoding="utf-8"))


def validate_algorithm(module: ModuleType) -> None:
    """Validate required algorithm interface on the loaded module.

    Args:
        module: Imported module produced by :func:`load_algorithm`.

    Raises:
        ValueError: If the required ``build_agent(env, context)`` callable is
            missing.
    """
    build_agent = getattr(module, REQUIRED_ALGORITHM_FUNCTION, None)
    if build_agent is None or not callable(build_agent):
        raise ValueError(
            f"Algorithm module must expose callable {REQUIRED_ALGORITHM_FUNCTION}(env, context)"
        )

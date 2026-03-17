from __future__ import annotations

from pathlib import Path

from ._algorithm import load_algorithm, load_algorithm_from_file, validate_algorithm
from ._config import BenchmarkConfig
from ._runner import run_episodes

__all__ = ["run_benchmark", "BenchmarkConfig"]


def run_benchmark(
    algorithm: str | Path,
    config: BenchmarkConfig | None = None,
) -> dict:
    """Run a Grid2Op benchmark and return results with KPIs.

    Args:
        algorithm: Path to a .py file, or a source code string.
        config: Benchmark configuration. Uses defaults if omitted.

    Returns:
        dict with keys ``environment``, ``episodes``, and ``kpis``.
    """
    if config is None:
        config = BenchmarkConfig()

    if isinstance(algorithm, Path):
        module = load_algorithm_from_file(algorithm)
    else:
        module = load_algorithm(algorithm)

    validate_algorithm(module)
    return run_episodes(config, module)

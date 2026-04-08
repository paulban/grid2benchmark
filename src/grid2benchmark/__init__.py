"""Public package API for grid2benchmark.

This module exposes the high-level orchestration function used by external
services and scripts:

- :func:`run_benchmark`
- :class:`BenchmarkConfig`
- :class:`ScenarioConfig`
"""

from __future__ import annotations

from pathlib import Path

from ._algorithm import load_algorithm, load_algorithm_from_file, validate_algorithm
from ._config import BenchmarkConfig, ScenarioConfig
from ._runner import run_scenarios

__all__ = ["run_benchmark", "BenchmarkConfig", "ScenarioConfig"]


def run_benchmark(
    algorithm: str | Path,
    config: BenchmarkConfig | None = None,
) -> dict:
    """Run the benchmark for a submitted algorithm.

    Args:
        algorithm: Either a path to a Python file or a Python source-code string.
            If a :class:`pathlib.Path` is provided, the file is read from disk.
            If a string is provided, it is interpreted as source code.
        config: Benchmark configuration. If omitted, package defaults are used.

    Returns:
        JSON-serializable benchmark result with top-level keys:

        - ``scenarios``: list of per-scenario results
        - ``summary``: cross-scenario aggregate statistics
    """
    if config is None:
        config = BenchmarkConfig()

    if isinstance(algorithm, Path):
        module = load_algorithm_from_file(algorithm)
    else:
        module = load_algorithm(algorithm)

    validate_algorithm(module)
    return run_scenarios(config, module)

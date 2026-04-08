"""Configuration models and constants for grid2benchmark.

This module defines immutable dataclasses that represent benchmark execution
inputs and performs validation at construction time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_ENV_NAME = "l2rpn_case14_sandbox"
DEFAULT_MAX_STEPS = 200
REQUIRED_ALGORITHM_FUNCTION = "build_agent"
AVAILABLE_KPI_NAMES = (
    "survival",
    "violations",
    "latency",
    "carbon_intensity",
    "operation_score",
    "topological_action_complexity",
)
DEFAULT_KPIS = AVAILABLE_KPI_NAMES


@dataclass(frozen=True)
class ScenarioConfig:
    """Configuration for a single benchmark scenario.

    Attributes:
        env_name: Grid2Op environment name to load.
        time_series_ids: Optional list of time-series indices to execute.
            If ``None``, all available time series are executed.
        env_path: Optional dataset path override passed to ``grid2op.make``.
    """

    env_name: str = DEFAULT_ENV_NAME
    time_series_ids: tuple[int, ...] | None = None
    env_path: Path | None = None

    def __post_init__(self) -> None:
        """Normalize and validate scenario fields."""
        if not self.env_name:
            raise ValueError("env_name must be a non-empty string")

        if self.time_series_ids is not None:
            object.__setattr__(self, "time_series_ids", tuple(self.time_series_ids))
            if len(self.time_series_ids) == 0:
                raise ValueError("time_series_ids must not be empty when provided")
            if any(ts_id < 0 for ts_id in self.time_series_ids):
                raise ValueError("time_series_ids must contain non-negative integers")

        if self.env_path is not None:
            object.__setattr__(self, "env_path", Path(self.env_path))


@dataclass(frozen=True)
class BenchmarkConfig:
    """Top-level benchmark execution configuration.

    Attributes:
        scenarios: One or more :class:`ScenarioConfig` entries.
        max_steps: Maximum number of steps per episode.
        kpis: KPI names to evaluate.
    """

    scenarios: tuple[ScenarioConfig, ...] = field(
        default_factory=lambda: (ScenarioConfig(),)
    )
    max_steps: int = DEFAULT_MAX_STEPS
    kpis: tuple[str, ...] = DEFAULT_KPIS

    def __post_init__(self) -> None:
        """Normalize list-like inputs and validate constraints."""
        object.__setattr__(self, "scenarios", tuple(self.scenarios))
        object.__setattr__(self, "kpis", tuple(self.kpis))

        if self.max_steps <= 0:
            raise ValueError("max_steps must be > 0")

        if len(self.scenarios) == 0:
            raise ValueError("scenarios must contain at least one ScenarioConfig")

        invalid_kpis = [kpi for kpi in self.kpis if kpi not in AVAILABLE_KPI_NAMES]
        if invalid_kpis:
            raise ValueError(
                f"Unknown KPI(s): {invalid_kpis}. Allowed values: {AVAILABLE_KPI_NAMES}"
            )

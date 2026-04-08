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
    "carbon_intensity",
    "operation_score",
    "topological_action_complexity",
)
DEFAULT_KPIS = AVAILABLE_KPI_NAMES
SUPPORTED_TOPOLOGY_FORMATS = ("pandapower",)
SUPPORTED_TIME_SERIES_FORMATS = ("grid2op_chronics_dir",)
SUPPORTED_BACKENDS = ("pandapower", "lightsim2grid", "pypowsybl")


@dataclass(frozen=True)
class TopologySource:
    """Topology source description for one scenario.

    Attributes:
        format: Topology format identifier. Phase 1 supports ``pandapower``.
        path: Filesystem path to the topology file.
    """

    format: str
    path: Path

    def __post_init__(self) -> None:
        """Validate topology source format and path."""
        if self.format not in SUPPORTED_TOPOLOGY_FORMATS:
            raise ValueError(
                f"Unsupported topology format '{self.format}'. "
                f"Allowed values: {SUPPORTED_TOPOLOGY_FORMATS}"
            )

        object.__setattr__(self, "path", Path(self.path))
        if not self.path.exists():
            raise ValueError(f"Topology path does not exist: {self.path}")
        if not self.path.is_file():
            raise ValueError(f"Topology path must be a file: {self.path}")

        if self.format == "pandapower" and self.path.suffix.lower() != ".json":
            raise ValueError("Pandapower topology file must use .json extension")


@dataclass(frozen=True)
class TimeSeriesSource:
    """Time-series source description for one scenario.

    Attributes:
        format: Time-series format identifier. Phase 1 supports
            ``grid2op_chronics_dir``.
        path: Directory containing Grid2Op-compatible chronic files.
    """

    format: str
    path: Path

    def __post_init__(self) -> None:
        """Validate time-series source format and path."""
        if self.format not in SUPPORTED_TIME_SERIES_FORMATS:
            raise ValueError(
                f"Unsupported time series format '{self.format}'. "
                f"Allowed values: {SUPPORTED_TIME_SERIES_FORMATS}"
            )

        object.__setattr__(self, "path", Path(self.path))
        if not self.path.exists():
            raise ValueError(f"Time series path does not exist: {self.path}")
        if not self.path.is_dir():
            raise ValueError(f"Time series path must be a directory: {self.path}")


@dataclass(frozen=True)
class ScenarioConfig:
    """Configuration for a single benchmark scenario.

    Attributes:
        env_name: Grid2Op environment name to load.
        time_series_ids: Optional list of time-series indices to execute.
            If ``None``, all available time series are executed.
        topology: Optional topology source (for file-based scenario input).
        time_series: Optional time-series source (for file-based scenario input).
        backend: Optional backend simulator name. One of ``"pandapower"``,
            ``"lightsim2grid"``, or ``"pypowsybl"``. When ``None``, defaults to
            ``pandapower`` if a topology source is provided, otherwise Grid2Op's
            built-in default.
    """

    env_name: str = DEFAULT_ENV_NAME
    time_series_ids: tuple[int, ...] | None = None
    topology: TopologySource | None = None
    time_series: TimeSeriesSource | None = None
    backend: str | None = None

    def __post_init__(self) -> None:
        """Normalize and validate scenario fields."""
        if not self.env_name:
            raise ValueError("env_name must be a non-empty string")

        if self.backend is not None and self.backend not in SUPPORTED_BACKENDS:
            raise ValueError(
                f"Unsupported backend '{self.backend}'. "
                f"Allowed values: {SUPPORTED_BACKENDS}"
            )

        if self.time_series_ids is not None:
            object.__setattr__(self, "time_series_ids", tuple(self.time_series_ids))
            if len(self.time_series_ids) == 0:
                raise ValueError("time_series_ids must not be empty when provided")
            if any(ts_id < 0 for ts_id in self.time_series_ids):
                raise ValueError("time_series_ids must contain non-negative integers")


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

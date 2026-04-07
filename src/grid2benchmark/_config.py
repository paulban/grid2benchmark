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
    env_name: str = DEFAULT_ENV_NAME
    chronic_ids: tuple[int, ...] | None = None
    env_path: Path | None = None

    def __post_init__(self) -> None:
        if not self.env_name:
            raise ValueError("env_name must be a non-empty string")

        if self.chronic_ids is not None:
            object.__setattr__(self, "chronic_ids", tuple(self.chronic_ids))
            if len(self.chronic_ids) == 0:
                raise ValueError("chronic_ids must not be empty when provided")
            if any(chronic_id < 0 for chronic_id in self.chronic_ids):
                raise ValueError("chronic_ids must contain non-negative integers")

        if self.env_path is not None:
            object.__setattr__(self, "env_path", Path(self.env_path))


@dataclass(frozen=True)
class BenchmarkConfig:
    scenarios: tuple[ScenarioConfig, ...] = field(
        default_factory=lambda: (ScenarioConfig(),)
    )
    max_steps: int = DEFAULT_MAX_STEPS
    kpis: tuple[str, ...] = DEFAULT_KPIS

    def __post_init__(self) -> None:
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

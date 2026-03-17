from __future__ import annotations

from dataclasses import dataclass

DEFAULT_ENV_NAME = "l2rpn_case14_sandbox"
DEFAULT_MAX_STEPS = 200
DEFAULT_EPISODES = 1
REQUIRED_ALGORITHM_FUNCTION = "build_agent"


@dataclass(frozen=True)
class BenchmarkConfig:
    env_name: str = DEFAULT_ENV_NAME
    max_steps: int = DEFAULT_MAX_STEPS
    episodes: int = DEFAULT_EPISODES

    def __post_init__(self) -> None:
        if self.max_steps <= 0:
            raise ValueError("max_steps must be > 0")
        if self.episodes <= 0:
            raise ValueError("episodes must be > 0")

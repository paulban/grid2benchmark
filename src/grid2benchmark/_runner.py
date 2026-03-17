from __future__ import annotations

import inspect
import logging
import tempfile
import time
from pathlib import Path
from types import ModuleType
from typing import Any

from ._config import BenchmarkConfig, REQUIRED_ALGORITHM_FUNCTION
from ._kpi import evaluate_kpis

logger = logging.getLogger(__name__)


def _call_agent_act(agent: Any, observation: Any, reward: float, done: bool) -> Any:
    """Call agent.act() supporting 1-, 2-, and 3-parameter signatures."""
    act_fn = agent.act
    try:
        param_count = len(inspect.signature(act_fn).parameters)
    except (TypeError, ValueError):
        param_count = 1

    if param_count <= 1:
        return act_fn(observation)
    if param_count == 2:
        return act_fn(observation, reward)
    return act_fn(observation, reward, done)


def run_episodes(config: BenchmarkConfig, module: ModuleType) -> dict[str, Any]:
    """Run benchmark episodes with EnvRecorder and return results + KPIs."""
    import grid2op  # type: ignore
    from grid2op.Environment.EnvRecorder import EnvRecorder  # type: ignore

    env = grid2op.make(config.env_name, test=True)
    build_agent = getattr(module, REQUIRED_ALGORITHM_FUNCTION)

    agent_context = {
        "benchmark": {
            "env_name": config.env_name,
            "max_steps": config.max_steps,
            "episodes": config.episodes,
        },
    }

    with tempfile.TemporaryDirectory(prefix="benchmark_record_") as record_dir:
        record_path = Path(record_dir)

        with EnvRecorder(env, record_path) as env_rec:
            # Build agents against the original env so common baselines can
            # access attributes like action_space.
            agent = build_agent(env, agent_context)
            if not hasattr(agent, "act") or not callable(agent.act):
                raise ValueError("Agent must expose a callable act(observation) method")

            episode_results: list[dict[str, Any]] = []

            for episode_idx in range(config.episodes):
                reset_result = env_rec.reset()
                obs = (
                    reset_result[0] if isinstance(reset_result, tuple) else reset_result
                )

                done = False
                reward = 0.0
                steps = 0
                overload_violations = 0
                started = time.perf_counter()

                while not done and steps < config.max_steps:
                    action = _call_agent_act(agent, obs, reward, done)
                    step_result = env_rec.step(action)

                    if isinstance(step_result, tuple) and len(step_result) == 5:
                        obs, reward, terminated, truncated, info = step_result
                        done = bool(terminated or truncated)
                    else:
                        obs, reward, done, info = step_result

                    steps += 1
                    if isinstance(info, dict):
                        if info.get("is_illegal", False) or info.get(
                            "is_ambiguous", False
                        ):
                            overload_violations += 1

                episode_results.append(
                    {
                        "episode_index": episode_idx,
                        "steps": steps,
                        "overload_violations": overload_violations,
                        "runtime_seconds": time.perf_counter() - started,
                        "terminated": done,
                    }
                )

        kpis = evaluate_kpis(record_path, episode_results)

    return {
        "environment": {"env_name": config.env_name, "fixed_environment": True},
        "episodes": episode_results,
        "kpis": kpis,
    }

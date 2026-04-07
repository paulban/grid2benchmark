from __future__ import annotations

import inspect
import logging
import numbers
import tempfile
import time
from pathlib import Path
from types import ModuleType
from typing import Any

from ._config import BenchmarkConfig, REQUIRED_ALGORITHM_FUNCTION, ScenarioConfig
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


def _resolve_chronic_ids(env: Any, scenario: ScenarioConfig) -> list[int]:
    if scenario.chronic_ids is not None:
        return list(scenario.chronic_ids)

    available_chronics = env.chronics_handler.available_chronics()
    return list(range(len(available_chronics)))


def _make_env(grid2op_module: Any, scenario: ScenarioConfig) -> Any:
    make_kwargs: dict[str, Any] = {"test": True}
    if scenario.env_path is not None:
        make_kwargs["dataset_path"] = str(scenario.env_path)
    return grid2op_module.make(scenario.env_name, **make_kwargs)


def _run_episode(
    env_rec: Any,
    agent: Any,
    max_steps: int,
    episode_index: int,
    chronic_id: int,
) -> dict[str, Any]:
    reset_result = env_rec.reset(options={"time serie id": chronic_id})
    obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result

    done = False
    reward = 0.0
    steps = 0
    overload_violations = 0
    started = time.perf_counter()

    while not done and steps < max_steps:
        action = _call_agent_act(agent, obs, reward, done)
        step_result = env_rec.step(action)

        if isinstance(step_result, tuple) and len(step_result) == 5:
            obs, reward, terminated, truncated, info = step_result
            done = bool(terminated or truncated)
        else:
            obs, reward, done, info = step_result

        steps += 1
        if isinstance(info, dict):
            if info.get("is_illegal", False) or info.get("is_ambiguous", False):
                overload_violations += 1

    return {
        "episode_index": episode_index,
        "chronic_id": chronic_id,
        "steps": steps,
        "overload_violations": overload_violations,
        "runtime_seconds": time.perf_counter() - started,
        "terminated": done,
    }


def _extract_numeric_values(value: Any) -> list[float]:
    if isinstance(value, bool):
        return []
    if isinstance(value, numbers.Real):
        return [float(value)]
    if isinstance(value, list):
        out: list[float] = []
        for item in value:
            out.extend(_extract_numeric_values(item))
        return out
    if isinstance(value, dict):
        out = []
        for item in value.values():
            out.extend(_extract_numeric_values(item))
        return out
    return []


def _aggregate_summary(scenario_results: list[dict[str, Any]]) -> dict[str, Any]:
    per_key_values: dict[str, list[float]] = {}
    total_episodes = 0

    for scenario in scenario_results:
        total_episodes += len(scenario.get("episodes", []))
        kpis = scenario.get("kpis", {})
        if not isinstance(kpis, dict):
            continue
        for kpi_name, kpi_value in kpis.items():
            numeric_values = _extract_numeric_values(kpi_value)
            if not numeric_values:
                continue
            per_key_values.setdefault(kpi_name, []).extend(numeric_values)

    summary_kpis: dict[str, dict[str, float | int]] = {}
    for kpi_name, values in per_key_values.items():
        summary_kpis[kpi_name] = {
            "mean": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
            "count": len(values),
        }

    return {
        "scenario_count": len(scenario_results),
        "episode_count": total_episodes,
        "kpis": summary_kpis,
    }


def run_scenarios(config: BenchmarkConfig, module: ModuleType) -> dict[str, Any]:
    """Run all configured scenarios and return per-scenario results with summary."""
    import grid2op  # type: ignore
    from grid2op.Environment.EnvRecorder import EnvRecorder  # type: ignore

    build_agent = getattr(module, REQUIRED_ALGORITHM_FUNCTION)
    scenario_results: list[dict[str, Any]] = []

    for scenario_idx, scenario in enumerate(config.scenarios):
        env = _make_env(grid2op, scenario)
        chronic_ids = _resolve_chronic_ids(env, scenario)

        with tempfile.TemporaryDirectory(prefix="benchmark_record_") as record_dir:
            record_path = Path(record_dir)

            with EnvRecorder(env, record_path) as env_rec:
                agent_context = {
                    "benchmark": {
                        "max_steps": config.max_steps,
                        "kpis": list(config.kpis),
                        "scenario_index": scenario_idx,
                    },
                    "scenario": {
                        "env_name": scenario.env_name,
                        "env_path": (
                            str(scenario.env_path) if scenario.env_path else None
                        ),
                        "chronic_ids": chronic_ids,
                    },
                }

                # Build agents against the original environment so common
                # baselines can access attributes like action_space.
                agent = build_agent(env, agent_context)
                if not hasattr(agent, "act") or not callable(agent.act):
                    raise ValueError(
                        "Agent must expose a callable act(observation) method"
                    )

                episode_results = [
                    _run_episode(
                        env_rec=env_rec,
                        agent=agent,
                        max_steps=config.max_steps,
                        episode_index=episode_idx,
                        chronic_id=chronic_id,
                    )
                    for episode_idx, chronic_id in enumerate(chronic_ids)
                ]

            kpis = evaluate_kpis(record_path, episode_results, config.kpis)

        scenario_results.append(
            {
                "scenario_index": scenario_idx,
                "environment": {
                    "env_name": scenario.env_name,
                    "env_path": str(scenario.env_path) if scenario.env_path else None,
                    "fixed_environment": True,
                },
                "executed_chronic_ids": chronic_ids,
                "episodes": episode_results,
                "kpis": kpis,
            }
        )

    return {
        "scenarios": scenario_results,
        "summary": _aggregate_summary(scenario_results),
    }

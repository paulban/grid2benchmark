"""Integration tests — require a real Grid2Op environment and grid2evaluate.

These tests exercise the full pipeline: algorithm loading → episode execution
→ KPI computation → result structure validation.

They are marked with ``@pytest.mark.integration`` so they can be selectively
run or excluded:

    pytest -m integration           # run only integration tests
    pytest -m "not integration"     # skip them (fast unit-only run)

Each test is also given a generous timeout via ``pytest-timeout`` because
Grid2Op power-flow simulation is CPU-bound.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from grid2benchmark import BenchmarkConfig, ScenarioConfig, run_benchmark
from grid2benchmark._config import AVAILABLE_KPI_NAMES


pytestmark = pytest.mark.integration

ALGORITHM_TEMPLATE = Path(__file__).parent.parent / "examples" / "algorithm_template.py"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _default_config(**kwargs) -> BenchmarkConfig:
    """One scenario, time series 0 only, 5 steps — fast enough for CI."""
    base = {
        "scenarios": (
            ScenarioConfig(env_name="l2rpn_case14_sandbox", time_series_ids=(0,)),
        ),
        "max_steps": 5,
        "kpis": ("carbon_intensity",),
    }
    base.update(kwargs)
    return BenchmarkConfig(**base)


# ---------------------------------------------------------------------------
# Result structure
# ---------------------------------------------------------------------------


class TestResultStructure:
    @pytest.mark.timeout(120)
    def test_top_level_keys(self):
        result = run_benchmark(ALGORITHM_TEMPLATE, _default_config())
        assert set(result.keys()) == {"scenarios", "summary"}

    @pytest.mark.timeout(120)
    def test_scenario_keys_present(self):
        result = run_benchmark(ALGORITHM_TEMPLATE, _default_config())
        scenario = result["scenarios"][0]
        assert "scenario_index" in scenario
        assert "environment" in scenario
        assert "executed_time_series_ids" in scenario
        assert "episodes" in scenario
        assert "kpis" in scenario

    @pytest.mark.timeout(120)
    def test_episode_count_matches_time_series_ids(self):
        config = _default_config(
            scenarios=(
                ScenarioConfig(
                    env_name="l2rpn_case14_sandbox",
                    time_series_ids=(0, 1),
                ),
            ),
        )
        result = run_benchmark(ALGORITHM_TEMPLATE, config)
        assert len(result["scenarios"][0]["episodes"]) == 2

    @pytest.mark.timeout(120)
    def test_executed_time_series_ids_correct(self):
        config = _default_config(
            scenarios=(
                ScenarioConfig(
                    env_name="l2rpn_case14_sandbox",
                    time_series_ids=(0, 2),
                ),
            ),
        )
        result = run_benchmark(ALGORITHM_TEMPLATE, config)
        assert result["scenarios"][0]["executed_time_series_ids"] == [0, 2]

    @pytest.mark.timeout(120)
    def test_episode_fields(self):
        result = run_benchmark(ALGORITHM_TEMPLATE, _default_config())
        ep = result["scenarios"][0]["episodes"][0]
        assert "steps" in ep
        assert "overload_violations" in ep
        assert "runtime_seconds" in ep
        assert "terminated" in ep
        assert "time_series_id" in ep

    @pytest.mark.timeout(120)
    def test_steps_bounded_by_max_steps(self):
        result = run_benchmark(ALGORITHM_TEMPLATE, _default_config())
        ep = result["scenarios"][0]["episodes"][0]
        assert ep["steps"] <= 5

    @pytest.mark.timeout(120)
    def test_summary_keys_present(self):
        result = run_benchmark(ALGORITHM_TEMPLATE, _default_config())
        summary = result["summary"]
        assert "scenario_count" in summary
        assert "episode_count" in summary
        assert "kpis" in summary

    @pytest.mark.timeout(120)
    def test_summary_scenario_count(self):
        result = run_benchmark(ALGORITHM_TEMPLATE, _default_config())
        assert result["summary"]["scenario_count"] == 1

    @pytest.mark.timeout(120)
    def test_summary_kpi_stats_present(self):
        result = run_benchmark(ALGORITHM_TEMPLATE, _default_config())
        # grid2evaluate KPIs are intentionally not flattened into summary stats.
        assert "carbon_intensity" not in result["summary"]["kpis"]


# ---------------------------------------------------------------------------
# KPI selection
# ---------------------------------------------------------------------------


class TestKpiSelection:
    @pytest.mark.timeout(120)
    def test_only_selected_kpis_present(self):
        config = _default_config(kpis=("carbon_intensity",))
        result = run_benchmark(ALGORITHM_TEMPLATE, config)
        kpis = result["scenarios"][0]["kpis"]
        assert "carbon_intensity" in kpis
        assert "operation_score" not in kpis
        assert "topological_action_complexity" not in kpis

    @pytest.mark.timeout(120)
    def test_evaluation_backend_is_grid2evaluate(self):
        config = _default_config(kpis=("carbon_intensity",))
        result = run_benchmark(ALGORITHM_TEMPLATE, config)
        assert result["scenarios"][0]["kpis"]["evaluation_backend"] == "grid2evaluate"

    @pytest.mark.timeout(120)
    def test_carbon_intensity_triggers_g2e(self):
        config = _default_config(kpis=("carbon_intensity",))
        result = run_benchmark(ALGORITHM_TEMPLATE, config)
        kpis = result["scenarios"][0]["kpis"]
        backend = kpis["evaluation_backend"]
        assert backend == "grid2evaluate"

    @pytest.mark.timeout(120)
    def test_invalid_kpi_rejected(self):
        with pytest.raises(ValueError, match="Unknown KPI"):
            _default_config(kpis=("carbon_intensity", "imaginary_kpi"))


# ---------------------------------------------------------------------------
# Multi-scenario
# ---------------------------------------------------------------------------


class TestMultiScenario:
    @pytest.mark.timeout(240)
    def test_two_scenarios_produce_two_results(self):
        config = BenchmarkConfig(
            scenarios=(
                ScenarioConfig(env_name="l2rpn_case14_sandbox", time_series_ids=(0,)),
                ScenarioConfig(env_name="l2rpn_case14_sandbox", time_series_ids=(1,)),
            ),
            max_steps=5,
            kpis=("carbon_intensity",),
        )
        result = run_benchmark(ALGORITHM_TEMPLATE, config)
        assert len(result["scenarios"]) == 2
        assert result["summary"]["scenario_count"] == 2
        assert result["summary"]["episode_count"] == 2

    @pytest.mark.timeout(240)
    def test_scenario_indices_sequential(self):
        config = BenchmarkConfig(
            scenarios=(
                ScenarioConfig(env_name="l2rpn_case14_sandbox", time_series_ids=(0,)),
                ScenarioConfig(env_name="l2rpn_case14_sandbox", time_series_ids=(1,)),
            ),
            max_steps=5,
            kpis=("carbon_intensity",),
        )
        result = run_benchmark(ALGORITHM_TEMPLATE, config)
        assert result["scenarios"][0]["scenario_index"] == 0
        assert result["scenarios"][1]["scenario_index"] == 1


# ---------------------------------------------------------------------------
# Algorithm API (context propagation)
# ---------------------------------------------------------------------------


_CONTEXT_CAPTURE_SRC = """
captured = {}

class _Agent:
    def __init__(self, action_space, context):
        self._action_space = action_space

    def act(self, observation):
        return self._action_space()

def build_agent(env, context):
    captured.update(context)
    return _Agent(env.action_space, context)
"""


class TestAlgorithmContext:
    @pytest.mark.timeout(120)
    def test_context_has_benchmark_and_scenario_keys(self):
        from grid2benchmark._algorithm import load_algorithm, validate_algorithm
        from grid2benchmark._runner import run_scenarios

        module = load_algorithm(_CONTEXT_CAPTURE_SRC)
        validate_algorithm(module)

        config = _default_config(kpis=("carbon_intensity",))
        run_scenarios(config, module)

        captured = module.captured
        assert "benchmark" in captured
        assert "scenario" in captured
        assert "max_steps" in captured["benchmark"]
        assert "env_name" in captured["scenario"]

    @pytest.mark.timeout(120)
    def test_context_time_series_ids_matches_execution(self):
        from grid2benchmark._algorithm import load_algorithm, validate_algorithm
        from grid2benchmark._runner import run_scenarios

        module = load_algorithm(_CONTEXT_CAPTURE_SRC)
        validate_algorithm(module)

        config = BenchmarkConfig(
            scenarios=(
                ScenarioConfig(
                    env_name="l2rpn_case14_sandbox",
                    time_series_ids=(0, 1),
                ),
            ),
            max_steps=5,
            kpis=("carbon_intensity",),
        )
        run_scenarios(config, module)

        time_series_ids = module.captured["scenario"]["time_series_ids"]
        assert time_series_ids == [0, 1]


# ---------------------------------------------------------------------------
# JSON serialisability (important for orchestrator integration)
# ---------------------------------------------------------------------------


class TestJsonSerialisation:
    @pytest.mark.timeout(120)
    def test_result_is_json_serialisable(self):
        result = run_benchmark(ALGORITHM_TEMPLATE, _default_config())
        # Must not raise
        serialised = json.dumps(result)
        parsed = json.loads(serialised)
        assert "scenarios" in parsed
        assert "summary" in parsed

    @pytest.mark.timeout(120)
    def test_all_episode_fields_are_json_safe(self):
        result = run_benchmark(ALGORITHM_TEMPLATE, _default_config())
        for scenario in result["scenarios"]:
            for ep in scenario["episodes"]:
                json.dumps(ep)  # must not raise

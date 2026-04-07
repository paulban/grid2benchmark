"""Unit tests for ScenarioConfig and BenchmarkConfig."""

from __future__ import annotations

from pathlib import Path

import pytest

from grid2benchmark._config import (
    AVAILABLE_KPI_NAMES,
    DEFAULT_ENV_NAME,
    DEFAULT_MAX_STEPS,
    BenchmarkConfig,
    ScenarioConfig,
)


# ---------------------------------------------------------------------------
# ScenarioConfig
# ---------------------------------------------------------------------------


class TestScenarioConfigDefaults:
    def test_default_env_name(self):
        sc = ScenarioConfig()
        assert sc.env_name == DEFAULT_ENV_NAME

    def test_default_time_series_ids_is_none(self):
        sc = ScenarioConfig()
        assert sc.time_series_ids is None

    def test_default_env_path_is_none(self):
        sc = ScenarioConfig()
        assert sc.env_path is None


class TestScenarioConfigValidation:
    def test_empty_env_name_raises(self):
        with pytest.raises(ValueError, match="env_name"):
            ScenarioConfig(env_name="")

    def test_empty_time_series_ids_list_raises(self):
        with pytest.raises(ValueError, match="time_series_ids must not be empty"):
            ScenarioConfig(time_series_ids=())

    def test_negative_time_series_id_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            ScenarioConfig(time_series_ids=(-1,))

    def test_valid_time_series_ids_accepted(self):
        sc = ScenarioConfig(time_series_ids=(0, 1, 2))
        assert sc.time_series_ids == (0, 1, 2)

    def test_list_time_series_ids_coerced_to_tuple(self):
        sc = ScenarioConfig(time_series_ids=[0, 1])  # type: ignore[arg-type]
        assert isinstance(sc.time_series_ids, tuple)
        assert sc.time_series_ids == (0, 1)

    def test_env_path_string_coerced_to_path(self):
        sc = ScenarioConfig(env_path="/some/path")  # type: ignore[arg-type]
        assert isinstance(sc.env_path, Path)

    def test_env_path_path_accepted(self):
        sc = ScenarioConfig(env_path=Path("/some/path"))
        assert sc.env_path == Path("/some/path")

    def test_is_frozen(self):
        sc = ScenarioConfig()
        with pytest.raises((AttributeError, TypeError)):
            sc.env_name = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# BenchmarkConfig
# ---------------------------------------------------------------------------


class TestBenchmarkConfigDefaults:
    def test_default_max_steps(self):
        bc = BenchmarkConfig()
        assert bc.max_steps == DEFAULT_MAX_STEPS

    def test_default_kpis_is_all(self):
        bc = BenchmarkConfig()
        assert set(bc.kpis) == set(AVAILABLE_KPI_NAMES)

    def test_default_scenarios_has_one_entry(self):
        bc = BenchmarkConfig()
        assert len(bc.scenarios) == 1
        assert bc.scenarios[0].env_name == DEFAULT_ENV_NAME


class TestBenchmarkConfigValidation:
    def test_max_steps_zero_raises(self):
        with pytest.raises(ValueError, match="max_steps"):
            BenchmarkConfig(max_steps=0)

    def test_max_steps_negative_raises(self):
        with pytest.raises(ValueError, match="max_steps"):
            BenchmarkConfig(max_steps=-5)

    def test_empty_scenarios_raises(self):
        with pytest.raises(ValueError, match="scenarios"):
            BenchmarkConfig(scenarios=())

    def test_unknown_kpi_raises(self):
        with pytest.raises(ValueError, match="Unknown KPI"):
            BenchmarkConfig(kpis=("survival", "nonexistent_kpi"))

    def test_valid_single_kpi_accepted(self):
        bc = BenchmarkConfig(kpis=("survival",))
        assert bc.kpis == ("survival",)

    def test_list_scenarios_coerced_to_tuple(self):
        bc = BenchmarkConfig(scenarios=[ScenarioConfig()])  # type: ignore[arg-type]
        assert isinstance(bc.scenarios, tuple)

    def test_list_kpis_coerced_to_tuple(self):
        bc = BenchmarkConfig(kpis=["survival", "latency"])  # type: ignore[arg-type]
        assert isinstance(bc.kpis, tuple)

    def test_multiple_scenarios_accepted(self):
        bc = BenchmarkConfig(
            scenarios=[
                ScenarioConfig(env_name="l2rpn_case14_sandbox", time_series_ids=(0,)),
                ScenarioConfig(env_name="l2rpn_case14_sandbox", time_series_ids=(1,)),
            ]
        )
        assert len(bc.scenarios) == 2

    def test_is_frozen(self):
        bc = BenchmarkConfig()
        with pytest.raises((AttributeError, TypeError)):
            bc.max_steps = 1  # type: ignore[misc]


class TestAvailableKpiNames:
    def test_known_manual_kpis_present(self):
        for name in ("survival", "violations", "latency"):
            assert name in AVAILABLE_KPI_NAMES

    def test_known_g2e_kpis_present(self):
        for name in (
            "carbon_intensity",
            "operation_score",
            "topological_action_complexity",
        ):
            assert name in AVAILABLE_KPI_NAMES

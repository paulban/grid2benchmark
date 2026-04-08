"""Unit tests for runner helper functions that do not need Grid2Op.

The full simulation loop is covered by integration tests.
"""

from __future__ import annotations

from typing import Any

import pytest

from grid2benchmark._runner import _aggregate_summary, _extract_numeric_values


# ---------------------------------------------------------------------------
# _extract_numeric_values
# ---------------------------------------------------------------------------


class TestExtractNumericValues:
    def test_scalar_int(self):
        assert _extract_numeric_values(5) == [5.0]

    def test_scalar_float(self):
        assert _extract_numeric_values(3.14) == pytest.approx([3.14])

    def test_bool_excluded(self):
        assert _extract_numeric_values(True) == []
        assert _extract_numeric_values(False) == []

    def test_flat_list(self):
        assert _extract_numeric_values([1, 2, 3]) == [1.0, 2.0, 3.0]

    def test_nested_list(self):
        assert _extract_numeric_values([[1, 2], [3]]) == [1.0, 2.0, 3.0]

    def test_dict_values_extracted(self):
        vals = _extract_numeric_values({"a": 1.0, "b": 2.0})
        assert sorted(vals) == [1.0, 2.0]

    def test_mixed_dict_ignores_bools(self):
        vals = _extract_numeric_values({"flag": True, "score": 5.0})
        assert vals == [5.0]

    def test_non_numeric_string_ignored(self):
        assert _extract_numeric_values("hello") == []

    def test_none_ignored(self):
        assert _extract_numeric_values(None) == []

    def test_nested_structure(self):
        data: Any = {
            "survival": {"average_episode_length": 50.0, "episode_lengths": [50]}
        }
        result = _extract_numeric_values(data)
        assert 50.0 in result


# ---------------------------------------------------------------------------
# _aggregate_summary
# ---------------------------------------------------------------------------


def _make_scenario_result(
    scenario_index: int, episodes: list[dict], kpis: dict
) -> dict:
    return {
        "scenario_index": scenario_index,
        "environment": {
            "env_name": "env",
            "topology": None,
            "time_series": None,
            "fixed_environment": True,
        },
        "executed_time_series_ids": list(range(len(episodes))),
        "episodes": episodes,
        "kpis": kpis,
    }


def _episode(steps: int = 50) -> dict:
    return {
        "episode_index": 0,
        "time_series_id": 0,
        "steps": steps,
        "overload_violations": 0,
        "runtime_seconds": 0.1,
        "terminated": False,
    }


class TestAggregateSummary:
    def test_scenario_count(self):
        results = [
            _make_scenario_result(0, [_episode()], {}),
            _make_scenario_result(1, [_episode()], {}),
        ]
        summary = _aggregate_summary(results)
        assert summary["scenario_count"] == 2

    def test_episode_count(self):
        results = [
            _make_scenario_result(0, [_episode(), _episode()], {}),
            _make_scenario_result(1, [_episode()], {}),
        ]
        summary = _aggregate_summary(results)
        assert summary["episode_count"] == 3

    def test_scalar_kpi_aggregated(self):
        results = [
            _make_scenario_result(0, [_episode()], {"score": 10.0}),
            _make_scenario_result(1, [_episode()], {"score": 20.0}),
        ]
        summary = _aggregate_summary(results)
        assert summary["kpis"]["score"]["mean"] == pytest.approx(15.0)
        assert summary["kpis"]["score"]["min"] == pytest.approx(10.0)
        assert summary["kpis"]["score"]["max"] == pytest.approx(20.0)
        assert summary["kpis"]["score"]["count"] == 2

    def test_list_kpi_aggregated(self):
        results = [
            _make_scenario_result(
                0,
                [_episode()],
                {"survival": {"average_episode_length": 50.0, "episode_lengths": [50]}},
            ),
            _make_scenario_result(
                1,
                [_episode()],
                {
                    "survival": {
                        "average_episode_length": 100.0,
                        "episode_lengths": [100],
                    }
                },
            ),
        ]
        summary = _aggregate_summary(results)
        assert "survival" in summary["kpis"]
        assert summary["kpis"]["survival"]["min"] == pytest.approx(50.0)
        assert summary["kpis"]["survival"]["max"] == pytest.approx(100.0)

    def test_string_kpi_keys_ignored(self):
        results = [
            _make_scenario_result(
                0, [_episode()], {"evaluation_backend": "manual_only"}
            ),
        ]
        summary = _aggregate_summary(results)
        # String value "manual_only" yields no numeric values → key absent
        assert "evaluation_backend" not in summary["kpis"]

    def test_grid2evaluate_kpis_not_aggregated(self):
        results = [
            _make_scenario_result(
                0,
                [_episode()],
                {
                    "carbon_intensity": {
                        "per_step": [100.0, 120.0],
                        "metadata": {"unit": "gCO2eq/kWh"},
                    }
                },
            )
        ]
        summary = _aggregate_summary(results)
        assert "carbon_intensity" not in summary["kpis"]

    def test_empty_scenario_results(self):
        summary = _aggregate_summary([])
        assert summary["scenario_count"] == 0
        assert summary["episode_count"] == 0
        assert summary["kpis"] == {}

    def test_missing_kpis_key_tolerated(self):
        results = [{"scenario_index": 0, "episodes": [_episode()]}]
        summary = _aggregate_summary(results)
        assert summary["episode_count"] == 1
        assert summary["kpis"] == {}

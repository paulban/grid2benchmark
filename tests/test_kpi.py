"""Unit tests for KPI evaluation — manual KPIs and filtering logic.

grid2evaluate integration is only exercised in integration tests because it
requires a real recorded environment directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from grid2benchmark._kpi import _compute_manual_kpis, evaluate_kpis


def _make_episodes(*steps_list: int) -> list[dict[str, Any]]:
    return [
        {
            "episode_index": i,
            "time_series_id": i,
            "steps": s,
            "overload_violations": i,
            "runtime_seconds": 0.1 * (i + 1),
            "terminated": s < 200,
        }
        for i, s in enumerate(steps_list)
    ]


class TestComputeManualKpis:
    def test_survival_computed(self):
        episodes = _make_episodes(50, 100)
        result = _compute_manual_kpis(episodes, {"survival"})
        assert "survival" in result
        assert result["survival"]["average_episode_length"] == 75.0
        assert result["survival"]["episode_lengths"] == [50, 100]

    def test_violations_computed(self):
        episodes = _make_episodes(50, 100)
        result = _compute_manual_kpis(episodes, {"violations"})
        assert "violations" in result
        assert result["violations"]["total_overload_violations"] == 1
        assert result["violations"]["overload_violations_per_episode"] == [0, 1]

    def test_latency_computed(self):
        episodes = _make_episodes(50, 100)
        result = _compute_manual_kpis(episodes, {"latency"})
        assert "latency" in result
        assert result["latency"]["total_runtime_seconds"] == pytest.approx(0.3)
        assert result["latency"]["average_runtime_seconds"] == pytest.approx(0.15)

    def test_excluded_kpi_absent(self):
        episodes = _make_episodes(50)
        result = _compute_manual_kpis(episodes, {"survival"})
        assert "violations" not in result
        assert "latency" not in result

    def test_all_kpis_selected(self):
        episodes = _make_episodes(50, 100, 150)
        result = _compute_manual_kpis(episodes, {"survival", "violations", "latency"})
        assert "survival" in result
        assert "violations" in result
        assert "latency" in result

    def test_empty_selection_returns_empty(self):
        episodes = _make_episodes(50)
        result = _compute_manual_kpis(episodes, set())
        assert result == {}


class TestEvaluateKpisManualOnly:
    def test_only_manual_kpis_backend_marked(self, tmp_path: Path):
        episodes = _make_episodes(50)
        result = evaluate_kpis(tmp_path, episodes, ("survival",))
        assert result["evaluation_backend"] == "manual_only"

    def test_survival_only_no_g2e_keys(self, tmp_path: Path):
        episodes = _make_episodes(50)
        result = evaluate_kpis(tmp_path, episodes, ("survival",))
        assert "survival" in result
        assert "carbon_intensity" not in result
        assert "operation_score" not in result

    def test_all_manual_kpis_present(self, tmp_path: Path):
        episodes = _make_episodes(50, 100)
        result = evaluate_kpis(
            tmp_path, episodes, ("survival", "violations", "latency")
        )
        assert "survival" in result
        assert "violations" in result
        assert "latency" in result

    def test_unknown_kpi_raises(self, tmp_path: Path):
        with pytest.raises(ValueError, match="Unknown KPI"):
            evaluate_kpis(tmp_path, _make_episodes(50), ("nonexistent",))


class TestEvaluateKpisG2eFiltering:
    """Verify that grid2evaluate calls are skipped for KPIs not in the selection."""

    def test_g2e_not_called_when_not_selected(self, tmp_path: Path):
        episodes = _make_episodes(50)
        # Patch the import so we can verify it's never called
        kpi_mock = MagicMock()
        with patch.dict(
            "sys.modules",
            {
                "grid2evaluate": MagicMock(),
                "grid2evaluate.carbon_intensity_kpi": MagicMock(
                    CarbonIntensityKpi=kpi_mock
                ),
                "grid2evaluate.operation_score_kpi": MagicMock(
                    OperationScoreKpi=kpi_mock
                ),
                "grid2evaluate.topological_action_complexity_kpi": MagicMock(
                    TopologicalActionComplexityKpi=kpi_mock
                ),
            },
        ):
            result = evaluate_kpis(tmp_path, episodes, ("survival",))
        kpi_mock.return_value.evaluate.assert_not_called()
        assert result["evaluation_backend"] == "manual_only"

    def test_g2e_called_when_carbon_intensity_selected(self, tmp_path: Path):
        episodes = _make_episodes(50)
        ci_mock = MagicMock()
        ci_mock.return_value.evaluate.return_value = [100.0]
        os_mock = MagicMock()
        tc_mock = MagicMock()
        with patch.dict(
            "sys.modules",
            {
                "grid2evaluate": MagicMock(),
                "grid2evaluate.carbon_intensity_kpi": MagicMock(
                    CarbonIntensityKpi=ci_mock
                ),
                "grid2evaluate.operation_score_kpi": MagicMock(
                    OperationScoreKpi=os_mock
                ),
                "grid2evaluate.topological_action_complexity_kpi": MagicMock(
                    TopologicalActionComplexityKpi=tc_mock
                ),
            },
        ):
            result = evaluate_kpis(tmp_path, episodes, ("carbon_intensity",))
        ci_mock.return_value.evaluate.assert_called_once_with(tmp_path)
        os_mock.return_value.evaluate.assert_not_called()
        assert "carbon_intensity" in result

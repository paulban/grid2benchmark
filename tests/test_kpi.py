"""Unit tests for KPI evaluation in grid2evaluate-only mode."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from grid2benchmark._kpi import evaluate_kpis


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


class TestEvaluateKpis:
    def test_unknown_kpi_raises(self, tmp_path: Path):
        with pytest.raises(ValueError, match="Unknown KPI"):
            evaluate_kpis(tmp_path, _make_episodes(50), ("nonexistent",))

    def test_requires_grid2evaluate(self, tmp_path: Path):
        episodes = _make_episodes(50)
        with patch.dict(
            "sys.modules",
            {
                "grid2evaluate": None,
                "grid2evaluate.carbon_intensity_kpi": None,
                "grid2evaluate.operation_score_kpi": None,
                "grid2evaluate.topological_action_complexity_kpi": None,
            },
        ):
            with pytest.raises(RuntimeError, match="grid2evaluate is required"):
                evaluate_kpis(tmp_path, episodes, ("carbon_intensity",))

    def test_returns_raw_payload_and_backend_marker(self, tmp_path: Path):
        episodes = _make_episodes(50)
        ci_mock = MagicMock()
        ci_payload = {
            "per_step": [100.0, 120.0],
            "meta": {"unit": "gCO2eq/kWh"},
        }
        ci_mock.return_value.evaluate.return_value = ci_payload
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
        tc_mock.return_value.evaluate.assert_not_called()
        assert result["carbon_intensity"] == ci_payload
        assert result["evaluation_backend"] == "grid2evaluate"

    def test_selected_kpi_failure_raises(self, tmp_path: Path):
        episodes = _make_episodes(50)
        ci_mock = MagicMock()
        ci_mock.return_value.evaluate.side_effect = RuntimeError("boom")
        with patch.dict(
            "sys.modules",
            {
                "grid2evaluate": MagicMock(),
                "grid2evaluate.carbon_intensity_kpi": MagicMock(
                    CarbonIntensityKpi=ci_mock
                ),
                "grid2evaluate.operation_score_kpi": MagicMock(
                    OperationScoreKpi=MagicMock()
                ),
                "grid2evaluate.topological_action_complexity_kpi": MagicMock(
                    TopologicalActionComplexityKpi=MagicMock()
                ),
            },
        ):
            with pytest.raises(RuntimeError, match="KPI evaluation failed"):
                evaluate_kpis(tmp_path, episodes, ("carbon_intensity",))

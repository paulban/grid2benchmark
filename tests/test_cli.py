"""Unit tests for CLI helper functions (no actual benchmark execution needed)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from grid2benchmark._config import DEFAULT_ENV_NAME, DEFAULT_KPIS, ScenarioConfig
from grid2benchmark.cli import _load_scenarios, _parse_time_series_ids, _parse_kpis


# ---------------------------------------------------------------------------
# _parse_time_series_ids
# ---------------------------------------------------------------------------


class TestParseTimeSeriesIds:
    def test_none_returns_none(self):
        assert _parse_time_series_ids(None) is None

    def test_single_id(self):
        assert _parse_time_series_ids("0") == (0,)

    def test_multiple_ids(self):
        assert _parse_time_series_ids("0,1,2") == (0, 1, 2)

    def test_whitespace_stripped(self):
        assert _parse_time_series_ids(" 0 , 1 , 2 ") == (0, 1, 2)

    def test_empty_string_returns_none(self):
        assert _parse_time_series_ids("") is None

    def test_non_integer_raises(self):
        with pytest.raises(ValueError):
            _parse_time_series_ids("a,b")


# ---------------------------------------------------------------------------
# _parse_kpis
# ---------------------------------------------------------------------------


class TestParseKpis:
    def test_none_returns_all_defaults(self):
        result = _parse_kpis(None)
        assert set(result) == set(DEFAULT_KPIS)

    def test_single_kpi(self):
        assert _parse_kpis("survival") == ("survival",)

    def test_multiple_kpis(self):
        assert set(_parse_kpis("survival,carbon_intensity")) == {
            "survival",
            "carbon_intensity",
        }

    def test_whitespace_stripped(self):
        assert _parse_kpis(" survival , latency ") == ("survival", "latency")

    def test_empty_string_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            _parse_kpis("")


# ---------------------------------------------------------------------------
# _load_scenarios
# ---------------------------------------------------------------------------


class TestLoadScenarios:
    def _write(self, tmp_path: Path, content: object) -> Path:
        f = tmp_path / "scenarios.json"
        f.write_text(json.dumps(content), encoding="utf-8")
        return f

    def test_list_of_dicts(self, tmp_path: Path):
        f = self._write(tmp_path, [{"env_name": "l2rpn_case14_sandbox"}])
        scenarios = _load_scenarios(f)
        assert len(scenarios) == 1
        assert scenarios[0].env_name == "l2rpn_case14_sandbox"

    def test_wrapped_dict_format(self, tmp_path: Path):
        f = self._write(
            tmp_path,
            {"scenarios": [{"env_name": "l2rpn_case14_sandbox"}]},
        )
        scenarios = _load_scenarios(f)
        assert len(scenarios) == 1

    def test_defaults_used_when_keys_absent(self, tmp_path: Path):
        f = self._write(tmp_path, [{}])
        scenarios = _load_scenarios(f)
        assert scenarios[0].env_name == DEFAULT_ENV_NAME
        assert scenarios[0].time_series_ids is None
        assert scenarios[0].env_path is None

    def test_time_series_ids_parsed(self, tmp_path: Path):
        f = self._write(
            tmp_path,
            [{"env_name": "l2rpn_case14_sandbox", "time_series_ids": [0, 1, 2]}],
        )
        scenarios = _load_scenarios(f)
        assert scenarios[0].time_series_ids == (0, 1, 2)

    def test_env_path_parsed(self, tmp_path: Path):
        f = self._write(
            tmp_path,
            [{"env_name": "l2rpn_case14_sandbox", "env_path": "/some/grid/path"}],
        )
        scenarios = _load_scenarios(f)
        assert scenarios[0].env_path == Path("/some/grid/path")

    def test_multiple_scenarios(self, tmp_path: Path):
        f = self._write(
            tmp_path,
            [
                {"env_name": "l2rpn_case14_sandbox", "time_series_ids": [0]},
                {"env_name": "l2rpn_case14_sandbox", "time_series_ids": [1]},
            ],
        )
        scenarios = _load_scenarios(f)
        assert len(scenarios) == 2
        assert scenarios[0].time_series_ids == (0,)
        assert scenarios[1].time_series_ids == (1,)

    def test_non_list_payload_raises(self, tmp_path: Path):
        f = self._write(tmp_path, {"env_name": "l2rpn_case14_sandbox"})
        with pytest.raises(ValueError, match="list"):
            _load_scenarios(f)

    def test_non_dict_item_raises(self, tmp_path: Path):
        f = self._write(tmp_path, ["not_a_dict"])
        with pytest.raises(ValueError, match="JSON object"):
            _load_scenarios(f)

    def test_non_list_time_series_ids_raises(self, tmp_path: Path):
        f = self._write(
            tmp_path,
            [{"env_name": "l2rpn_case14_sandbox", "time_series_ids": 7}],
        )
        with pytest.raises(ValueError, match="time_series_ids must be a list"):
            _load_scenarios(f)

    def test_returns_tuple_of_scenario_configs(self, tmp_path: Path):
        f = self._write(tmp_path, [{}])
        result = _load_scenarios(f)
        assert isinstance(result, tuple)
        assert all(isinstance(s, ScenarioConfig) for s in result)

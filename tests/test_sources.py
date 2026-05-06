"""Unit tests for topology/time-series source adapters."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from grid2benchmark._config import ScenarioConfig, TimeSeriesSource, TopologySource
from grid2benchmark._converter import EnvDirResult
from grid2benchmark._sources import (
    _resolve_backend,
    build_make_kwargs,
    prepare_scenario,
)


class TestBuildMakeKwargs:
    def test_defaults_only_test_flag(self):
        scenario = ScenarioConfig(env_name="l2rpn_case14_sandbox")
        kwargs = build_make_kwargs(scenario)
        assert kwargs == {"test": True}

    def test_with_pandapower_topology(self, tmp_path: Path):
        topo_file = tmp_path / "grid.json"
        topo_file.write_text("{}", encoding="utf-8")

        scenario = ScenarioConfig(
            env_name="l2rpn_case14_sandbox",
            topology=TopologySource(format="pandapower", path=topo_file),
        )

        kwargs = build_make_kwargs(scenario)
        assert kwargs["test"] is True
        assert kwargs["grid_path"] == str(topo_file)
        assert "backend" in kwargs

    def test_with_chronics_directory(self, tmp_path: Path):
        chronics_dir = tmp_path / "chronics"
        chronics_dir.mkdir()

        scenario = ScenarioConfig(
            env_name="l2rpn_case14_sandbox",
            time_series=TimeSeriesSource(
                format="grid2op_chronics_dir",
                path=chronics_dir,
            ),
        )

        kwargs = build_make_kwargs(scenario)
        assert kwargs["test"] is True
        assert kwargs["chronics_path"] == str(chronics_dir)

    def test_with_topology_and_chronics(self, tmp_path: Path):
        topo_file = tmp_path / "grid.json"
        topo_file.write_text("{}", encoding="utf-8")
        chronics_dir = tmp_path / "chronics"
        chronics_dir.mkdir()

        scenario = ScenarioConfig(
            env_name="l2rpn_case14_sandbox",
            topology=TopologySource(format="pandapower", path=topo_file),
            time_series=TimeSeriesSource(
                format="grid2op_chronics_dir",
                path=chronics_dir,
            ),
        )

        kwargs = build_make_kwargs(scenario)
        assert kwargs["grid_path"] == str(topo_file)
        assert kwargs["chronics_path"] == str(chronics_dir)


class TestResolveBackend:
    def test_none_backend_no_topology_returns_none(self):
        scenario = ScenarioConfig(env_name="l2rpn_case14_sandbox")
        backend = _resolve_backend(scenario)
        assert backend is None

    def test_none_backend_with_topology_returns_pandapower(self, tmp_path: Path):
        topo_file = tmp_path / "grid.json"
        topo_file.write_text("{}", encoding="utf-8")
        scenario = ScenarioConfig(
            env_name="l2rpn_case14_sandbox",
            topology=TopologySource(format="pandapower", path=topo_file),
        )
        backend = _resolve_backend(scenario)
        assert backend is not None
        assert type(backend).__name__ == "PandaPowerBackend"

    def test_explicit_pandapower_backend(self):
        scenario = ScenarioConfig(
            env_name="l2rpn_case14_sandbox",
            backend="pandapower",
        )
        backend = _resolve_backend(scenario)
        assert backend is not None
        assert type(backend).__name__ == "PandaPowerBackend"

    def test_lightsim2grid_backend_missing_raises_import_error(self):
        scenario = ScenarioConfig(
            env_name="l2rpn_case14_sandbox",
            backend="lightsim2grid",
        )
        with patch.dict("sys.modules", {"lightsim2grid": None}):
            with pytest.raises(ImportError, match="lightsim2grid"):
                _resolve_backend(scenario)

    def test_lightsim2grid_backend_installed(self):
        mock_backend = MagicMock()
        mock_module = MagicMock()
        mock_module.LightSimBackend = mock_backend
        scenario = ScenarioConfig(
            env_name="l2rpn_case14_sandbox",
            backend="lightsim2grid",
        )
        with patch.dict("sys.modules", {"lightsim2grid": mock_module}):
            backend = _resolve_backend(scenario)
        mock_backend.assert_called_once()
        assert backend is mock_backend.return_value

    def test_pypowsybl_backend_missing_raises_import_error(self):
        scenario = ScenarioConfig(
            env_name="l2rpn_case14_sandbox",
            backend="pypowsybl",
        )
        with patch.dict("sys.modules", {"pypowsybl2grid": None}):
            with pytest.raises(ImportError, match="pypowsybl2grid"):
                _resolve_backend(scenario)

    def test_pypowsybl_backend_installed(self):
        mock_backend = MagicMock()
        mock_module = MagicMock()
        mock_module.PyPowSyBlBackend = mock_backend
        scenario = ScenarioConfig(
            env_name="l2rpn_case14_sandbox",
            backend="pypowsybl",
        )
        with patch.dict("sys.modules", {"pypowsybl2grid": mock_module}):
            backend = _resolve_backend(scenario)
        mock_backend.assert_called_once()
        assert backend is mock_backend.return_value

    def test_explicit_compatible_backend_overrides_topology_default(
        self, tmp_path: Path
    ):
        """A compatible explicit backend wins even if topology is set."""
        topo_file = tmp_path / "grid.json"
        topo_file.write_text("{}", encoding="utf-8")
        mock_backend = MagicMock()
        mock_module = MagicMock()
        mock_module.LightSimBackend = mock_backend
        scenario = ScenarioConfig(
            env_name="l2rpn_case14_sandbox",
            topology=TopologySource(format="pandapower", path=topo_file),
            backend="lightsim2grid",
        )
        with patch.dict("sys.modules", {"lightsim2grid": mock_module}):
            backend = _resolve_backend(scenario)
        assert backend is mock_backend.return_value

    def test_build_make_kwargs_includes_backend_when_explicit(self):
        mock_backend_instance = MagicMock()
        mock_module = MagicMock()
        mock_module.LightSimBackend.return_value = mock_backend_instance
        scenario = ScenarioConfig(
            env_name="l2rpn_case14_sandbox",
            backend="lightsim2grid",
        )
        with patch.dict("sys.modules", {"lightsim2grid": mock_module}):
            kwargs = build_make_kwargs(scenario)
        assert kwargs["backend"] is mock_backend_instance


class TestPrepareScenario:
    """Tests for prepare_scenario() — the unified scenario preparation entry point."""

    def test_native_formats_return_none_tmp(self, tmp_path: Path):
        """Scenarios with pandapower + grid2op_chronics_dir skip conversion."""
        topo_file = tmp_path / "grid.json"
        topo_file.write_text("{}", encoding="utf-8")
        chronics_dir = tmp_path / "chronics"
        chronics_dir.mkdir()

        scenario = ScenarioConfig(
            env_name="l2rpn_case14_sandbox",
            topology=TopologySource(format="pandapower", path=topo_file),
            time_series=TimeSeriesSource(
                format="grid2op_chronics_dir", path=chronics_dir
            ),
        )
        env_name, kwargs, tmp_dir = prepare_scenario(scenario)

        assert tmp_dir is None
        assert env_name == "l2rpn_case14_sandbox"
        assert kwargs["grid_path"] == str(topo_file)
        assert kwargs["chronics_path"] == str(chronics_dir)

    def test_no_sources_returns_none_tmp(self):
        """Plain env_name-only scenarios skip conversion."""
        scenario = ScenarioConfig(env_name="l2rpn_case14_sandbox")
        env_name, kwargs, tmp_dir = prepare_scenario(scenario)

        assert tmp_dir is None
        assert env_name == "l2rpn_case14_sandbox"

    def test_csv_time_series_triggers_conversion(self, tmp_path: Path):
        """CSV time series format triggers build_env_dir."""
        topo_file = tmp_path / "grid.json"
        topo_file.write_text("{}", encoding="utf-8")
        ts_dir = tmp_path / "ts"
        ts_dir.mkdir()

        scenario = ScenarioConfig(
            env_name="l2rpn_case14_sandbox",
            topology=TopologySource(format="pandapower", path=topo_file),
            time_series=TimeSeriesSource(format="csv", path=ts_dir),
        )

        mock_tmp = MagicMock(spec=tempfile.TemporaryDirectory)
        mock_tmp.name = str(tmp_path / "fake_env")
        mock_result = EnvDirResult(tmp_dir=mock_tmp, extra_make_kwargs={})

        with patch(
            "grid2benchmark._sources.build_env_dir", return_value=mock_result
        ) as mock_build:
            env_name, kwargs, tmp_dir = prepare_scenario(scenario)

        mock_build.assert_called_once_with(scenario.topology, scenario.time_series)
        assert tmp_dir is mock_tmp
        assert env_name == mock_tmp.name
        assert kwargs.get("test") is True

    def test_pypowsybl_topology_triggers_conversion(self, tmp_path: Path):
        """pypowsybl topology format triggers build_env_dir."""
        xml_file = tmp_path / "network.xml"
        xml_file.write_text("<network/>", encoding="utf-8")

        scenario = ScenarioConfig(
            env_name="l2rpn_case14_sandbox",
            topology=TopologySource(format="pypowsybl", path=xml_file),
        )

        mock_tmp = MagicMock(spec=tempfile.TemporaryDirectory)
        mock_tmp.name = str(tmp_path / "fake_env")
        mock_result = EnvDirResult(tmp_dir=mock_tmp, extra_make_kwargs={})

        with patch("grid2benchmark._sources.build_env_dir", return_value=mock_result):
            env_name, kwargs, tmp_dir = prepare_scenario(scenario)

        assert tmp_dir is mock_tmp
        assert env_name == mock_tmp.name

    def test_cgmes_topology_triggers_conversion(self, tmp_path: Path):
        """cgmes topology format triggers build_env_dir."""
        cgmes_dir = tmp_path / "cgmes"
        cgmes_dir.mkdir()

        scenario = ScenarioConfig(
            env_name="l2rpn_case14_sandbox",
            topology=TopologySource(format="cgmes", path=cgmes_dir),
        )

        mock_tmp = MagicMock(spec=tempfile.TemporaryDirectory)
        mock_tmp.name = str(tmp_path / "fake_env")
        mock_result = EnvDirResult(tmp_dir=mock_tmp, extra_make_kwargs={})

        with patch("grid2benchmark._sources.build_env_dir", return_value=mock_result):
            env_name, kwargs, tmp_dir = prepare_scenario(scenario)

        assert tmp_dir is mock_tmp

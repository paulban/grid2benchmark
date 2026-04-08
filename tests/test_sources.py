"""Unit tests for topology/time-series source adapters."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from grid2benchmark._config import ScenarioConfig, TimeSeriesSource, TopologySource
from grid2benchmark._sources import _resolve_backend, build_make_kwargs


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

    def test_explicit_backend_overrides_topology_default(self, tmp_path: Path):
        """When backend is explicit, it wins even if topology is set."""
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

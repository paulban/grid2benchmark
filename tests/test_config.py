"""Unit tests for ScenarioConfig and BenchmarkConfig."""

from __future__ import annotations

from pathlib import Path

import pytest

from grid2benchmark._config import (
    AVAILABLE_KPI_NAMES,
    COMPATIBLE_BACKENDS_BY_TOPOLOGY,
    DEFAULT_ENV_NAME,
    DEFAULT_MAX_STEPS,
    BenchmarkConfig,
    ScenarioConfig,
    TimeSeriesSource,
    TopologySource,
    SUPPORTED_BACKENDS,
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

    def test_default_topology_is_none(self):
        sc = ScenarioConfig()
        assert sc.topology is None

    def test_default_time_series_source_is_none(self):
        sc = ScenarioConfig()
        assert sc.time_series is None

    def test_default_backend_is_none(self):
        sc = ScenarioConfig()
        assert sc.backend is None


class TestScenarioConfigBackend:
    def test_valid_pandapower_backend(self):
        sc = ScenarioConfig(backend="pandapower")
        assert sc.backend == "pandapower"

    def test_valid_lightsim2grid_backend(self):
        sc = ScenarioConfig(backend="lightsim2grid")
        assert sc.backend == "lightsim2grid"

    def test_valid_pypowsybl_backend(self):
        sc = ScenarioConfig(backend="pypowsybl")
        assert sc.backend == "pypowsybl"

    def test_none_backend_accepted(self):
        sc = ScenarioConfig(backend=None)
        assert sc.backend is None

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="Unsupported backend"):
            ScenarioConfig(backend="unknown_solver")

    def test_all_supported_backends_constant(self):
        assert "pandapower" in SUPPORTED_BACKENDS
        assert "lightsim2grid" in SUPPORTED_BACKENDS
        assert "pypowsybl" in SUPPORTED_BACKENDS

    def test_compatible_backends_constant(self):
        assert COMPATIBLE_BACKENDS_BY_TOPOLOGY["pandapower"] == (
            "pandapower",
            "lightsim2grid",
        )
        assert COMPATIBLE_BACKENDS_BY_TOPOLOGY["pypowsybl"] == ("pypowsybl",)
        assert COMPATIBLE_BACKENDS_BY_TOPOLOGY["cgmes"] == ("pypowsybl",)


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

    def test_is_frozen(self):
        sc = ScenarioConfig()
        with pytest.raises((AttributeError, TypeError)):
            sc.env_name = "other"  # type: ignore[misc]

    def test_pandapower_topology_accepts_lightsim2grid_backend(self, tmp_path: Path):
        topo_file = tmp_path / "grid.json"
        topo_file.write_text("{}", encoding="utf-8")

        sc = ScenarioConfig(
            topology=TopologySource(format="pandapower", path=topo_file),
            backend="lightsim2grid",
        )

        assert sc.backend == "lightsim2grid"

    def test_pandapower_topology_rejects_pypowsybl_backend(self, tmp_path: Path):
        topo_file = tmp_path / "grid.json"
        topo_file.write_text("{}", encoding="utf-8")

        with pytest.raises(ValueError, match="not compatible"):
            ScenarioConfig(
                topology=TopologySource(format="pandapower", path=topo_file),
                backend="pypowsybl",
            )

    def test_pypowsybl_topology_rejects_pandapower_backend(self, tmp_path: Path):
        xml_file = tmp_path / "network.xiidm"
        xml_file.write_text("<network/>", encoding="utf-8")

        with pytest.raises(ValueError, match="not compatible"):
            ScenarioConfig(
                topology=TopologySource(format="pypowsybl", path=xml_file),
                backend="pandapower",
            )

    def test_cgmes_topology_rejects_lightsim2grid_backend(self, tmp_path: Path):
        cgmes_dir = tmp_path / "cgmes"
        cgmes_dir.mkdir()

        with pytest.raises(ValueError, match="not compatible"):
            ScenarioConfig(
                topology=TopologySource(format="cgmes", path=cgmes_dir),
                backend="lightsim2grid",
            )


class TestTopologySource:
    def test_valid_json_file(self, tmp_path: Path):
        topo_file = tmp_path / "grid.json"
        topo_file.write_text("{}", encoding="utf-8")
        src = TopologySource(format="pandapower", path=topo_file)
        assert src.path == topo_file

    def test_xlsx_file_raises(self, tmp_path: Path):
        topo_file = tmp_path / "grid.xlsx"
        topo_file.write_bytes(b"dummy")
        with pytest.raises(ValueError, match=".json extension"):
            TopologySource(format="pandapower", path=topo_file)

    def test_unsupported_format_raises(self, tmp_path: Path):
        topo_file = tmp_path / "grid.json"
        topo_file.write_text("{}", encoding="utf-8")
        with pytest.raises(ValueError, match="Unsupported topology format"):
            TopologySource(format="matpower", path=topo_file)

    def test_missing_file_raises(self, tmp_path: Path):
        with pytest.raises(ValueError, match="does not exist"):
            TopologySource(format="pandapower", path=tmp_path / "missing.json")

    def test_wrong_extension_raises(self, tmp_path: Path):
        topo_file = tmp_path / "grid.txt"
        topo_file.write_text("x", encoding="utf-8")
        with pytest.raises(ValueError, match=".json extension"):
            TopologySource(format="pandapower", path=topo_file)

    # --- pypowsybl ---

    def test_valid_pypowsybl_xml_file(self, tmp_path: Path):
        xml_file = tmp_path / "network.xml"
        xml_file.write_text("<network/>", encoding="utf-8")
        src = TopologySource(format="pypowsybl", path=xml_file)
        assert src.path == xml_file
        assert src.format == "pypowsybl"

    def test_pypowsybl_non_xml_raises(self, tmp_path: Path):
        f = tmp_path / "network.json"
        f.write_text("{}", encoding="utf-8")
        with pytest.raises(ValueError, match=".xml or .xiidm extension"):
            TopologySource(format="pypowsybl", path=f)

    def test_pypowsybl_directory_raises(self, tmp_path: Path):
        with pytest.raises(ValueError, match="must be a file"):
            TopologySource(format="pypowsybl", path=tmp_path)

    # --- cgmes ---

    def test_valid_cgmes_directory(self, tmp_path: Path):
        cgmes_dir = tmp_path / "cgmes"
        cgmes_dir.mkdir()
        src = TopologySource(format="cgmes", path=cgmes_dir)
        assert src.path == cgmes_dir
        assert src.format == "cgmes"

    def test_cgmes_file_instead_of_dir_raises(self, tmp_path: Path):
        xml_file = tmp_path / "EQ.xml"
        xml_file.write_text("<network/>", encoding="utf-8")
        with pytest.raises(ValueError, match="must be a directory"):
            TopologySource(format="cgmes", path=xml_file)

    def test_cgmes_missing_path_raises(self, tmp_path: Path):
        with pytest.raises(ValueError, match="does not exist"):
            TopologySource(format="cgmes", path=tmp_path / "missing_dir")


class TestTimeSeriesSource:
    def test_valid_directory(self, tmp_path: Path):
        chronics_dir = tmp_path / "chronics"
        chronics_dir.mkdir()
        src = TimeSeriesSource(format="grid2op_chronics_dir", path=chronics_dir)
        assert src.path == chronics_dir

    def test_unsupported_format_raises(self, tmp_path: Path):
        chronics_dir = tmp_path / "chronics"
        chronics_dir.mkdir()
        with pytest.raises(ValueError, match="Unsupported time series format"):
            TimeSeriesSource(format="hdf5", path=chronics_dir)

    def test_missing_path_raises(self, tmp_path: Path):
        with pytest.raises(ValueError, match="does not exist"):
            TimeSeriesSource(
                format="grid2op_chronics_dir",
                path=tmp_path / "missing",
            )

    def test_file_instead_of_directory_raises(self, tmp_path: Path):
        file_path = tmp_path / "not_a_dir"
        file_path.write_text("x", encoding="utf-8")
        with pytest.raises(ValueError, match="must be a directory"):
            TimeSeriesSource(format="grid2op_chronics_dir", path=file_path)

    # --- csv ---

    def test_valid_csv_directory(self, tmp_path: Path):
        csv_dir = tmp_path / "timeseries"
        csv_dir.mkdir()
        src = TimeSeriesSource(format="csv", path=csv_dir)
        assert src.path == csv_dir
        assert src.format == "csv"

    def test_csv_missing_path_raises(self, tmp_path: Path):
        with pytest.raises(ValueError, match="does not exist"):
            TimeSeriesSource(format="csv", path=tmp_path / "missing")

    def test_csv_file_instead_of_dir_raises(self, tmp_path: Path):
        f = tmp_path / "data.csv"
        f.write_text("a,b\n1,2\n", encoding="utf-8")
        with pytest.raises(ValueError, match="must be a directory"):
            TimeSeriesSource(format="csv", path=f)

    # --- parquet ---

    def test_valid_parquet_directory(self, tmp_path: Path):
        pq_dir = tmp_path / "parquet_ts"
        pq_dir.mkdir()
        src = TimeSeriesSource(format="parquet", path=pq_dir)
        assert src.path == pq_dir
        assert src.format == "parquet"


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
            BenchmarkConfig(kpis=("carbon_intensity", "nonexistent_kpi"))

    def test_valid_single_kpi_accepted(self):
        bc = BenchmarkConfig(kpis=("carbon_intensity",))
        assert bc.kpis == ("carbon_intensity",)

    def test_list_scenarios_coerced_to_tuple(self):
        bc = BenchmarkConfig(scenarios=[ScenarioConfig()])  # type: ignore[arg-type]
        assert isinstance(bc.scenarios, tuple)

    def test_list_kpis_coerced_to_tuple(self):
        bc = BenchmarkConfig(  # type: ignore[arg-type]
            kpis=["carbon_intensity", "operation_score"]
        )
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
    def test_known_g2e_kpis_present(self):
        for name in (
            "carbon_intensity",
            "operation_score",
            "topological_action_complexity",
        ):
            assert name in AVAILABLE_KPI_NAMES

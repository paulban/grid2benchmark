"""Unit tests for _converter module."""

from __future__ import annotations

import bz2
import csv
import json
import tempfile
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from grid2benchmark._config import TimeSeriesSource, TopologySource
from grid2benchmark._converter import (
    _generate_zero_chronics,
    _has_quantity_file,
    _read_tabular,
    _write_chronics,
    _write_grid_layout,
    _write_one_chronic,
    _write_prods_charac,
    build_env_dir,
    needs_conversion,
)

# ---------------------------------------------------------------------------
# needs_conversion
# ---------------------------------------------------------------------------


class TestNeedsConversion:
    def test_none_none_returns_false(self):
        assert needs_conversion(None, None) is False

    def test_pandapower_chronics_dir_returns_false(self, tmp_path):
        topo = TopologySource.__new__(TopologySource)
        object.__setattr__(topo, "format", "pandapower")
        ts = TimeSeriesSource.__new__(TimeSeriesSource)
        object.__setattr__(ts, "format", "grid2op_chronics_dir")
        assert needs_conversion(topo, ts) is False

    def test_pypowsybl_topology_returns_true(self):
        topo = TopologySource.__new__(TopologySource)
        object.__setattr__(topo, "format", "pypowsybl")
        assert needs_conversion(topo, None) is True

    def test_cgmes_topology_returns_true(self):
        topo = TopologySource.__new__(TopologySource)
        object.__setattr__(topo, "format", "cgmes")
        assert needs_conversion(topo, None) is True

    def test_csv_time_series_returns_true(self):
        ts = TimeSeriesSource.__new__(TimeSeriesSource)
        object.__setattr__(ts, "format", "csv")
        assert needs_conversion(None, ts) is True

    def test_parquet_time_series_returns_true(self):
        ts = TimeSeriesSource.__new__(TimeSeriesSource)
        object.__setattr__(ts, "format", "parquet")
        assert needs_conversion(None, ts) is True

    def test_pandapower_csv_returns_true(self):
        topo = TopologySource.__new__(TopologySource)
        object.__setattr__(topo, "format", "pandapower")
        ts = TimeSeriesSource.__new__(TimeSeriesSource)
        object.__setattr__(ts, "format", "csv")
        assert needs_conversion(topo, ts) is True


# ---------------------------------------------------------------------------
# _write_prods_charac
# ---------------------------------------------------------------------------


def _make_mock_pandapower_net(n_gen=2, n_load=2):
    """Return a minimal mock pandapower network."""
    import pandas as pd

    net = MagicMock()
    net.gen = pd.DataFrame(
        {
            "name": [f"gen_{i}" for i in range(n_gen)],
            "bus": list(range(n_gen)),
            "max_p_mw": [100.0 * (i + 1) for i in range(n_gen)],
            "min_p_mw": [0.0] * n_gen,
        }
    )
    net.load = pd.DataFrame(
        {
            "name": [f"load_{i}" for i in range(n_load)],
            "bus": list(range(n_load)),
            "p_mw": [50.0] * n_load,
        }
    )
    net.bus = pd.DataFrame(
        {"name": [f"bus_{i}" for i in range(max(n_gen, n_load))]},
        index=range(max(n_gen, n_load)),
    )
    net.bus_geodata = pd.DataFrame()
    return net


class TestWriteProdsCharac:
    def test_creates_csv_with_required_columns(self, tmp_path):
        net = _make_mock_pandapower_net(n_gen=3, n_load=2)
        _write_prods_charac(net, tmp_path)

        csv_path = tmp_path / "prods_charac.csv"
        assert csv_path.exists()

        df = pd.read_csv(csv_path)
        required_cols = {"name", "Pmax", "Pmin"}
        assert required_cols.issubset(set(df.columns))
        assert len(df) == 3

    def test_pmax_pmin_values_correct(self, tmp_path):
        net = _make_mock_pandapower_net(n_gen=2, n_load=1)
        _write_prods_charac(net, tmp_path)

        df = pd.read_csv(tmp_path / "prods_charac.csv")
        assert df.loc[0, "Pmax"] == pytest.approx(100.0)
        assert df.loc[1, "Pmax"] == pytest.approx(200.0)
        assert df.loc[0, "Pmin"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# _write_grid_layout
# ---------------------------------------------------------------------------


class TestWriteGridLayout:
    def test_creates_json_without_geodata(self, tmp_path):
        net = _make_mock_pandapower_net()
        _write_grid_layout(net, tmp_path)

        layout_path = tmp_path / "grid_layout.json"
        assert layout_path.exists()
        data = json.loads(layout_path.read_text())
        assert isinstance(data, dict)
        # No geodata → empty layout
        assert data == {}

    def test_creates_json_with_geodata(self, tmp_path):
        net = _make_mock_pandapower_net(n_gen=2, n_load=2)
        net.bus_geodata = pd.DataFrame({"x": [0.0, 1.0], "y": [2.0, 3.0]}, index=[0, 1])
        _write_grid_layout(net, tmp_path)

        data = json.loads((tmp_path / "grid_layout.json").read_text())
        assert "bus_0" in data or any(isinstance(v, list) for v in data.values())


# ---------------------------------------------------------------------------
# _read_tabular
# ---------------------------------------------------------------------------


class TestReadTabular:
    def test_read_csv_with_datetime_index(self, tmp_path):
        csv_content = "timestamp,gen_0,gen_1\n2020-01-01 00:00,100,200\n2020-01-01 00:05,110,210\n"
        csv_file = tmp_path / "prod_p.csv"
        csv_file.write_text(csv_content, encoding="utf-8")

        df = _read_tabular(csv_file, "csv")
        assert len(df) == 2
        assert "gen_0" in df.columns

    def test_read_csv_without_datetime_index(self, tmp_path):
        csv_content = "idx,gen_0\n0,100\n1,110\n"
        csv_file = tmp_path / "prod_p.csv"
        csv_file.write_text(csv_content, encoding="utf-8")

        df = _read_tabular(csv_file, "csv")
        assert len(df) == 2

    def test_read_parquet(self, tmp_path):
        df_orig = pd.DataFrame({"gen_0": [100.0, 110.0], "gen_1": [200.0, 210.0]})
        pq_file = tmp_path / "prod_p.parquet"
        df_orig.to_parquet(pq_file)

        df = _read_tabular(pq_file, "parquet")
        assert list(df.columns) == ["gen_0", "gen_1"]
        assert len(df) == 2


# ---------------------------------------------------------------------------
# _write_one_chronic
# ---------------------------------------------------------------------------


class TestWriteOneChronic:
    def _make_csv_source(self, tmp_path, n_rows=3):
        """Write prod_p.csv and load_p.csv into tmp_path."""
        for stem, cols in [("prod_p", ["gen_0", "gen_1"]), ("load_p", ["load_0"])]:
            rows = [cols] + [[str(i * 10) for _ in cols] for i in range(n_rows)]
            (tmp_path / f"{stem}.csv").write_text(
                "\n".join(",".join(r) for r in rows), encoding="utf-8"
            )

    def test_writes_bz2_files(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        dest = tmp_path / "dest"
        dest.mkdir()
        self._make_csv_source(src)

        _write_one_chronic(src, dest, "csv", ["gen_0", "gen_1"], ["load_0"])

        assert (dest / "prod_p.csv.bz2").exists()
        assert (dest / "load_p.csv.bz2").exists()

    def test_writes_grid2op_separator(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        dest = tmp_path / "dest"
        dest.mkdir()
        self._make_csv_source(src)

        _write_one_chronic(src, dest, "csv", ["gen_0", "gen_1"], ["load_0"])

        with bz2.open(dest / "prod_p.csv.bz2", "rt", encoding="utf-8") as fh:
            header = fh.readline().strip()

        assert header == "gen_0;gen_1"

    def test_writes_info_files_with_defaults(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        dest = tmp_path / "dest"
        dest.mkdir()
        self._make_csv_source(src)

        _write_one_chronic(src, dest, "csv", [], [])

        assert (dest / "start_datetime.info").exists()
        assert (dest / "time_interval.info").exists()

    def test_missing_required_file_raises(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        dest = tmp_path / "dest"
        dest.mkdir()
        # Only prod_p.csv, no load_p.csv
        (src / "prod_p.csv").write_text("gen_0\n100\n", encoding="utf-8")

        with pytest.raises(FileNotFoundError, match="load_p"):
            _write_one_chronic(src, dest, "csv", ["gen_0"], ["load_0"])

    def test_optional_missing_files_skipped(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        dest = tmp_path / "dest"
        dest.mkdir()
        # prod_p and load_p only; no forecasted files
        for stem in ("prod_p", "load_p"):
            (src / f"{stem}.csv").write_text("col\n1\n2\n", encoding="utf-8")

        _write_one_chronic(src, dest, "csv", [], [])

        # Optional files should NOT be created in dest
        assert not (dest / "prod_p_forecasted.csv.bz2").exists()


# ---------------------------------------------------------------------------
# _write_chronics — flat vs. sub-directory layout
# ---------------------------------------------------------------------------


class TestWriteChronics:
    def _make_ts_source(self, src: Path):
        for stem in ("prod_p", "load_p"):
            (src / f"{stem}.csv").write_text("col\n1\n2\n", encoding="utf-8")

    def test_flat_layout_produces_single_chronic(self, tmp_path):
        src = tmp_path / "ts"
        src.mkdir()
        self._make_ts_source(src)

        ts = TimeSeriesSource.__new__(TimeSeriesSource)
        object.__setattr__(ts, "format", "csv")
        object.__setattr__(ts, "path", src)

        chronics_dir = tmp_path / "chronics"
        chronics_dir.mkdir()

        _write_chronics(ts, [], [], chronics_dir)

        subdirs = list(chronics_dir.iterdir())
        assert len(subdirs) == 1
        assert subdirs[0].name == "0000"

    def test_subdir_layout_produces_multiple_chronics(self, tmp_path):
        src = tmp_path / "ts"
        src.mkdir()
        for chronic_name in ("0000", "0001"):
            chronic_src = src / chronic_name
            chronic_src.mkdir()
            self._make_ts_source(chronic_src)

        ts = TimeSeriesSource.__new__(TimeSeriesSource)
        object.__setattr__(ts, "format", "csv")
        object.__setattr__(ts, "path", src)

        chronics_dir = tmp_path / "chronics"
        chronics_dir.mkdir()

        _write_chronics(ts, [], [], chronics_dir)

        subdirs = sorted(d.name for d in chronics_dir.iterdir())
        assert subdirs == ["0000", "0001"]


# ---------------------------------------------------------------------------
# build_env_dir — integration-style (mocked network loading)
# ---------------------------------------------------------------------------


class TestBuildEnvDir:
    def test_returns_temporary_directory(self, tmp_path):
        """build_env_dir returns an EnvDirResult with a TemporaryDirectory."""
        result = build_env_dir(None, None)
        assert isinstance(result.tmp_dir, tempfile.TemporaryDirectory)
        path = result.tmp_dir.name
        result.tmp_dir.cleanup()
        assert not Path(path).exists()

    def test_creates_chronics_dir_from_csv(self, tmp_path):
        src = tmp_path / "ts"
        src.mkdir()
        for stem in ("prod_p", "load_p"):
            (src / f"{stem}.csv").write_text("col\n1\n2\n", encoding="utf-8")

        ts = TimeSeriesSource.__new__(TimeSeriesSource)
        object.__setattr__(ts, "format", "csv")
        object.__setattr__(ts, "path", src)

        result = build_env_dir(None, ts)
        try:
            chronics = Path(result.tmp_dir.name) / "chronics"
            assert chronics.is_dir()
            assert any(chronics.iterdir())
        finally:
            result.tmp_dir.cleanup()

    def test_pandapower_topology_writes_grid_json(self, tmp_path):
        """Pandapower JSON topology is loaded and written as grid.json."""
        import pandapower as pp

        net = pp.create_empty_network()
        b0 = pp.create_bus(net, vn_kv=110, name="bus_0")
        b1 = pp.create_bus(net, vn_kv=110, name="bus_1")
        pp.create_ext_grid(net, bus=b0)
        pp.create_load(net, bus=b1, p_mw=10, q_mvar=2, name="load_0")
        pp.create_gen(net, bus=b0, p_mw=50, name="gen_0", max_p_mw=100, min_p_mw=0)
        pp.create_line_from_parameters(
            net,
            from_bus=b0,
            to_bus=b1,
            length_km=1,
            r_ohm_per_km=0.1,
            x_ohm_per_km=0.1,
            c_nf_per_km=0,
            max_i_ka=1,
            name="line_0",
        )

        topo_file = tmp_path / "grid.json"
        pp.to_json(net, str(topo_file))

        topo = TopologySource(format="pandapower", path=topo_file)

        result = build_env_dir(topo, None)
        try:
            env_path = Path(result.tmp_dir.name)
            assert (env_path / "grid.json").exists()
            assert (env_path / "prods_charac.csv").exists()
            assert (env_path / "grid_layout.json").exists()
        finally:
            result.tmp_dir.cleanup()

    def test_topology_without_time_series_generates_zero_chronic(self, tmp_path):
        """When topology is given but no time series, a zero chronic is created."""
        import pandapower as pp

        net = pp.create_empty_network()
        b0 = pp.create_bus(net, vn_kv=110, name="bus_0")
        b1 = pp.create_bus(net, vn_kv=110, name="bus_1")
        pp.create_ext_grid(net, bus=b0)
        pp.create_load(net, bus=b1, p_mw=10, q_mvar=2, name="load_0")
        pp.create_gen(net, bus=b0, p_mw=50, name="gen_0", max_p_mw=100, min_p_mw=0)
        pp.create_line_from_parameters(
            net,
            from_bus=b0,
            to_bus=b1,
            length_km=1,
            r_ohm_per_km=0.1,
            x_ohm_per_km=0.1,
            c_nf_per_km=0,
            max_i_ka=1,
            name="line_0",
        )

        topo_file = tmp_path / "grid.json"
        pp.to_json(net, str(topo_file))

        topo = TopologySource(format="pandapower", path=topo_file)

        result = build_env_dir(topo, None)
        try:
            env_path = Path(result.tmp_dir.name)
            chronic_0000 = env_path / "chronics" / "0000"
            assert chronic_0000.is_dir()
            assert (chronic_0000 / "prod_p.csv.bz2").exists()
            assert (chronic_0000 / "load_p.csv.bz2").exists()
            assert (chronic_0000 / "prod_v.csv.bz2").exists()
            assert (chronic_0000 / "load_q.csv.bz2").exists()
            assert (chronic_0000 / "start_datetime.info").exists()
            assert (chronic_0000 / "time_interval.info").exists()
        finally:
            result.tmp_dir.cleanup()


# ---------------------------------------------------------------------------
# _generate_zero_chronics — unit tests
# ---------------------------------------------------------------------------


class TestGenerateZeroChronics:
    def test_creates_expected_files(self, tmp_path):
        net = _make_mock_pandapower_net(n_gen=2, n_load=3)
        chronics_dir = tmp_path / "chronics"
        chronics_dir.mkdir()
        _generate_zero_chronics(net, chronics_dir)

        chronic = chronics_dir / "0000"
        assert (chronic / "prod_p.csv.bz2").exists()
        assert (chronic / "load_p.csv.bz2").exists()
        assert (chronic / "prod_v.csv.bz2").exists()
        assert (chronic / "load_q.csv.bz2").exists()
        assert (chronic / "start_datetime.info").exists()
        assert (chronic / "time_interval.info").exists()

    def test_prod_p_columns_match_gen_names(self, tmp_path):
        net = _make_mock_pandapower_net(n_gen=3, n_load=1)
        chronics_dir = tmp_path / "chronics"
        chronics_dir.mkdir()
        _generate_zero_chronics(net, chronics_dir)

        df = pd.read_csv(chronics_dir / "0000" / "prod_p.csv.bz2", sep=";")
        assert list(df.columns) == ["gen_0", "gen_1", "gen_2"]
        assert (df == 0.0).all().all()

    def test_prod_v_defaults_to_one(self, tmp_path):
        net = _make_mock_pandapower_net(n_gen=2, n_load=1)
        chronics_dir = tmp_path / "chronics"
        chronics_dir.mkdir()
        _generate_zero_chronics(net, chronics_dir)

        df = pd.read_csv(chronics_dir / "0000" / "prod_v.csv.bz2", sep=";")
        assert (df == 1.0).all().all()

    def test_two_timesteps_written(self, tmp_path):
        net = _make_mock_pandapower_net(n_gen=1, n_load=1)
        chronics_dir = tmp_path / "chronics"
        chronics_dir.mkdir()
        _generate_zero_chronics(net, chronics_dir)

        df = pd.read_csv(chronics_dir / "0000" / "prod_p.csv.bz2", sep=";")
        assert len(df) == 2


# ---------------------------------------------------------------------------
# Integration test: initial_snapshot.xiidm -> env dir with zero chronics
# ---------------------------------------------------------------------------

INITIAL_SNAPSHOT_PATH = (
    Path(__file__).parent.parent
    / "test_data"
    / "initial_snapshot.xiidm"
    / "initial_snapshot.xiidm"
)


@pytest.mark.integration
class TestInitialSnapshotConversion:
    """End-to-end conversion of the bundled IIDM snapshot to a Grid2Op env dir."""

    @pytest.fixture(autouse=True)
    def require_snapshot(self):
        if not INITIAL_SNAPSHOT_PATH.exists():
            pytest.skip("initial_snapshot.xiidm not found in test_data/")

    @pytest.fixture(autouse=True)
    def require_pypowsybl(self):
        pytest.importorskip("pypowsybl", reason="pypowsybl not installed")
        pytest.importorskip("pypowsybl2grid", reason="pypowsybl2grid not installed")

    def test_env_dir_created(self):
        """build_env_dir succeeds and produces the required env files."""
        topo = TopologySource(format="pypowsybl", path=INITIAL_SNAPSHOT_PATH)
        result = build_env_dir(topo, None)
        try:
            env = Path(result.tmp_dir.name)
            assert (env / "grid.xiidm").exists(), "grid.xiidm missing"
            assert (env / "prods_charac.csv").exists(), "prods_charac.csv missing"
            assert (env / "grid_layout.json").exists(), "grid_layout.json missing"
        finally:
            result.tmp_dir.cleanup()

    def test_zero_chronics_created(self):
        """A zero-valued chronic is synthesised when no time series are given."""
        topo = TopologySource(format="pypowsybl", path=INITIAL_SNAPSHOT_PATH)
        result = build_env_dir(topo, None)
        try:
            chronic_0000 = Path(result.tmp_dir.name) / "chronics" / "0000"
            assert chronic_0000.is_dir(), "chronics/0000 missing"
            assert (chronic_0000 / "prod_p.csv.bz2").exists()
            assert (chronic_0000 / "load_p.csv.bz2").exists()
        finally:
            result.tmp_dir.cleanup()

    def test_prods_charac_has_network_generators(self):
        """prods_charac.csv contains one row per generator in the IIDM network."""
        topo = TopologySource(format="pypowsybl", path=INITIAL_SNAPSHOT_PATH)
        result = build_env_dir(topo, None)
        try:
            df = pd.read_csv(Path(result.tmp_dir.name) / "prods_charac.csv")
            assert len(df) > 0, "prods_charac.csv is empty"
            assert "Pmax" in df.columns
            assert "Pmin" in df.columns
        finally:
            result.tmp_dir.cleanup()

    def test_prod_p_columns_match_prods_charac_names(self):
        """Generator columns in prod_p.csv.bz2 match names in prods_charac.csv."""
        topo = TopologySource(format="pypowsybl", path=INITIAL_SNAPSHOT_PATH)
        result = build_env_dir(topo, None)
        try:
            env = Path(result.tmp_dir.name)
            charac = pd.read_csv(env / "prods_charac.csv")
            prod_p = pd.read_csv(env / "chronics" / "0000" / "prod_p.csv.bz2")
            assert set(prod_p.columns) == set(charac["name"].astype(str))
        finally:
            result.tmp_dir.cleanup()

    def test_all_chronic_values_are_zero_or_one(self):
        """prod_p and load_p are zero; prod_v is 1.0."""
        topo = TopologySource(format="pypowsybl", path=INITIAL_SNAPSHOT_PATH)
        result = build_env_dir(topo, None)
        try:
            chronic = Path(result.tmp_dir.name) / "chronics" / "0000"
            prod_p = pd.read_csv(chronic / "prod_p.csv.bz2")
            load_p = pd.read_csv(chronic / "load_p.csv.bz2")
            prod_v = pd.read_csv(chronic / "prod_v.csv.bz2")
            assert (prod_p == 0.0).all().all()
            assert (load_p == 0.0).all().all()
            assert (prod_v == 1.0).all().all()
        finally:
            result.tmp_dir.cleanup()

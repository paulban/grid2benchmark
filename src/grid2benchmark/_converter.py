"""Grid data and time-series conversion utilities.

This module converts external grid formats (pandapower JSON, pypowsybl IIDM,
CGMES) and time-series formats (CSV, Parquet) into a temporary Grid2Op
environment directory.
"""

from __future__ import annotations

import json
import logging
import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

if TYPE_CHECKING:
    from ._config import TimeSeriesSource, TopologySource

logger = logging.getLogger(__name__)

_CHRONIC_QUANTITIES: list[tuple[str, bool]] = [
    ("prod_p", False),
    ("load_p", False),
    ("prod_v", True),
    ("load_q", True),
    ("prod_p_forecasted", True),
    ("load_p_forecasted", True),
    ("prod_v_forecasted", True),
    ("load_q_forecasted", True),
    ("prices", True),
]

_DEFAULT_START_DATETIME = "2020-01-01 00:00"
_DEFAULT_TIME_INTERVAL = "00:05"
_IIDM_GRID_FILENAME = "grid.xiidm"


class EnvDirResult(NamedTuple):
    """Result returned by build_env_dir."""

    tmp_dir: tempfile.TemporaryDirectory  # type: ignore[type-arg]
    extra_make_kwargs: dict[str, Any]


def needs_conversion(
    topology: "TopologySource | None", time_series: "TimeSeriesSource | None"
) -> bool:
    """Return True when scenario inputs require format conversion."""
    if topology is not None and topology.format in ("pypowsybl", "cgmes"):
        return True
    if time_series is not None and time_series.format in ("csv", "parquet"):
        return True
    return False


def build_env_dir(
    topology: "TopologySource | None",
    time_series: "TimeSeriesSource | None",
) -> EnvDirResult:
    """Build a Grid2Op-compatible temporary environment directory."""
    tmp = tempfile.TemporaryDirectory(prefix="g2b_env_")
    env_path = Path(tmp.name)
    extra_make_kwargs: dict[str, Any] = {}

    gen_names: list[str] = []
    load_names: list[str] = []
    backend_kind = "pandapower"

    if topology is not None:
        if topology.format == "pandapower":
            net = _load_pandapower_network(topology)
            _write_grid(net, env_path)
            _write_prods_charac(net, env_path)
            _write_grid_layout(net, env_path)
            gen_names = [str(x) for x in net.gen["name"].astype(str)]
            load_names = [str(x) for x in net.load["name"].astype(str)]
        else:
            backend_kind = "pypowsybl"
            pynet = _load_pypowsybl_network(topology)
            grid_path = _write_iidm_grid(pynet, env_path)
            _write_prods_charac_from_pypowsybl(pynet, env_path)
            _write_grid_layout_from_pypowsybl(pynet, env_path)
            gen_names = [str(x) for x in pynet.get_generators().index]
            load_names = [str(x) for x in pynet.get_loads().index]

            try:
                from pypowsybl2grid import PyPowSyBlBackend  # type: ignore
            except ImportError as exc:
                raise ImportError(
                    "pypowsybl2grid is required for pypowsybl/cgmes conversion. "
                    "Install it with: pip install pypowsybl2grid"
                ) from exc

            extra_make_kwargs["backend"] = PyPowSyBlBackend()
            extra_make_kwargs["grid_path"] = str(grid_path)

    chronics_dir = env_path / "chronics"
    chronics_dir.mkdir(parents=True, exist_ok=True)
    has_forecasts = False

    if time_series is not None:
        has_forecasts = _write_chronics(
            time_series, gen_names, load_names, chronics_dir
        )
    elif gen_names or load_names:
        _generate_zero_chronics_from_names(gen_names, load_names, chronics_dir)

    _write_env_package_marker(env_path)
    _write_env_config(env_path, backend_kind, has_forecasts)

    return EnvDirResult(tmp_dir=tmp, extra_make_kwargs=extra_make_kwargs)


def _load_pandapower_network(topology: "TopologySource") -> Any:
    try:
        import pandapower as pp  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "pandapower is required for pandapower topology conversion. "
            "Install it with: pip install pandapower"
        ) from exc
    return pp.from_json(str(topology.path))


def _load_pypowsybl_network(topology: "TopologySource") -> Any:
    try:
        import pypowsybl.network as pn  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "pypowsybl is required for IIDM/CGMES conversion. "
            "Install it with: pip install pypowsybl"
        ) from exc

    if topology.format == "pypowsybl":
        return pn.load(str(topology.path))

    return _load_cgmes_dir(topology.path)


def _load_cgmes_dir(cgmes_dir: Path) -> Any:
    import pypowsybl.network as pn  # type: ignore

    xml_files = sorted(cgmes_dir.glob("*.xml"))
    if not xml_files:
        raise ValueError(f"No XML files found in CGMES directory: {cgmes_dir}")

    return pn.load(str(xml_files[0]), extra_files=[str(f) for f in xml_files[1:]])


def _write_grid(net: Any, dest_dir: Path) -> None:
    import pandapower as pp  # type: ignore

    pp.to_json(net, str(dest_dir / "grid.json"))


def _write_env_package_marker(dest_dir: Path) -> None:
    (dest_dir / "__init__.py").write_text(
        "# DO NOT REMOVE, automatically generated by grid2benchmark\n",
        encoding="utf-8",
    )


def _write_env_config(dest_dir: Path, backend_kind: str, has_forecasts: bool) -> None:
    if backend_kind == "pypowsybl":
        backend_import = "from pypowsybl2grid import PyPowSyBlBackend"
        backend_class = "PyPowSyBlBackend"
    else:
        backend_import = "from grid2op.Backend import PandaPowerBackend"
        backend_class = "PandaPowerBackend"

    grid_value_class = (
        "GridStateFromFileWithForecasts" if has_forecasts else "GridStateFromFile"
    )

    config_text = "\n".join(
        [
            "from grid2op.Action import TopologyAndDispatchAction",
            "from grid2op.Reward import RedispReward",
            "from grid2op.Rules import DefaultRules",
            "from grid2op.Chronics import Multifolder, GridStateFromFile, GridStateFromFileWithForecasts",
            backend_import,
            "",
            "config = {",
            f'    "backend": {backend_class},',
            '    "action_class": TopologyAndDispatchAction,',
            '    "observation_class": None,',
            '    "reward_class": RedispReward,',
            '    "gamerules_class": DefaultRules,',
            '    "chronics_class": Multifolder,',
            f'    "grid_value_class": {grid_value_class},',
            '    "volagecontroler_class": None,',
            '    "thermal_limits": None,',
            '    "names_chronics_to_grid": None,',
            "}",
            "",
        ]
    )
    (dest_dir / "config.py").write_text(config_text, encoding="utf-8")


def _write_prods_charac(net: Any, dest_dir: Path) -> None:
    import pandas as pd  # type: ignore

    gen = net.gen.copy()
    bus = net.bus

    rows: list[dict[str, Any]] = []
    for i, g in gen.iterrows():
        bus_idx = g["bus"]
        bus_name = bus.loc[bus_idx, "name"] if "name" in bus.columns else str(bus_idx)
        pmin = max(0.0, float(g.get("min_p_mw", 0.0)))
        pmax = max(pmin, float(g.get("max_p_mw", 0.0)))
        rows.append(
            {
                "name": str(g.get("name", f"gen_{i}")),
                "Pmax": pmax,
                "Pmin": pmin,
                "type": str(g.get("type")) if "type" in g.index else float("nan"),
                "bus": str(bus_name),
                "max_ramp_up": 0.0,
                "max_ramp_down": 0.0,
                "min_up_time": 0.0,
                "min_down_time": 0.0,
                "marginal_cost": 1.0,
                "shut_down_cost": 0.0,
                "start_cost": 0.0,
                "zone": float("nan"),
            }
        )

    pd.DataFrame(rows).to_csv(dest_dir / "prods_charac.csv", index=False)


def _write_grid_layout(net: Any, dest_dir: Path) -> None:
    layout: dict[str, list[float]] = {}

    if (
        hasattr(net, "bus_geodata")
        and net.bus_geodata is not None
        and not net.bus_geodata.empty
    ):
        bus = net.bus
        for idx, row in net.bus_geodata.iterrows():
            name = bus.loc[idx, "name"] if "name" in bus.columns else str(idx)
            layout[str(name)] = [float(row.get("x", 0.0)), float(row.get("y", 0.0))]

    with open(dest_dir / "grid_layout.json", "w", encoding="utf-8") as fh:
        json.dump(layout, fh, indent=2)


def _write_iidm_grid(pynet: Any, dest_dir: Path) -> Path:
    grid_path = dest_dir / _IIDM_GRID_FILENAME
    pynet.save(str(grid_path), format="XIIDM")
    return grid_path


def _write_prods_charac_from_pypowsybl(pynet: Any, dest_dir: Path) -> None:
    import pandas as pd  # type: ignore

    gen_df = pynet.get_generators()
    source_map = {
        "HYDRO": "hydro",
        "WIND": "wind",
        "SOLAR": "solar",
        "NUCLEAR": "nuclear",
        "THERMAL": "thermal",
        "OTHER": "thermal",
        "": "thermal",
    }

    rows: list[dict[str, Any]] = []
    for gen_id, g in gen_df.iterrows():
        source = str(g.get("energy_source", "")) if "energy_source" in g.index else ""
        pmin = max(0.0, float(g.get("min_p", 0.0)))
        pmax = max(pmin, float(g.get("max_p", 0.0)))
        rows.append(
            {
                "name": str(gen_id),
                "Pmax": pmax,
                "Pmin": pmin,
                "type": source_map.get(source, "thermal"),
                "bus": str(g.get("bus_id", "")),
                "max_ramp_up": 0.0,
                "max_ramp_down": 0.0,
                "min_up_time": 0.0,
                "min_down_time": 0.0,
                "marginal_cost": 0.0,
                "shut_down_cost": 0.0,
                "start_cost": 0.0,
                "zone": float("nan"),
            }
        )

    pd.DataFrame(rows).to_csv(dest_dir / "prods_charac.csv", index=False)


def _write_grid_layout_from_pypowsybl(pynet: Any, dest_dir: Path) -> None:
    # IIDM/CGMES geodata is often missing; write an empty layout as a valid default.
    _ = pynet
    with open(dest_dir / "grid_layout.json", "w", encoding="utf-8") as fh:
        json.dump({}, fh, indent=2)


def _write_chronics(
    time_series: "TimeSeriesSource",
    gen_names: list[str],
    load_names: list[str],
    chronics_dir: Path,
) -> bool:
    ext = "csv" if time_series.format == "csv" else "parquet"
    src_root = time_series.path
    has_forecasts = False

    # Flat layout: quantity files at root.
    if _has_quantity_file(src_root, ext):
        dest = chronics_dir / "0000"
        dest.mkdir(parents=True, exist_ok=True)
        return _write_one_chronic(src_root, dest, ext, gen_names, load_names)

    # Sub-directory layout: one chronic per child directory.
    chronic_dirs = sorted([p for p in src_root.iterdir() if p.is_dir()])
    if not chronic_dirs:
        raise ValueError(
            f"No '{ext}' quantity files found in {src_root} and no chronic sub-directories present"
        )

    for chronic_src in chronic_dirs:
        dest = chronics_dir / chronic_src.name
        dest.mkdir(parents=True, exist_ok=True)
        has_forecasts = (
            _write_one_chronic(chronic_src, dest, ext, gen_names, load_names)
            or has_forecasts
        )

    return has_forecasts


def _has_quantity_file(directory: Path, ext: str) -> bool:
    return any(
        (directory / f"{stem}.{ext}").exists() for stem, _ in _CHRONIC_QUANTITIES
    )


def _write_one_chronic(
    src_dir: Path,
    dest_dir: Path,
    ext: str,
    gen_names: list[str],
    load_names: list[str],
) -> bool:
    import pandas as pd  # type: ignore

    has_forecasts = False

    for stem, optional in _CHRONIC_QUANTITIES:
        src_file = src_dir / f"{stem}.{ext}"

        if not src_file.exists():
            if optional:
                continue
            raise FileNotFoundError(f"Required quantity file missing: {src_file}")

        if stem.endswith("_forecasted"):
            has_forecasts = True

        df = _read_tabular(src_file, ext)

        # Keep only known generator/load columns when names are provided.
        if stem.startswith("prod_") and gen_names:
            missing = [c for c in gen_names if c not in df.columns]
            if missing:
                raise ValueError(
                    f"{src_file.name}: missing generator columns: {missing[:5]}"
                )
            df = df.loc[:, gen_names]

        if stem.startswith("load_") and load_names:
            missing = [c for c in load_names if c not in df.columns]
            if missing:
                raise ValueError(
                    f"{src_file.name}: missing load columns: {missing[:5]}"
                )
            df = df.loc[:, load_names]

        out_file = dest_dir / f"{stem}.csv.bz2"
        df.to_csv(out_file, index=False, sep=";", compression="bz2")

    for info_name, default_value in (
        ("start_datetime.info", _DEFAULT_START_DATETIME),
        ("time_interval.info", _DEFAULT_TIME_INTERVAL),
    ):
        src = src_dir / info_name
        if src.exists():
            shutil.copy(src, dest_dir / info_name)
        else:
            (dest_dir / info_name).write_text(default_value, encoding="utf-8")

    return has_forecasts


def _read_tabular(path: Path, ext: str):
    import pandas as pd  # type: ignore

    if ext == "csv":
        return pd.read_csv(path)
    if ext == "parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported tabular extension: {ext}")


def _generate_zero_chronics(net: Any, chronics_dir: Path) -> None:
    gen_names = list(net.gen["name"].astype(str))
    load_names = list(net.load["name"].astype(str))
    _generate_zero_chronics_from_names(gen_names, load_names, chronics_dir)


def _generate_zero_chronics_from_names(
    gen_names: list[str],
    load_names: list[str],
    chronics_dir: Path,
) -> None:
    import pandas as pd  # type: ignore

    chronic_dest = chronics_dir / "0000"
    chronic_dest.mkdir(parents=True, exist_ok=True)

    quantities: dict[str, list[str]] = {
        "prod_p": gen_names,
        "prod_v": gen_names,
        "load_p": load_names,
        "load_q": load_names,
    }

    n_rows = 2
    for stem, names in quantities.items():
        if not names:
            continue
        if stem == "prod_v":
            data = {name: [1.0] * n_rows for name in names}
        else:
            data = {name: [0.0] * n_rows for name in names}
        pd.DataFrame(data).to_csv(
            chronic_dest / f"{stem}.csv.bz2",
            index=False,
            sep=";",
            compression="bz2",
        )

    (chronic_dest / "start_datetime.info").write_text(
        _DEFAULT_START_DATETIME, encoding="utf-8"
    )
    (chronic_dest / "time_interval.info").write_text(
        _DEFAULT_TIME_INTERVAL, encoding="utf-8"
    )

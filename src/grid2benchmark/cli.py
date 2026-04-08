"""Command-line interface for grid2benchmark.

Entry point used by the ``grid2benchmark`` console script and
``python -m grid2benchmark.cli``.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any

from ._config import (
    DEFAULT_KPIS,
    DEFAULT_ENV_NAME,
    DEFAULT_MAX_STEPS,
    BenchmarkConfig,
    ScenarioConfig,
    TimeSeriesSource,
    TopologySource,
    SUPPORTED_BACKENDS,
)
from . import run_benchmark


def _configure_warning_filters() -> None:
    """Suppress noisy pandas copy warnings emitted by Grid2Op internals."""
    try:
        from pandas.errors import SettingWithCopyWarning

        warnings.filterwarnings(
            "ignore",
            category=SettingWithCopyWarning,
            module=r"grid2op\.Backend\.pandaPowerBackend",
        )
    except Exception:
        # If pandas is unavailable for any reason, keep default warning behavior.
        return


def _parse_time_series_ids(raw: str | None) -> tuple[int, ...] | None:
    """Parse comma-separated time-series IDs from CLI input.

    Args:
        raw: String like ``"0,1,2"`` or ``None``.

    Returns:
        Tuple of integer IDs, or ``None`` if no IDs were provided.
    """
    if raw is None:
        return None

    ids = [part.strip() for part in raw.split(",") if part.strip()]
    if not ids:
        return None
    return tuple(int(ts_id) for ts_id in ids)


def _load_scenarios(scenarios_file: Path) -> tuple[ScenarioConfig, ...]:
    """Load scenario definitions from JSON file.

    Accepted formats:

    - list of scenario objects
    - object containing ``{"scenarios": [...]}``
    """
    payload: Any = json.loads(scenarios_file.read_text(encoding="utf-8"))

    if isinstance(payload, dict) and "scenarios" in payload:
        payload = payload["scenarios"]

    if not isinstance(payload, list):
        raise ValueError("Scenario file must contain a list or {'scenarios': [...]}.")

    scenarios: list[ScenarioConfig] = []
    for idx, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"Scenario index {idx} must be a JSON object")

        env_name = item.get("env_name", DEFAULT_ENV_NAME)
        time_series_ids = item.get("time_series_ids")
        topology_raw = item.get("topology")
        time_series_raw = item.get("time_series")

        topology: TopologySource | None = None
        if topology_raw is not None:
            if not isinstance(topology_raw, dict):
                raise ValueError(f"Scenario index {idx} topology must be an object")
            if "format" not in topology_raw or "path" not in topology_raw:
                raise ValueError(
                    f"Scenario index {idx} topology must include format and path"
                )
            topology = TopologySource(
                format=str(topology_raw["format"]),
                path=Path(topology_raw["path"]),
            )

        time_series: TimeSeriesSource | None = None
        if time_series_raw is not None:
            if not isinstance(time_series_raw, dict):
                raise ValueError(f"Scenario index {idx} time_series must be an object")
            if "format" not in time_series_raw or "path" not in time_series_raw:
                raise ValueError(
                    f"Scenario index {idx} time_series must include format and path"
                )
            time_series = TimeSeriesSource(
                format=str(time_series_raw["format"]),
                path=Path(time_series_raw["path"]),
            )

        parsed_time_series_ids: tuple[int, ...] | None = None
        if time_series_ids is not None:
            if not isinstance(time_series_ids, list):
                raise ValueError(f"Scenario index {idx} time_series_ids must be a list")
            parsed_time_series_ids = tuple(int(v) for v in time_series_ids)

        backend_raw = item.get("backend")
        backend: str | None = str(backend_raw) if backend_raw is not None else None

        scenarios.append(
            ScenarioConfig(
                env_name=str(env_name),
                time_series_ids=parsed_time_series_ids,
                topology=topology,
                time_series=time_series,
                backend=backend,
            )
        )

    return tuple(scenarios)


def _parse_kpis(raw: str | None) -> tuple[str, ...]:
    """Parse comma-separated KPI names from CLI input."""
    if raw is None:
        return tuple(DEFAULT_KPIS)
    parsed = tuple(part.strip() for part in raw.split(",") if part.strip())
    if not parsed:
        raise ValueError("--kpis must contain at least one KPI name")
    return parsed


def main() -> None:
    """CLI main function.

    Parses command-line arguments, builds benchmark configuration, runs the
    benchmark, and prints or writes JSON output.
    """
    _configure_warning_filters()

    parser = argparse.ArgumentParser(
        prog="grid2benchmark",
        description="Run a Grid2Op algorithm benchmark",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Run a benchmark against a Grid2Op environment")
    run_p.add_argument("--algorithm", required=True, help="Path to algorithm .py file")
    run_p.add_argument(
        "--env", default=None, help="Grid2Op environment name (single-scenario mode)"
    )
    run_p.add_argument(
        "--time-series",
        default=None,
        help="Comma-separated time series ids for --env mode (default: all)",
    )
    run_p.add_argument(
        "--topology-file",
        default=None,
        help="Path to pandapower topology file (.json or .xlsx)",
    )
    run_p.add_argument(
        "--time-series-dir",
        default=None,
        help="Path to Grid2Op-style chronics directory",
    )
    run_p.add_argument(
        "--backend",
        default=None,
        choices=list(SUPPORTED_BACKENDS),
        help=(
            "Backend simulator to use. One of: "
            + ", ".join(SUPPORTED_BACKENDS)
            + ". Default: pandapower when --topology-file is set, "
            "Grid2Op built-in default otherwise."
        ),
    )
    run_p.add_argument(
        "--scenarios",
        default=None,
        help="Path to scenario JSON file",
    )
    run_p.add_argument(
        "--kpis",
        default=",".join(DEFAULT_KPIS),
        help="Comma-separated KPI names to evaluate",
    )
    run_p.add_argument(
        "--max-steps", type=int, default=DEFAULT_MAX_STEPS, help="Max steps per episode"
    )
    run_p.add_argument(
        "--output",
        default=None,
        help="Write JSON result to this file (default: stdout)",
    )

    args = parser.parse_args()

    if args.scenarios and args.env:
        raise ValueError("Use either --scenarios or --env, not both")

    if args.scenarios:
        scenarios = _load_scenarios(Path(args.scenarios))
    else:
        scenarios = (
            ScenarioConfig(
                env_name=args.env or DEFAULT_ENV_NAME,
                time_series_ids=_parse_time_series_ids(args.time_series),
                topology=(
                    TopologySource(
                        format="pandapower",
                        path=Path(args.topology_file),
                    )
                    if args.topology_file
                    else None
                ),
                time_series=(
                    TimeSeriesSource(
                        format="grid2op_chronics_dir",
                        path=Path(args.time_series_dir),
                    )
                    if args.time_series_dir
                    else None
                ),
                backend=args.backend,
            ),
        )

    config = BenchmarkConfig(
        scenarios=scenarios,
        max_steps=args.max_steps,
        kpis=_parse_kpis(args.kpis),
    )

    result = run_benchmark(Path(args.algorithm), config)
    result_json = json.dumps(result, indent=2)

    if args.output:
        Path(args.output).write_text(result_json, encoding="utf-8")
        print(f"Results written to {args.output}")
    else:
        print(result_json)


if __name__ == "__main__":
    main()

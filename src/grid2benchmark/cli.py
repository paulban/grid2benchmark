from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

from ._config import (
    DEFAULT_ENV_NAME,
    DEFAULT_EPISODES,
    DEFAULT_MAX_STEPS,
    BenchmarkConfig,
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


def main() -> None:
    _configure_warning_filters()

    parser = argparse.ArgumentParser(
        prog="grid2benchmark",
        description="Run a Grid2Op algorithm benchmark",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Run a benchmark against a Grid2Op environment")
    run_p.add_argument("--algorithm", required=True, help="Path to algorithm .py file")
    run_p.add_argument(
        "--env", default=DEFAULT_ENV_NAME, help="Grid2Op environment name"
    )
    run_p.add_argument(
        "--episodes", type=int, default=DEFAULT_EPISODES, help="Number of episodes"
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

    config = BenchmarkConfig(
        env_name=args.env,
        max_steps=args.max_steps,
        episodes=args.episodes,
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

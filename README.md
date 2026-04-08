# grid2benchmark

grid2benchmark is a lightweight benchmarking harness for Grid2Op agents. It is under development for the AI-EFFECT project.
It lets you:

- load a submitted algorithm module (`build_agent(env, context)`),
- run one or multiple benchmark scenarios,
- evaluate selected KPIs using grid2evaluate,
- export JSON results that include per-scenario outputs and an aggregated summary.

Current implementation status for external data inputs:

- topology from file: implemented for pandapower (`.json`, `.xlsx`)
- time series from file: implemented for Grid2Op chronics directory

## Features

- Scenario-based benchmarking (`ScenarioConfig`)
- Explicit topology/time-series sources per scenario (`TopologySource`, `TimeSeriesSource`)
- Time-series selection per scenario (`time_series_ids`)
- KPI filtering (`carbon_intensity`, `operation_score`, `topological_action_complexity`)
- Python API for orchestration systems
- CLI entry point for local runs
- Structured JSON output suitable for storage and downstream analytics

## Installation

This repository uses `uv`.

```bash
uv sync
```

If you need development dependencies:

```bash
uv sync --group dev
```

## Quick Start (CLI)

Run a single scenario:

```bash
uv run python -m grid2benchmark.cli run \
	--algorithm examples/algorithm_template.py \
	--env l2rpn_case14_sandbox \
	--topology-file ./data/grid.json \
	--time-series-dir ./data/chronics \
	--time-series 0,1 \
	--kpis carbon_intensity,operation_score \
	--max-steps 100 \
	--output results.json
```

Run with a scenario file:

```bash
uv run python -m grid2benchmark.cli run \
	--algorithm examples/algorithm_template.py \
	--scenarios scenarios.json
```

## Quick Start (Python API)

```python
from pathlib import Path

from grid2benchmark import (
		BenchmarkConfig,
		ScenarioConfig,
		TimeSeriesSource,
		TopologySource,
		run_benchmark,
)

result = run_benchmark(
		Path("examples/algorithm_template.py"),
		BenchmarkConfig(
				scenarios=(
						ScenarioConfig(
								env_name="l2rpn_case14_sandbox",
								topology=TopologySource(
										format="pandapower",
										path=Path("./data/grid.json"),
								),
								time_series=TimeSeriesSource(
										format="grid2op_chronics_dir",
										path=Path("./data/chronics"),
								),
								time_series_ids=(0, 1),
						),
				),
				max_steps=100,
				kpis=("carbon_intensity", "operation_score"),
		),
)

print(result["summary"])
```

## Algorithm Contract

Your algorithm module must expose:

```python
def build_agent(env, context):
		...
```

The returned object must provide:

```python
def act(self, observation):
		...
```

`act` may also accept `(observation, reward)` or `(observation, reward, done)`.

### Context Structure

`context` contains benchmark and scenario metadata:

```python
{
	"benchmark": {
		"max_steps": int,
		"kpis": list[str],
		"scenario_index": int,
	},
	"scenario": {
		"env_name": str,
		"topology": {"format": str, "path": str} | None,
		"time_series": {"format": str, "path": str} | None,
		"time_series_ids": list[int],
	},
}
```

## Source Formats (Phase 1)

- `TopologySource.format`:
	- `pandapower` — topology file path (`.json`)
- `TimeSeriesSource.format`:
	- `grid2op_chronics_dir` — path to a Grid2Op chronics directory

## Backend Selection

Choose the power-flow simulator via the `backend` field on `ScenarioConfig` or the
`--backend` CLI flag. When omitted, the default is `pandapower` if a topology file
is provided, otherwise Grid2Op applies its own built-in default.

| Backend | Install required | Description |
|---|---|---|
| `pandapower` | included | Default pandapower solver |
| `lightsim2grid` | `pip install lightsim2grid` | Fast C++ simulator, same grid format as pandapower |
| `pypowsybl` | `pip install pypowsybl2grid` | PowSyBl-based solver; accepts `.json` files (converted internally) and all formats supported by `pp.network.load()` |

All three backends can load pandapower `.json` topology files.  If a backend
package is not installed, a clear `ImportError` with an install hint is raised at
run time — not at import time.

### CLI example

```bash
# Run with the lightweight LightSim2Grid backend
grid2benchmark run \
  --algorithm examples/algorithm_template.py \
  --env rte_case118_example \
  --topology-file data/grid.json \
  --time-series-dir data/chronics \
  --backend lightsim2grid
```

### Python API example

```python
from grid2benchmark import run_benchmark, BenchmarkConfig, ScenarioConfig, TopologySource

result = run_benchmark(
    "examples/algorithm_template.py",
    BenchmarkConfig(
        scenarios=[
            ScenarioConfig(
                env_name="rte_case118_example",
                topology=TopologySource(format="pandapower", path="data/grid.json"),
                backend="pypowsybl",
            )
        ]
    ),
)
```

### Scenario file with backend

```json
[
	{
		"env_name": "rte_case118_example",
		"backend": "lightsim2grid",
		"topology": {
			"format": "pandapower",
			"path": "./data/grid.json"
		},
		"time_series": {
			"format": "grid2op_chronics_dir",
			"path": "./data/chronics"
		}
	}
]
```

## Scenario File Format

`--scenarios` accepts either a top-level list or an object with a `scenarios` key.

```json
[
	{
		"env_name": "l2rpn_case14_sandbox",
		"topology": {
			"format": "pandapower",
			"path": "./data/grid.json"
		},
		"time_series": {
			"format": "grid2op_chronics_dir",
			"path": "./data/chronics"
		},
		"time_series_ids": [0, 1]
	}
]
```

## Output Shape

The benchmark returns:

- `scenarios`: list of per-scenario results
- `summary`: aggregate statistics across scenarios

Each scenario entry includes:

- `scenario_index`
- `environment`
- `executed_time_series_ids`
- `episodes`
- `kpis`

## Testing

Run unit tests:

```bash
uv run pytest -m "not integration" -v
```

Run integration tests:

```bash
uv run pytest -m integration -v
```

## Automatic Documentation

All core modules, classes, and functions include docstrings so tools like Sphinx,
pdoc, and mkdocstrings can generate API documentation automatically.

Recommended API modules:

- `grid2benchmark`
- `grid2benchmark._config`
- `grid2benchmark._algorithm`
- `grid2benchmark._runner`
- `grid2benchmark._kpi`
- `grid2benchmark.cli`

## Next steps
- give grid data to benchmark as input files in different structures (cgmes, powsybl) --> need to always go via grid2op pandapower format conversion or possible to stay in pypowsbl?
- also pass time series data to benchmark --> possible to have different form from chronics? what is typical?
- add further kpis --> in grid2evaluate? updated versions here?
- are inputs and outputs final? how to input chronics in CLI? able to compress whole folder? how to do in workflow?
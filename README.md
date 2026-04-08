# grid2benchmark

grid2benchmark is a lightweight benchmarking harness for Grid2Op agents. It is developed for the AI-EFFECT project.
It lets you:

- load a submitted algorithm module (`build_agent(env, context)`),
- run one or multiple benchmark scenarios,
- evaluate selected KPIs (manual + grid2evaluate),
- export JSON results that include per-scenario outputs and an aggregated summary.

## Features

- Scenario-based benchmarking (`ScenarioConfig`)
- Time-series selection per scenario (`time_series_ids`)
- KPI filtering (`survival`, `violations`, `latency`, `carbon_intensity`, `operation_score`, `topological_action_complexity`)
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
	--time-series 0,1 \
	--kpis survival,violations,latency \
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

from grid2benchmark import BenchmarkConfig, ScenarioConfig, run_benchmark

result = run_benchmark(
		Path("examples/algorithm_template.py"),
		BenchmarkConfig(
				scenarios=(
						ScenarioConfig(
								env_name="l2rpn_case14_sandbox",
								time_series_ids=(0, 1),
						),
				),
				max_steps=100,
				kpis=("survival", "latency"),
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
		"env_path": str | None,
		"time_series_ids": list[int],
	},
}
```

## Scenario File Format

`--scenarios` accepts either a top-level list or an object with a `scenarios` key.

```json
[
	{
		"env_name": "l2rpn_case14_sandbox",
		"time_series_ids": [0, 1],
		"env_path": null
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

"""Source adapters for topology and time-series scenario inputs."""

from __future__ import annotations

import tempfile
from typing import Any

from ._config import ScenarioConfig
from ._converter import build_env_dir, needs_conversion


def _resolve_backend(scenario: ScenarioConfig) -> Any:
    """Return the Grid2Op backend instance for the scenario."""
    backend_name = scenario.backend

    if backend_name is None:
        if scenario.topology is not None:
            from grid2op.Backend import PandaPowerBackend  # type: ignore

            return PandaPowerBackend()
        return None

    if backend_name == "pandapower":
        from grid2op.Backend import PandaPowerBackend  # type: ignore

        return PandaPowerBackend()

    if backend_name == "lightsim2grid":
        try:
            from lightsim2grid import LightSimBackend  # type: ignore

            return LightSimBackend()
        except ImportError as exc:
            raise ImportError(
                "lightsim2grid is required for the 'lightsim2grid' backend. "
                "Install it with: pip install lightsim2grid"
            ) from exc

    if backend_name == "pypowsybl":
        try:
            from pypowsybl2grid import PyPowSyBlBackend  # type: ignore

            return PyPowSyBlBackend()
        except ImportError as exc:
            raise ImportError(
                "pypowsybl2grid is required for the 'pypowsybl' backend. "
                "Install it with: pip install pypowsybl2grid"
            ) from exc

    raise ValueError(f"Unknown backend: {backend_name!r}")  # pragma: no cover


def build_make_kwargs(scenario: ScenarioConfig) -> dict[str, Any]:
    """Build kwargs for grid2op.make based on scenario source config."""
    make_kwargs: dict[str, Any] = {"test": True}

    backend = _resolve_backend(scenario)
    if backend is not None:
        make_kwargs["backend"] = backend

    if scenario.topology is not None and scenario.topology.format == "pandapower":
        make_kwargs["grid_path"] = str(scenario.topology.path)

    if (
        scenario.time_series is not None
        and scenario.time_series.format == "grid2op_chronics_dir"
    ):
        make_kwargs["chronics_path"] = str(scenario.time_series.path)

    return make_kwargs


def prepare_scenario(
    scenario: ScenarioConfig,
) -> tuple[str, dict[str, Any], tempfile.TemporaryDirectory | None]:  # type: ignore[type-arg]
    """Prepare env name, kwargs and optional temp directory for a scenario."""
    if needs_conversion(scenario.topology, scenario.time_series):
        result = build_env_dir(scenario.topology, scenario.time_series)
        make_kwargs: dict[str, Any] = {"test": True, **result.extra_make_kwargs}

        # Explicit scenario backend overrides converter defaults.
        explicit_backend = _resolve_backend(scenario)
        if explicit_backend is not None:
            make_kwargs["backend"] = explicit_backend

        return result.tmp_dir.name, make_kwargs, result.tmp_dir

    return scenario.env_name, build_make_kwargs(scenario), None

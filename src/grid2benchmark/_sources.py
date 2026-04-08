"""Source adapters for topology and time-series scenario inputs.

This module translates high-level ScenarioConfig sources to keyword arguments
accepted by ``grid2op.make``.
"""

from __future__ import annotations

from typing import Any

from ._config import ScenarioConfig


def _resolve_backend(scenario: ScenarioConfig) -> Any:
    """Return the Grid2Op backend instance for the scenario.

    When ``scenario.backend`` is ``None`` and a topology source is provided the
    default ``PandaPowerBackend`` is used, preserving the pre-backend-selection
    behaviour.  When no topology is provided and no explicit backend is set,
    ``None`` is returned so Grid2Op can apply its own default.

    Args:
        scenario: Scenario configuration with optional ``backend`` and
            ``topology`` fields.

    Returns:
        An instantiated Grid2Op backend, or ``None``.

    Raises:
        ImportError: If the requested backend package is not installed.
    """
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
                "Install it with: pip install pypowsybl2grid "
                "or: uv add pypowsybl2grid"
            ) from exc

    # Validation in ScenarioConfig.__post_init__ should prevent this.
    raise ValueError(f"Unknown backend: {backend_name!r}")  # pragma: no cover


def build_make_kwargs(scenario: ScenarioConfig) -> dict[str, Any]:
    """Build kwargs for ``grid2op.make`` based on scenario source config.

    Args:
        scenario: Scenario configuration with optional topology, time-series,
            and backend fields.

    Returns:
        Dictionary of keyword arguments consumable by ``grid2op.make``.
    """
    make_kwargs: dict[str, Any] = {"test": True}

    backend = _resolve_backend(scenario)
    if backend is not None:
        make_kwargs["backend"] = backend

    if scenario.topology is not None:
        if scenario.topology.format == "pandapower":
            make_kwargs["grid_path"] = str(scenario.topology.path)

    if scenario.time_series is not None:
        if scenario.time_series.format == "grid2op_chronics_dir":
            make_kwargs["chronics_path"] = str(scenario.time_series.path)

    return make_kwargs

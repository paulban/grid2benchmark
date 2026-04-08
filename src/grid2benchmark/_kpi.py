"""KPI evaluation logic for benchmark episode outputs.

This module combines:

- manual KPIs computed directly from episode metadata,
- optional grid2evaluate KPIs computed from EnvRecorder outputs.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ._config import AVAILABLE_KPI_NAMES

logger = logging.getLogger(__name__)


def _compute_manual_kpis(
    episode_results: list[dict[str, Any]],
    selected_kpis: set[str],
) -> dict[str, Any]:
    """Compute manual KPIs directly from episode-level runtime data.

    Args:
        episode_results: Episode dictionaries produced by the runner.
        selected_kpis: KPI names requested by the caller.

    Returns:
        Dictionary with the subset of manual KPI blocks requested.
    """
    total_steps = sum(e["steps"] for e in episode_results)
    total_violations = sum(e["overload_violations"] for e in episode_results)
    total_runtime = sum(e["runtime_seconds"] for e in episode_results)

    manual: dict[str, Any] = {}

    if "survival" in selected_kpis:
        manual["survival"] = {
            "episode_lengths": [e["steps"] for e in episode_results],
            "average_episode_length": total_steps / len(episode_results),
        }

    if "violations" in selected_kpis:
        manual["violations"] = {
            "overload_violations_per_episode": [
                e["overload_violations"] for e in episode_results
            ],
            "total_overload_violations": total_violations,
        }

    if "latency" in selected_kpis:
        manual["latency"] = {
            "runtime_seconds_per_episode": [
                e["runtime_seconds"] for e in episode_results
            ],
            "total_runtime_seconds": total_runtime,
            "average_runtime_seconds": total_runtime / len(episode_results),
        }

    return manual


def evaluate_kpis(
    record_directory: Path,
    episode_results: list[dict[str, Any]],
    kpis: tuple[str, ...],
) -> dict[str, Any]:
    """Evaluate selected KPIs using manual and optional grid2evaluate backends.

    Args:
        record_directory: Directory containing EnvRecorder files.
        episode_results: Episode dictionaries returned by simulation.
        kpis: Names of KPIs to compute.

    Returns:
        KPI dictionary containing requested metrics and an
        ``evaluation_backend`` marker.

    Raises:
        ValueError: If unknown KPI names are requested.
    """
    selected_kpis = set(kpis)
    invalid_kpis = selected_kpis - set(AVAILABLE_KPI_NAMES)
    if invalid_kpis:
        raise ValueError(
            f"Unknown KPI(s): {sorted(invalid_kpis)}. Allowed values: {AVAILABLE_KPI_NAMES}"
        )

    result_kpis = _compute_manual_kpis(episode_results, selected_kpis)

    selected_g2e_kpis = {
        "carbon_intensity",
        "operation_score",
        "topological_action_complexity",
    } & selected_kpis

    if not selected_g2e_kpis:
        result_kpis["evaluation_backend"] = "manual_only"
        return result_kpis

    try:
        from grid2evaluate.carbon_intensity_kpi import CarbonIntensityKpi  # type: ignore
        from grid2evaluate.operation_score_kpi import OperationScoreKpi  # type: ignore
        from grid2evaluate.topological_action_complexity_kpi import (  # type: ignore
            TopologicalActionComplexityKpi,
        )

        errors: dict[str, str] = {}
        g2e_metrics: dict[str, Any] = {}

        for name, kpi_cls in [
            ("carbon_intensity", CarbonIntensityKpi),
            ("operation_score", OperationScoreKpi),
            ("topological_action_complexity", TopologicalActionComplexityKpi),
        ]:
            if name not in selected_g2e_kpis:
                continue
            try:
                g2e_metrics[name] = kpi_cls().evaluate(record_directory)
            except Exception as exc:
                errors[name] = str(exc)

        if g2e_metrics:
            result_kpis.update(g2e_metrics)
            result_kpis["evaluation_backend"] = (
                "grid2evaluate" if not errors else "grid2evaluate_partial"
            )
            if errors:
                result_kpis["grid2evaluate_errors"] = errors
            return result_kpis

        logger.warning(
            "All grid2evaluate KPIs failed for record_dir=%s", record_directory
        )

    except Exception as exc:
        logger.warning("grid2evaluate evaluation failed: %s", exc)

    result_kpis["evaluation_backend"] = "fallback_manual"
    return result_kpis

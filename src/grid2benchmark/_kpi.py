from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _compute_manual_kpis(episode_results: list[dict[str, Any]]) -> dict[str, Any]:
    total_steps = sum(e["steps"] for e in episode_results)
    total_violations = sum(e["overload_violations"] for e in episode_results)
    total_runtime = sum(e["runtime_seconds"] for e in episode_results)

    return {
        "survival": {
            "episode_lengths": [e["steps"] for e in episode_results],
            "average_episode_length": total_steps / len(episode_results),
        },
        "violations": {
            "overload_violations_per_episode": [
                e["overload_violations"] for e in episode_results
            ],
            "total_overload_violations": total_violations,
        },
        "latency": {
            "runtime_seconds_per_episode": [
                e["runtime_seconds"] for e in episode_results
            ],
            "total_runtime_seconds": total_runtime,
            "average_runtime_seconds": total_runtime / len(episode_results),
        },
    }


def evaluate_kpis(
    record_directory: Path,
    episode_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute KPIs from EnvRecorder parquet files plus manual episode statistics."""
    kpis = _compute_manual_kpis(episode_results)

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
            try:
                g2e_metrics[name] = kpi_cls().evaluate(record_directory)
            except Exception as exc:
                errors[name] = str(exc)

        if g2e_metrics:
            kpis["grid2evaluate"] = g2e_metrics
            kpis["evaluation_backend"] = (
                "grid2evaluate" if not errors else "grid2evaluate_partial"
            )
            if errors:
                kpis["grid2evaluate_errors"] = errors
            return kpis

        logger.warning(
            "All grid2evaluate KPIs failed for record_dir=%s", record_directory
        )

    except Exception as exc:
        logger.warning("grid2evaluate evaluation failed: %s", exc)

    kpis["evaluation_backend"] = "fallback_manual"
    return kpis

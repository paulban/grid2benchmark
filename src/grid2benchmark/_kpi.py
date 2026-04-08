"""KPI evaluation logic powered by grid2evaluate.

The benchmark forwards requested KPI names to grid2evaluate and returns the
raw KPI payloads unchanged.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ._config import AVAILABLE_KPI_NAMES

logger = logging.getLogger(__name__)


def evaluate_kpis(
    record_directory: Path,
    episode_results: list[dict[str, Any]],
    kpis: tuple[str, ...],
) -> dict[str, Any]:
    """Evaluate selected KPIs using grid2evaluate only.

    Args:
        record_directory: Directory containing EnvRecorder files.
        episode_results: Episode dictionaries returned by simulation.
            Present for API compatibility; not used directly.
        kpis: Names of KPIs to compute.

    Returns:
        KPI dictionary containing requested grid2evaluate metrics and an
        ``evaluation_backend`` marker set to ``"grid2evaluate"``.

    Raises:
        ValueError: If unknown KPI names are requested.
        RuntimeError: If grid2evaluate cannot be imported or KPI computation fails.
    """
    del episode_results

    selected_kpis = set(kpis)
    invalid_kpis = selected_kpis - set(AVAILABLE_KPI_NAMES)
    if invalid_kpis:
        raise ValueError(
            f"Unknown KPI(s): {sorted(invalid_kpis)}. Allowed values: {AVAILABLE_KPI_NAMES}"
        )

    try:
        from grid2evaluate.carbon_intensity_kpi import CarbonIntensityKpi  # type: ignore
        from grid2evaluate.operation_score_kpi import OperationScoreKpi  # type: ignore
        from grid2evaluate.topological_action_complexity_kpi import (  # type: ignore
            TopologicalActionComplexityKpi,
        )
    except Exception as exc:
        raise RuntimeError(
            "grid2evaluate is required to evaluate KPIs in this version"
        ) from exc

    kpi_class_map = {
        "carbon_intensity": CarbonIntensityKpi,
        "operation_score": OperationScoreKpi,
        "topological_action_complexity": TopologicalActionComplexityKpi,
    }

    result_kpis: dict[str, Any] = {}
    errors: dict[str, str] = {}

    for name in sorted(selected_kpis):
        try:
            result_kpis[name] = kpi_class_map[name]().evaluate(record_directory)
        except Exception as exc:
            errors[name] = str(exc)

    if errors:
        logger.error(
            "grid2evaluate KPI evaluation failed for %s on record_dir=%s",
            sorted(errors),
            record_directory,
        )
        raise RuntimeError(f"grid2evaluate KPI evaluation failed for {sorted(errors)}")

    result_kpis["evaluation_backend"] = "grid2evaluate"
    return result_kpis

"""Cortes cronológicos para validación walk-forward."""

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, timedelta


@dataclass(frozen=True)
class WalkForwardSplit:
    """Describe una ventana de entrenamiento y su horizonte de validación."""

    cutoff: date
    validation_start: date
    validation_end: date


def walk_forward_splits(
    dates: Iterable[date],
    *,
    test_start: date,
    horizon: int,
    step_size: int,
    n_windows: int,
) -> list[WalkForwardSplit]:
    """Construye cortes ascendentes sin usar el período de test.

    ``cutoff`` es el último día disponible para entrenar. La validación
    comienza al día siguiente y contiene exactamente ``horizon`` días.
    ``dates`` debe corresponder a una sola serie temporal, no a una tabla
    larga con varias estaciones.
    """
    date_list = list(dates)
    if not date_list:
        raise ValueError("dates no puede estar vacío")
    if any(
        left >= right
        for left, right in zip(date_list, date_list[1:], strict=False)
    ):
        raise ValueError("dates debe estar ordenado y no contener duplicados")
    if horizon <= 0 or step_size <= 0 or n_windows <= 0:
        raise ValueError("horizon, step_size y n_windows deben ser positivos")

    last_cutoff = test_start - timedelta(days=horizon + 1)
    available_dates = [value for value in date_list if value <= last_cutoff]
    if not available_dates:
        raise ValueError("No hay fechas suficientes antes del período de test")

    latest_cutoff = available_dates[-1]
    first_cutoff = latest_cutoff - timedelta(days=step_size * (n_windows - 1))
    cutoffs = [
        first_cutoff + timedelta(days=step_size * index)
        for index in range(n_windows)
    ]
    if cutoffs[0] < date_list[0]:
        raise ValueError("No hay fechas suficientes para n_windows")

    splits = [
        WalkForwardSplit(
            cutoff=cutoff,
            validation_start=cutoff + timedelta(days=1),
            validation_end=cutoff + timedelta(days=horizon),
        )
        for cutoff in cutoffs
    ]
    if splits[-1].validation_end >= test_start:
        raise ValueError("Una ventana de validación invade el período de test")
    return splits

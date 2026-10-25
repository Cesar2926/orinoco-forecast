"""Naive (persistence) and SeasonalNaive baselines via StatsForecast."""

from typing import Any

import polars as pl
from statsforecast import StatsForecast
from statsforecast.models import Naive, SeasonalNaive


class BaselinesRunner:
    """Persistencia y "lo mismo que el año pasado"; solo usan ``y``."""

    name = "baselines"
    model_columns = ["Naive", "SeasonalNaive"]

    def cross_validation(
        self, frame: pl.DataFrame, cfg: dict[str, Any]
    ) -> pl.DataFrame:
        models = [Naive(), SeasonalNaive(season_length=cfg["seasonality"])]
        # With polars input StatsForecast requires a polars offset ("1d"), not "D".
        sf = StatsForecast(models=models, freq="1d")
        return sf.cross_validation(
            h=cfg["horizon"],
            df=frame.select("unique_id", "ds", "y"),
            n_windows=cfg["n_windows"],
            step_size=cfg["step_size"],
            refit=cfg["refit"],
        )

"""Runner catalog: structure-only runners, canonical keys and aliases."""

from datetime import date

import polars as pl
import pytest

from orinoco_forecast.models.neural.lstm import NsmtRunner
from orinoco_forecast.models.neural.transformer import TransformerRunner
from orinoco_forecast.models.registry import (
    available_runners,
    get_runner,
)

CUTOFF = date(2022, 12, 17)
HORIZON = 3


def _windows() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "unique_id": ["ciudad_bolivar", "ciudad_bolivar"],
            "origin": [CUTOFF, CUTOFF],
            "target_1": [1.0, 1.0],
            "target_2": [1.0, 1.0],
            "target_3": [1.0, 1.0],
        }
    )


def _frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "unique_id": ["ciudad_bolivar"],
            "ds": [CUTOFF],
            "y": [10.0],
        },
        schema={"unique_id": pl.String, "ds": pl.Date, "y": pl.Float64},
    )


def test_available_runners_are_the_canonical_catalog() -> None:
    assert available_runners() == [
        "arimax",
        "booster",
        "cnn_lstm",
        "naive",
        "nsmt",
        "seasonal_naive",
        "tcn",
        "timesfm3",
        "transformer",
    ]


def test_registry_resolves_aliases() -> None:
    assert get_runner("arima").name == "arimax"
    assert get_runner("lstm").name == "nsmt"
    assert get_runner("timefm3").name == "timesfm3"
    assert get_runner("timesfm").name == "timesfm3"
    assert get_runner("tcn").name == "tcn"


def test_implemented_flags_are_explicit() -> None:
    assert get_runner("naive").implemented is True
    assert get_runner("seasonal_naive").implemented is True
    assert get_runner("arimax").implemented is True
    assert get_runner("booster").implemented is True
    assert get_runner("tcn").implemented is True
    assert get_runner("nsmt").implemented is False
    assert get_runner("transformer").implemented is False
    assert get_runner("timesfm3").implemented is True


def test_unknown_runner_fails() -> None:
    with pytest.raises(KeyError, match="no registrado"):
        get_runner("does_not_exist")


@pytest.mark.parametrize(
    ("runner", "config_key"),
    [
        (NsmtRunner, "hidden_size"),
        (TransformerRunner, "d_model"),
    ],
)
def test_structure_only_runners_load_their_config(runner, config_key) -> None:
    instance = runner()
    assert config_key in instance.params
    assert instance.implemented is False


@pytest.mark.parametrize("name", ["nsmt", "transformer"])
def test_structure_only_runners_raise_a_clear_error(name) -> None:
    runner = get_runner(name)
    with pytest.raises(NotImplementedError, match="estructura"):
        runner.fit(_windows(), _frame(), cutoff=CUTOFF, horizon=HORIZON)
    with pytest.raises(NotImplementedError, match="estructura"):
        runner.predict(_frame(), cutoff=CUTOFF, horizon=HORIZON)

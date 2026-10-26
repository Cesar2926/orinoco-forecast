"""Shared fixtures: the raw workbook is read once per test session."""

import warnings
from dataclasses import dataclass

import polars as pl
import pytest

from orinoco_forecast.data.loading import load_primary
from orinoco_forecast.features.dataset import build_model_frame, load_levels


@dataclass(frozen=True)
class Loaded:
    frame: pl.DataFrame
    messages: list[str]


def _load_recording(loader) -> Loaded:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        frame = loader()
    return Loaded(frame, [str(w.message) for w in caught])


@pytest.fixture(scope="session")
def primary_loaded() -> Loaded:
    return _load_recording(load_primary)


@pytest.fixture(scope="session")
def primary(primary_loaded: Loaded) -> pl.DataFrame:
    return primary_loaded.frame


@pytest.fixture(scope="session")
def levels(primary: pl.DataFrame) -> pl.DataFrame:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return load_levels(primary=primary)


def _model_frame(stage: str, levels: pl.DataFrame) -> pl.DataFrame:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return build_model_frame(stage, levels=levels)


@pytest.fixture(scope="session")
def model_frame_a(levels: pl.DataFrame) -> pl.DataFrame:
    return _model_frame("A", levels)


@pytest.fixture(scope="session")
def model_frame_b(levels: pl.DataFrame) -> pl.DataFrame:
    return _model_frame("B", levels)

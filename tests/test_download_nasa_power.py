"""NASA POWER basin points: download helpers and the files kept in data/external."""

import importlib.util
import urllib.parse
from pathlib import Path

import pytest

from orinoco_forecast.data.exogenous import load_nasa_power
from orinoco_forecast.data.loading import load_config, repo_root

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "download_nasa_power.py"
spec = importlib.util.spec_from_file_location("download_nasa_power", SCRIPT)
download = importlib.util.module_from_spec(spec)
spec.loader.exec_module(download)

SETTINGS = {"community": "AG", "start": "19810101", "end": "20251231"}


def test_point_url_has_point_and_span() -> None:
    url = download.point_url(5.62, -61.78, SETTINGS)
    query = dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(url).query))
    assert query == {
        "parameters": "PRECTOTCORR",
        "community": "AG",
        "latitude": "5.62",
        "longitude": "-61.78",
        "start": "19810101",
        "end": "20251231",
        "format": "CSV",
    }


@pytest.mark.parametrize(
    "text",
    ['{"messages": ["error"]}', "-BEGIN HEADER-\n-END HEADER-\nYEAR,DOY,T2M\n"],
)
def test_validate_payload_rejects_unexpected(text: str) -> None:
    with pytest.raises(ValueError):
        download.validate_payload(text)


def test_basin_points_are_downloaded_and_readable() -> None:
    points = load_config()["exogenous"]["nasa_power_basin"]["points"]
    files = {n: f"data/external/nasa_power_{n}.csv" for n in points}
    missing = [f for f in files.values() if not (repo_root() / f).is_file()]
    assert not missing, "Ejecutar scripts/download_nasa_power.py"
    rain = load_nasa_power(files)
    assert rain.columns == ["ds", *(f"rain_{n}" for n in points)]
    assert rain.height == 16436  # 1981-01-01 .. 2025-12-31.

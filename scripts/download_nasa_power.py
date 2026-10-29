"""Download NASA POWER daily rainfall for the basin points in configs/data.yaml.

Usage: ``uv run python scripts/download_nasa_power.py [--force] [name ...]``.
Reads ``exogenous.nasa_power_basin`` and writes
``data/external/nasa_power_<name>.csv`` in the same CSV layout as the station
files, so ``data.exogenous.load_nasa_power`` reads them unchanged. Existing
files are kept unless ``--force``; each file is written only after its payload
is validated.
"""

import argparse
import time
import urllib.parse
import urllib.request
from typing import Any

from orinoco_forecast.data.loading import load_config, repo_root

API = "https://power.larc.nasa.gov/api/temporal/daily/point"
VARIABLE = "PRECTOTCORR"


def point_url(lat: float, lon: float, settings: dict[str, Any]) -> str:
    """Daily point API URL, CSV, local solar time (the API default)."""
    query = {
        "parameters": VARIABLE,
        "community": settings["community"],
        "latitude": lat,
        "longitude": lon,
        "start": settings["start"],
        "end": settings["end"],
        "format": "CSV",
    }
    return f"{API}?{urllib.parse.urlencode(query)}"


def validate_payload(text: str) -> None:
    """Reject responses that ``load_nasa_power`` could not read."""
    if not text.startswith("-BEGIN HEADER-"):
        raise ValueError("La respuesta no empieza con el encabezado de NASA POWER.")
    if f"YEAR,DOY,{VARIABLE}" not in text:
        raise ValueError(f"La respuesta no contiene la tabla YEAR,DOY,{VARIABLE}.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "names", nargs="*", help="Puntos a descargar (por defecto todos)."
    )
    parser.add_argument(
        "--force", action="store_true", help="Reemplaza archivos existentes."
    )
    args = parser.parse_args()

    settings = load_config()["exogenous"]["nasa_power_basin"]
    points = settings["points"]
    unknown = sorted(set(args.names) - set(points))
    if unknown:
        parser.error(f"Puntos no declarados en data.yaml: {unknown}")
    out_dir = repo_root() / "data" / "external"
    for name in args.names or points:
        path = out_dir / f"nasa_power_{name}.csv"
        if path.exists() and not args.force:
            print(f"{name}: ya existe, se omite ({path.name}).")
            continue
        url = point_url(points[name]["lat"], points[name]["lon"], settings)
        with urllib.request.urlopen(url, timeout=120) as response:
            text = response.read().decode("utf-8")
        validate_payload(text)
        tmp = path.with_suffix(".csv.tmp")
        tmp.write_text(text, encoding="utf-8", newline="\n")
        tmp.replace(path)
        print(f"{name}: {text.count(chr(10))} líneas -> {path.name}")
        time.sleep(1)  # One request per point; POWER asks clients not to hammer it.


if __name__ == "__main__":
    main()

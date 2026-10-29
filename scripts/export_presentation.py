"""Export the model comparison as offline files for the presentation.

Usage: ``uv run python scripts/export_presentation.py [--out reports/presentation]``.

Reads the versioned results (``reports/predictions/`` for the 24 validation folds
and ``reports/test/`` for the final test) through the dashboard helpers, so the
numbers are the same as in the dashboard. Writes CSV rankings, PNG charts and a
``resumen.md`` with the tables, usable without the dashboard or network.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import polars as pl  # noqa: E402

from orinoco_forecast.dashboard import data  # noqa: E402
from orinoco_forecast.data.loading import repo_root  # noqa: E402

STATIONS = {"ciudad_bolivar": "Ciudad Bolívar", "palua": "Palúa"}


def bar_chart(per_station: pl.DataFrame, title: str, path: Path) -> None:
    """MAE por modelo (ordenado por MAE medio) con una barra por estación."""
    order = (
        per_station.group_by("model")
        .agg(pl.col("mae").mean())
        .sort("mae", descending=True)["model"]
        .to_list()
    )
    fig, ax = plt.subplots(figsize=(9, 0.55 * len(order) + 1.5))
    height = 0.38
    for offset, (station, label) in enumerate(STATIONS.items()):
        rows = per_station.filter(pl.col("unique_id") == station)
        mae = dict(zip(rows["model"], rows["mae"], strict=True))
        ys = [i + (offset - 0.5) * height for i in range(len(order))]
        bars = ax.barh(ys, [mae.get(m, 0.0) for m in order], height, label=label)
        ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=8)
    ax.set_yticks(range(len(order)), order)
    ax.set_xlabel("MAE, días 1 a 14 (m)")
    ax.set_title(title)
    ax.set_xlim(0, per_station["mae"].max() * 1.15)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def horizon_chart(errors: pl.DataFrame, title: str, path: Path) -> None:
    """MAE por día de horizonte; Naive punteado."""
    fig, ax = plt.subplots(figsize=(9, 5))
    for model in sorted(errors["model"].unique()):
        rows = errors.filter(pl.col("model") == model).sort("horizon")
        style = "--" if model == data.BASELINES else "-"
        ax.plot(rows["horizon"], rows["mae"], style, marker="o", ms=3, label=model)
    ax.set_xlabel("Día de horizonte (días después del corte)")
    ax.set_ylabel("MAE (m)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def markdown(ranking: pl.DataFrame) -> str:
    """Tabla Markdown del ranking (MAE en metros, tres decimales)."""
    lines = [
        "| # | Modelo | Gana a Naive | MAE medio | Ciudad Bolívar | Palúa |",
        "|---|---|---|---|---|---|",
    ]
    for row in ranking.iter_rows(named=True):
        beats = {True: "sí", False: "no"}.get(row["beats_naive"], "referencia")
        lines.append(
            f"| {row['rank']} | {row['model']} | {beats} | {row['mean_mae']:.3f} | "
            f"{row['mae_ciudad_bolivar']:.3f} | {row['mae_palua']:.3f} |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out", type=Path, default=repo_root() / "reports" / "presentation"
    )
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    root = repo_root() / "reports" / "predictions"
    comparison = data.load_fold_comparison(root, data.discover_models(root))
    if comparison.is_empty():
        raise SystemExit(f"No hay resultados de 24 cortes en {root}.")
    sections = []
    val_stations = data.fold_station_mae(comparison)
    val_ranking = data.rank_stations(val_stations)
    val_ranking.write_csv(args.out / "ranking_validacion.csv")
    val_stations.write_csv(args.out / "mae_validacion_por_estacion.csv")
    bar_chart(
        val_stations,
        "Validación: 24 cortes (2021–2022)",
        args.out / "ranking_validacion.png",
    )
    for station, label in STATIONS.items():
        horizon_chart(
            data.fold_horizon_errors(comparison, station),
            f"Validación, {label}: error por día de horizonte",
            args.out / f"horizonte_validacion_{station}.png",
        )
    sections.append(
        "## Validación: 24 cortes walk-forward (2021-01-26 a 2022-12-17)\n\n"
        + markdown(val_ranking)
    )

    test_stations = data.load_test_stations()
    if test_stations.is_empty():
        print("Sin resultados de test en reports/test/; se exporta solo la validación.")
    else:
        test_ranking = data.rank_stations(test_stations)
        test_ranking.write_csv(args.out / "ranking_test.csv")
        test_stations.write_csv(args.out / "mae_test_por_estacion.csv")
        bar_chart(
            test_stations,
            "Test final: 52 ventanas de 14 días (2023–2024)",
            args.out / "ranking_test.png",
        )
        sections.append(
            "## Test final: 52 ventanas de 14 días (2023-01-02 a 2024-12-29)\n\n"
            + markdown(test_ranking)
        )

    summary = (
        "# Comparativa de modelos\n\nMAE en metros, días 1 a 14. Criterio: "
        "`rel_mae < 1` en cada estación y, entre esos, el menor MAE medio.\n\n"
        + "\n\n".join(sections)
        + "\n"
    )
    (args.out / "resumen.md").write_text(summary, encoding="utf-8")
    print(summary)
    print(f"Archivos en {args.out}")


if __name__ == "__main__":
    main()

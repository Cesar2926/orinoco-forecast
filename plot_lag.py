import csv
import statistics
from datetime import datetime, timedelta

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

CSV = "niveles_orinoco.csv"
PNG = "niveles_ultimos_2_anios.png"
ESTACIONES = ["ayacucho", "caicara", "ciudad_bolivar", "palua"]  # río arriba → río abajo

with open(CSV, encoding="utf-8") as f:
    filas = list(csv.reader(f))

datos = [
    (datetime.strptime(r[0], "%Y/%m/%d"), [float(v) if v else None for v in r[1:]])
    for r in filas[1:]
]
inicio = datos[-1][0] - timedelta(days=730)
datos = [d for d in datos if d[0] >= inicio]
fechas = [f for f, _ in datos]

fig, ejes = plt.subplots(4, 1, sharex=True, figsize=(12, 9), layout="constrained")
for eje, est in zip(ejes, ESTACIONES):
    vals = [v[ESTACIONES.index(est)] for _, v in datos]
    eje.plot(fechas, vals, lw=1.2)
    eje.set_ylabel(est)
    eje.grid(True, axis="x", alpha=0.3)

    # ponytail: pico = máximo local en ±30 días sobre media+0.5σ; basta para la crecida anual
    limpios = [v for v in vals if v is not None]
    umbral = statistics.mean(limpios) + 0.5 * statistics.pstdev(limpios)
    marcados = []
    for i, v in enumerate(vals):
        if v is None or v < umbral:
            continue
        ventana = [x for x in vals[max(0, i - 30):i + 31] if x is not None]
        if v == max(ventana) and all(abs(i - j) > 30 for j in marcados):
            marcados.append(i)
            eje.plot(fechas[i], v, "ro", ms=5)
            eje.annotate(
                fechas[i].strftime("%d %b"),
                (fechas[i], v),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=8,
                color="red",
            )

ejes[0].set_title("Niveles del Orinoco — últimos 2 años (río arriba → río abajo)")
ejes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
ejes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ejes[-1].set_xlabel("Fecha")
fig.supylabel("Nivel (m)")
fig.savefig(PNG, dpi=150)
print(PNG)

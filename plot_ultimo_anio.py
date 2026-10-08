import csv
from datetime import datetime, timedelta

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

CSV = "niveles_orinoco.csv"
PNG = "niveles_ultimo_anio.png"
ESTACIONES = ["ayacucho", "caicara", "ciudad_bolivar", "palua"]  # río arriba → río abajo

with open(CSV, encoding="utf-8") as f:
    filas = list(csv.reader(f))

datos = [
    (datetime.strptime(r[0], "%Y/%m/%d"), [float(v) if v else None for v in r[1:]])
    for r in filas[1:]
]
inicio = datos[-1][0] - timedelta(days=365)
datos = [d for d in datos if d[0] >= inicio]
fechas = [f for f, _ in datos]

fig, eje = plt.subplots(figsize=(12, 6), layout="constrained")
for i, est in enumerate(ESTACIONES):
    eje.plot(fechas, [v[i] for _, v in datos], lw=1.2, label=est)

eje.set_title("Niveles del Orinoco — último año (río arriba → río abajo)")
eje.set_xlabel("Fecha")
eje.set_ylabel("Nivel (m)")
eje.legend()
eje.grid(True, alpha=0.3)
eje.xaxis.set_major_locator(mdates.MonthLocator())
eje.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
fig.savefig(PNG, dpi=150)
print(PNG)

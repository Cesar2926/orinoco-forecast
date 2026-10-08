import csv
from datetime import datetime

import openpyxl

SRC = "Niveles Orinoco.xlsx"
DST = "niveles_orinoco.csv"

wb = openpyxl.load_workbook(SRC, read_only=True)
ws = wb[wb.sheetnames[0]]

with open(DST, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    rows = ws.iter_rows(values_only=True)
    writer.writerow(next(rows))  # header: fecha, ayacucho, caicara, ciudad_bolivar, palua
    for fecha, *valores in rows:
        writer.writerow([fecha.strftime("%Y/%m/%d")] + [float(v) if v is not None else "" for v in valores])

print(f"{DST}: {ws.max_row - 1} filas")

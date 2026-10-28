# Resultados de validación guardados (para no volver a ejecutar los modelos)

`data/predictions/` está excluida de git, así que aquí se guarda una copia de los
resultados con el mismo nombre de archivos y el mismo formato que escribe
`scripts/run_model.py`. Son datos de **validación**: el test final (2023 en adelante)
no se ha tocado.

## Estructura

```text
reports/
  baselines/naive/predictions.parquet     # baseline Naive en el corte 2022-12-17
  predictions/<modelo>/                   # booster, arimax, tcn, cnn_lstm
    predictions.parquet                   # un corte (2022-12-17), 14 días
    statistics.parquet                    # métricas de ese corte contra Naive
    backtest/naive/                       # barrido de 24 cortes contra Naive
      comparison.parquet                  # resumen por estación, modelo y paso
      fold_statistics.parquet             # métricas por corte
      fold_predictions.parquet            # predicciones por corte
```

Los resultados de Naive del barrido de 24 cortes no tienen archivo aparte: van dentro de
`fold_predictions.parquet` y `fold_statistics.parquet` como filas del baseline.

## Hardware con el que se generó cada archivo

| Modelo | Un corte | Barrido de 24 cortes |
|---|---|---|
| `booster` | CPU | CPU |
| `arimax` | CPU | CPU (583 s con la CPU compartida) |
| `tcn` | CPU | **GPU** (Colab, T4) |
| `cnn_lstm` | CPU | `backtest/naive`: CPU; `backtest/naive_gpu`: GPU (Colab) |

La GPU no es determinista, por lo que una nueva ejecución en GPU puede diferir un poco de
estos archivos. La comparación CPU contra GPU del CNN-LSTM está en
`docs/modelos_arboles_cnn.md`.

## Cómo usarlos con los scripts del proyecto

Los scripts leen de `data/predictions/`. Para usar estos resultados sin ejecutar nada:

```bash
cp -r reports/predictions/* data/predictions/
cp -r reports/baselines/* data/predictions/
uv run python scripts/validate_predictions.py booster --against naive
```

La carpeta `backtest/naive_gpu` del CNN-LSTM no la lee ningún script: es una copia para
poder comparar hardware.

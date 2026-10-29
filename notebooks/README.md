# Cuadernos

| Cuaderno | Qué cuenta |
|---|---|
| [`01_xgboost_booster.ipynb`](01_xgboost_booster.ipynb) | Proceso del XGBoost (`booster`): datos, partición, diseño, Optuna, validación de 24 cortes y limitaciones |
| [`02_tcn.ipynb`](02_tcn.ipynb) | Proceso de la red convolucional TCN, validación de 24 cortes (GPU) y el experimento de covariables |
| [`03_cnn_lstm.ipynb`](03_cnn_lstm.ipynb) | Proceso del híbrido CNN-LSTM, validación de 24 cortes (CPU y GPU) y comparación con los otros modelos |
| [`04_transformer.ipynb`](04_transformer.ipynb) | Proceso del Transformer propio (encoder de PyTorch sobre el pipeline del CNN-LSTM), Optuna y validación de 24 cortes (CPU) |
| [`colab_folds.ipynb`](colab_folds.ipynb) | Ejecuta `run_model.py <modelo> --folds` en Google Colab con GPU |

## Cómo usarlos

- Cada cuaderno explica cada decisión de diseño con el mismo orden: qué problema resuelve, cómo
  funciona por dentro, qué alternativa se descartó y cuándo sería un error.
- Las celdas de código son **lecturas ligeras** (polars, YAML y redes diminutas con datos
  aleatorios): no entrenan modelos ni escriben en `data/` o `artifacts/`.
- Cada celda avisa con un mensaje, sin fallar, si falta un archivo. Para generarlos, ver la
  sección "Cómo reproducir" de cada cuaderno.
- Se abren desde la raíz del repositorio o desde esta carpeta (buscan `pyproject.toml` hacia
  arriba). En VS Code hay que elegir el intérprete `.venv` del proyecto como kernel; si falta
  `ipykernel`, VS Code ofrece instalarlo.
- Los cuadernos se guardan sin salidas. Las cifras citadas coinciden con
  [`docs/modelos_arboles_cnn.md`](../docs/modelos_arboles_cnn.md) y con los CSV de
  `reports/tuning/`.

# Cómo correr el proyecto en la presentación

Todo lo que se muestra sale de resultados ya versionados. **No hay que entrenar nada el
día de la presentación**:

- `reports/predictions/<modelo>/`: validación de 24 cortes, 2021–2022;
- `reports/test/`: test final 2023–2024, evaluado una sola vez.

## 1. Preparar el equipo (una vez, con red)

```bash
git clone https://github.com/Cesar2926/orinoco-forecast.git
cd orinoco-forecast
uv sync --extra timesfm   # TimesFM 3 corre en vivo; descarga ~1,3 GB de pesos la primera vez
uv run python scripts/export_presentation.py   # respaldo sin red en reports/presentation/
```

Las tres redes (TCN, CNN-LSTM y Transformer) tardan minutos por pronóstico en una CPU, así
que sus pronósticos se calculan en Colab con GPU:

1. Abrir `notebooks/colab_folds.ipynb` en Colab con un entorno de GPU.
2. Correr las celdas de instalación.
3. Poner `PACK = True` en la celda «Paquete de la presentación» y ejecutarla.
4. Copiar el archivo descargado a `reports/live/forecasts.parquet`.

Antes de exponer, abre una vez cada comparación que piensas mostrar en «Probar
predicción». El dashboard la guarda en caché y en la presentación aparece al instante.

## 2. Abrir el dashboard

```bash
uv run streamlit run app/dashboard.py
```

Si existe una carpeta `data/predictions/` con corridas viejas, el dashboard la lee primero.
Para forzar los resultados oficiales:

```bash
ORINOCO_PREDICTIONS_DIR=reports/predictions uv run streamlit run app/dashboard.py
```

En PowerShell: `$env:ORINOCO_PREDICTIONS_DIR="reports/predictions"; uv run streamlit run app/dashboard.py`.

## 3. Recorrido sugerido

| Pestaña | Qué mostrar |
|---|---|
| Datos | La serie de Ciudad Bolívar y Palúa, y por qué Naive (repetir el último nivel) es difícil de batir en un río lento. |
| **Comparativa de modelos** | El modelo recomendado y por qué, el % de mejora de cada modelo sobre Naive por estación, la tabla de decisión (precisión, consistencia, costo, riesgos) y el mapa de calor por día de horizonte. |
| **Test final** | El mismo ranking en 2023–2024 (52 ventanas), con el puesto que cada modelo tenía en validación al lado. Es el resultado que se reporta. |
| **Probar predicción** | Los seis modelos pronosticando la misma fecha. Naive, ARIMAX, XGBoost y TimesFM 3 corren en vivo; las redes salen del paquete de Colab. Muestra quién acertó mejor en esa ventana. |
| Pronóstico por ventana | Los pronósticos guardados de la validación, corte por corte. |

Criterio de selección, el mismo en las dos pestañas: el modelo debe tener `rel_mae < 1`
(mejor que Naive) en **cada** estación y, entre los que cumplen, gana el menor MAE medio
entre Ciudad Bolívar y Palúa.

## 4. Respaldo sin red

`reports/presentation/` contiene:

- `resumen.md`: tablas de ranking de validación y de test, listas para copiar a las diapositivas;
- `ranking_validacion.png` y `ranking_test.png`: MAE por modelo y estación;
- `horizonte_validacion_<estación>.png`: error por día de horizonte;
- `ranking_*.csv` y `mae_*_por_estacion.csv`: los mismos números en tablas.

## 5. Cómo se generó el test final (no repetir)

```bash
uv run python scripts/make_report.py --eval test --windows 0 --step-days 14 --fit-once \
  --models arimax booster cnn_lstm transformer timesfm3 --output reports/test/ranking_main.csv
uv run python scripts/make_report.py --eval test --windows 0 --step-days 14 --fit-once \
  --models tcn --output reports/test/ranking_tcn.csv
uv run python scripts/make_report.py --eval test --windows 0 --step-days 14 --fit-once \
  --models seasonal_naive --output reports/test/ranking_baselines.csv
```

- **Ventanas:** 52 de 14 días, sin solaparse, con cortes del 2023-01-01 al 2024-12-15.
- **Entrenamiento:** cada red y TimesFM se entrenan una vez con los datos hasta el primer corte. `booster`, `arimax` y `tcn` se reajustan en cada corte.
- **Selección:** el modelo se eligió con la validación antes de correr el test. El test solo informa cuánto rinde en datos nuevos; volver a elegir con él sesgaría el resultado.

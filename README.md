# Orinoco Forecast

Scaffold para explorar pronósticos de niveles diarios del río Orinoco con fuentes versionadas, validación cronológica y varios modelos comparables. Este repositorio comenzó como `Proyecto-IA-Orinoco`; los Excel se conservan como fuentes y se retiraron los scripts, CSV generados y gráficos exploratorios heredados.

> **Alcance actual:** carpetas, configuración, entorno y archivos Python con placeholders. No se implementan los modelos, el entrenamiento, el ensamble, la carga de datos ni las métricas.

## Modelos previstos

| Modelo | Biblioteca | Rol en este proyecto |
|---|---|---|
| ARIMA | statsmodels / pmdarima | Previsto; solo existe el archivo placeholder |
| LSTM | PyTorch | Previsto; solo existe el archivo placeholder |
| Árbol / boosting | XGBoost | Elegido sobre LightGBM; solo existe el placeholder |
| CNN 1D | PyTorch | Prevista; solo existe el archivo placeholder |
| Transformer | PyTorch | Previsto; solo existe el archivo placeholder |
| TimesFM 3.0 | `timesfm[torch]` opcional | Previsto para inferencia solamente; sin adaptador implementado |

## Entorno y comandos

Se requiere Python 3.12 (fijado en `.python-version`) y `uv`. El stack de tablas es **Polars** (sin pandas) y el de redes es **PyTorch**.

```bash
uv sync
uv run python -c "import torch, polars, statsmodels"
uv run pytest
uv run ruff check .
uv run ruff format .
```

Por el alcance de scaffold, pytest encuentra dos pruebas placeholder y las omite;
todavía no hay pruebas de comportamiento.

Los pesos de TimesFM 3.0 no se descargan con la instalación normal. Para añadir el paquete de inferencia opcional:

```bash
uv sync --extra timesfm
```

El checkpoint `google/timesfm-3.0-pytorch` tiene la licencia **TimesFM Non-Commercial License v1.0**: los pesos autoalojados son solo para uso no comercial y no productivo. Consulta el [README oficial](https://github.com/google-research/timesfm#license-notice-for-pretrained-weights) y la [tarjeta del modelo](https://huggingface.co/google/timesfm-3.0-pytorch). La licencia del código fuente no sustituye la licencia de los pesos.

## Fuentes de datos

- `data/raw/` contiene entradas originales inmutables. El código solo las lee; las transformaciones futuras deben guardarse en `data/interim/` o `data/processed/`.
- `configs/data.yaml` registra las fuentes previstas. `src/orinoco_forecast/data/loading.py` es solo un placeholder; la lectura y validación aún no están implementadas.
- Para declarar una fuente: conserva el original en `data/raw/` y añade en `configs/data.yaml` `name`, `path`, `sheet`, `date_column`, `target_column` y `freq`; agrega `covariates` o `skiprows` si aplica.
- El libro histórico está organizado por años en columnas; `layout: wide_years` y `date_parts` documentan la conversión prevista, todavía no implementada. `dataset_imputado_simpleml.csv.xls` parece Excel por su sufijo, pero es CSV y coincide con la fuente diaria XLSX.

## Protocolo de comparación

`configs/backtest.yaml` es la única fuente de `horizon` y `context_length`, compartidos por todos los modelos. Los YAML describen el protocolo walk-forward y los parámetros previstos; no hay evaluador ni modelos implementados.

Los archivos de `scripts/` también son placeholders: todavía no son CLIs ejecutables.

## Estructura

```text
configs/                    # datos, backtest, modelos y ensamble
 data/                      # raw versionado; interim/processed/predictions ignorados
 notebooks/
  src/orinoco_forecast/      # módulos placeholder
  scripts/                   # archivos CLI placeholder
 tests/
 artifacts/                 # checkpoints locales, fuera de git
 reports/figures/
```

## Makefile

Objetivos disponibles: `make setup`, `make test`, `make lint`, `make format` y `make clean`.

### Windows local

`make` no viene preinstalado en Windows. Opciones:

1. Ejecutar los comandos uv equivalentes directamente (ver sección "Entorno y comandos"): `uv sync` = `make setup`, `uv run pytest` = `make test`, `uv run ruff check .` = `make lint`, `uv run ruff format .` = `make format`.
2. Instalar GNU Make una sola vez (`choco install make` o `scoop install make`) y usar `make <objetivo>` desde PowerShell igual que en Linux.

### Google Colab / Linux

`make` ya está disponible; `uv` no. En la primera celda del notebook:

```python
!curl -LsSf https://astral.sh/uv/install.sh | sh
!make setup
!make test
!make lint
```

En celdas siguientes conviene `%%bash` + `make <objetivo>`. `uv sync` descarga automáticamente el Python 3.12 gestionado por uv.

Cada target del `Makefile` solo envuelve los mismos comandos uv, así que el resultado es idéntico en local y en Colab.

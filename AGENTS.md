# AGENTS.md — Orinoco Forecast

Instrucciones para agentes de IA que trabajen en este repositorio. El proyecto es un scaffold de forecasting de niveles diarios del río Orinoco gestionado con `uv` (Python 3.12). El stack de tablas es **Polars** (no pandas) y el stack de redes es **PyTorch**.

## Estado del proyecto (leer antes de escribir código)

- Los módulos Python son **placeholders**: `NotImplementedError("TODO")` y tests omitidos a propósito.
- No implementes modelos, entrenamiento, ensamble, carga de datos ni métricas sin instrucción explícita del usuario.
- `data/raw/` es inmutable: ningún código escribe ahí. Derivados van a `data/interim/` o `data/processed/`; predicciones a `data/predictions/<modelo>/`; checkpoints a `artifacts/`.
- Anomalías de datos (duplicados, desorden, nulos, 29 de febrero ausente) se reportan con warnings; no se corrigen en silencio.

## Skills registradas

Carga la skill completa antes de trabajar en esas áreas:

| Skill | Cuándo activarla | Ruta |
|---|---|---|
| `polars-best-practices` | Carga, limpieza o transformación de datos con Polars | `.agents/skills/polars-best-practices/SKILL.md` |
| `torch-best-practices` | Cualquier código PyTorch (modelos, entrenamiento, inferencia) | `.agents/skills/torch-best-practices/SKILL.md` |

## Comandos base (uv)

```bash
uv sync                 # preparar entorno (.venv) y dependencias
uv run pytest           # tests (por ahora, 2 placeholders omitidos)
uv run ruff check .     # lint
uv run ruff format .    # formato
```

Dependencia opcional de inferencia: `uv sync --extra timesfm`. Los pesos `google/timesfm-3.0-pytorch` son de licencia no comercial / no productiva; no se descargan por defecto.

## Makefile en este proyecto

Objetivos: `make setup`, `make test`, `make lint`, `make format`, `make clean`.

### Windows local

`make` no viene instalado en Windows por defecto. Opciones:

1. Ejecutar los comandos uv directamente (equivalentes 1:1 a cada target):
   - `uv sync` (= `make setup`)
   - `uv run pytest` (= `make test`)
   - `uv run ruff check .` (= `make lint`)
   - `uv run ruff format .` (= `make format`)
2. Instalar GNU Make una sola vez (`choco install make` o `scoop install make`) y usar `make <objetivo>` igual que en Linux.

### Google Colab / Linux

`make` ya está disponible; `uv` no. En la primera celda:

```python
!curl -LsSf https://astral.sh/uv/install.sh | sh
!make setup
!make test
!make lint
```

Celdas siguientes pueden usar `%%bash` + `make <objetivo>`. `uv sync` descargará Python 3.12 gestionado por uv automáticamente.

## Convenciones

- Mensajes de warning y docstrings en español; identificadores de código en inglés.
- Commits pequeños y descriptivos estilo `chore:`/`feat:`/`docs:`; no hacer push a ramas principales.
- Configuración compartida (`horizon`, `step_size`, `n_windows`, `test_start`) vive solo en `configs/backtest.yaml`; los hiperparámetros por modelo (incluida la ventana de entrada) en `configs/models/`.
- Los archivos de `data/raw/` están versionados en git y los tests los leen directamente: no se editan, mueven ni borran sin instrucción.

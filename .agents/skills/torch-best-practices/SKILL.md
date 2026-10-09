---
name: torch-best-practices
description: "Trigger: torch, PyTorch, LSTM, CNN, Transformer, entrenamiento, epochs, checkpoint, dataloader, GPU. Aplica las reglas del proyecto al usar PyTorch en Orinoco Forecast."
license: Apache-2.0
metadata:
  author: "Cesar2926"
  version: "1.0"
---

# Torch Best Practices (Orinoco Forecast)

## Activation Contract

Activa esta skill al escribir o revisar cualquier código PyTorch del proyecto (modelos en `src/orinoco_forecast/models/neural/`, entrenamiento en `training/`, o inferencia).

## Hard Rules

- Respetar el alcance actual: los modelos son placeholders que lanzan `NotImplementedError("TODO")`. No implementes redes, entrenamiento ni ensamble sin instrucción explícita del usuario.
- Series de tiempo: prohibido barajar (`shuffle`) observaciones. Splits cronológicos con `configs/backtest.yaml` como única fuente de `horizon` y `context_length`.
- Fija semillas de `random`, NumPy y torch (incluida CUDA) al inicio de cada experimento.
- Código agnóstico de dispositivo: resuelve `device` automáticamente (`cuda` → `mps` → `cpu`); nunca asumas GPU disponible.
- Checkpoints solo en `artifacts/`; no versionar ni escribir `*.pt` fuera de esa carpeta.
- Nada escribe en `data/raw/`; predicciones van a `data/predictions/<modelo>/`.

## Decision Gates

| Necesidad | Elección |
|---|---|
| Definir una red | `nn.Module` con `forward` documentado; hiperparámetros solo desde `configs/models/<modelo>.yaml` |
| Dataset de ventanas | Construido de forma cronológica; ventana = `context_length`, salida = `horizon` |
| Entrenamiento | `training/trainer.py` + callbacks de `training/callbacks.py`; logs reproducibles |
| Guardar modelo | `torch.save`/`state_dict` bajo `artifacts/<modelo>/` |
| Reproducibilidad | Llamar a `set_seed(seed)` antes de crear tensores/dataloaders |

## Execution Steps

1. Lee `configs/models/<modelo>.yaml` y `configs/backtest.yaml`; nunca introduzcas horizonte o contexto propios.
2. Configura semilla y dispositivo antes de construir tensores.
3. Implementa `fit`/`predict` de la clase `Forecaster` correspondiente solo cuando el usuario lo autorice.
4. Verifica con `uv run pytest` y `uv run ruff check .`.

## Output Contract

Reporta: dispositivo usado, semilla, shapes de tensores/ventanas y rutas de checkpoints. Declara explícitamente cualquier desviación del YAML.

## References

- `configs/backtest.yaml` — protocolo compartido.
- `configs/models/` — hiperparámetros por modelo.
- `.agents/skills/polars-best-practices/SKILL.md` — reglas de datos previas al entrenamiento.

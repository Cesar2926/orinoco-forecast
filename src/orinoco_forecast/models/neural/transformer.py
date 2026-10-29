"""Temporal Transformer encoder in PyTorch, one network per seed.

Hyperparameters live only in ``configs/models/transformer.yaml``. Data handling,
scaling, training loop, early stopping and prediction are the CNN-LSTM ones
(``CnnLstmRunner``); only the network differs, so both models are compared on
exactly the same windows, scaling and validation split.
"""

from pathlib import Path
from typing import Any

import torch
import yaml
from torch import nn

from orinoco_forecast.data.loading import repo_root
from orinoco_forecast.models.neural.cnn_lstm import CnnLstmRunner

MODEL = "TRANSFORMER"


def load_transformer_config(path: str | Path | None = None) -> dict[str, Any]:
    """Read ``configs/models/transformer.yaml`` (or ``path``)."""
    path = (
        Path(path) if path else repo_root() / "configs" / "models" / "transformer.yaml"
    )
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


class TemporalTransformer(nn.Module):
    """Linear embedding + learned positions -> Transformer encoder -> linear head.

    ``forward`` takes ``(batch, days, features)`` and returns ``(batch, horizon)``
    scaled values read from the last day's token. No causal mask is needed: the
    whole window is history up to the cutoff.
    """

    def __init__(
        self,
        *,
        n_features: int,
        horizon: int,
        input_size: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if d_model % nhead:
            raise ValueError(
                f"d_model ({d_model}) debe ser múltiplo de nhead ({nhead})."
            )
        self.embed = nn.Linear(n_features, d_model)
        self.position = nn.Parameter(torch.zeros(1, input_size, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers, enable_nested_tensor=False
        )
        self.head = nn.Linear(d_model, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder(self.embed(x) + self.position)
        return self.head(hidden[:, -1])


class TransformerRunner(CnnLstmRunner):
    """Transformer global (todas las series del frame), una red por semilla.

    Columnas ``TRANSFORMER_s<semilla>``; la carpeta es ``transformer``. Entradas,
    ``target_transform``, escalado por ventana y validación son los del
    ``CnnLstmRunner``.
    """

    name = "transformer"
    model_prefix = MODEL
    implemented = True

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        super().__init__(params or load_transformer_config())

    def _build_net(self, n_features: int, horizon: int) -> nn.Module:
        p = self.params
        return TemporalTransformer(
            n_features=n_features,
            horizon=horizon,
            input_size=p["input_size"],
            d_model=p["d_model"],
            nhead=p["nhead"],
            num_layers=p["num_layers"],
            dim_feedforward=p["dim_feedforward"],
            dropout=p.get("dropout", 0.0),
        )

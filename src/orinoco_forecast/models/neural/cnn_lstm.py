"""Hybrid CNN-LSTM in PyTorch, one network per seed.

Hyperparameters live only in ``configs/models/cnn_lstm.yaml``. The runner fits
one cutoff, predicts 1..14 days and returns only a ``PredictionResult``.
"""

from copy import deepcopy
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Self

import numpy as np
import polars as pl
import torch
import yaml
from numpy.lib.stride_tricks import sliding_window_view
from torch import nn

from orinoco_forecast.data.loading import repo_root
from orinoco_forecast.features.dataset import NON_FEATURE_COLUMNS, drop_covariates
from orinoco_forecast.models.base import (
    DEFAULT_HORIZON,
    PredictionResult,
    validate_horizon,
    validate_windows,
)

KEYS = ("unique_id", "ds", "y")
MODEL = "CNNLSTM"
TRANSFORMS = ("level", "difference")
# Floor for the window IQR so a flat input window does not blow up the scaling.
MIN_SCALE = 1e-3


def load_cnn_lstm_config(path: str | Path | None = None) -> dict[str, Any]:
    """Lee ``configs/models/cnn_lstm.yaml`` (o ``path``)."""
    path = Path(path) if path else repo_root() / "configs" / "models" / "cnn_lstm.yaml"
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_device() -> torch.device:
    """``cuda`` -> ``mps`` -> ``cpu``; never assumes a GPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class CnnLstm(nn.Module):
    """Causal Conv1d blocks -> LSTM -> linear head.

    ``forward`` takes ``(batch, days, features)`` and returns ``(batch, horizon)``
    scaled values. ``conv`` works on ``(batch, features, days)`` and keeps the
    length: each block pads on the left, so an output at day t only sees days
    up to t.
    """

    def __init__(
        self,
        *,
        n_features: int,
        horizon: int,
        conv_channels: int,
        kernel_size: int,
        conv_layers: int,
        lstm_hidden_size: int,
        lstm_layers: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        blocks: list[nn.Module] = []
        channels = n_features
        for _ in range(conv_layers):
            blocks += [
                nn.ConstantPad1d((kernel_size - 1, 0), 0.0),
                nn.Conv1d(channels, conv_channels, kernel_size),
                nn.ReLU(),
            ]
            channels = conv_channels
        self.conv = nn.Sequential(*blocks)
        # nn.LSTM only applies dropout between stacked layers.
        self.lstm = nn.LSTM(
            conv_channels,
            lstm_hidden_size,
            lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(lstm_hidden_size, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv(x.transpose(1, 2)).transpose(1, 2)
        out, _ = self.lstm(features)
        return self.head(self.dropout(out[:, -1]))


def reconstruct_levels(changes: torch.Tensor, y_cut: torch.Tensor) -> torch.Tensor:
    """Levels from daily changes: ``y(cutoff) + cumsum(changes)`` per row."""
    return y_cut[:, None] + changes.cumsum(dim=1)


def robust_scale(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Scale each window with the median/IQR of its own input part.

    Returns the scaled window and the median and IQR of channel 0 (the target),
    which scale the targets and undo the scaling of the predictions.
    """
    probs = torch.tensor([0.25, 0.5, 0.75], dtype=x.dtype, device=x.device)
    q = torch.quantile(x, probs, dim=1)
    median, iqr = q[1], (q[2] - q[0]).clamp_min(MIN_SCALE)
    return (x - median[:, None]) / iqr[:, None], median[:, 0], iqr[:, 0]


def make_windows(
    series: np.ndarray, input_size: int, horizon: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Input windows, targets (channel 0) and the row index where each target ends.

    ``series`` is ``(days, features)`` with the target in column 0.
    """
    days = series.shape[0]
    count = days - horizon - input_size + 1
    if count <= 0:
        return (
            np.empty((0, input_size, series.shape[1]), np.float32),
            np.empty((0, horizon), np.float32),
            np.empty(0, np.int64),
        )
    x = sliding_window_view(series, input_size, axis=0)[:count].transpose(0, 2, 1)
    y = sliding_window_view(series[:, 0], horizon)[input_size : input_size + count]
    ends = np.arange(input_size, input_size + count) + horizon - 1
    return x, y, ends


class CnnLstmRunner:
    """CNN-LSTM global (todas las series del frame), una red por semilla.

    - Columnas ``CNNLSTM_s<semilla>`` en el resultado largo; la carpeta es
      ``cnn_lstm``.
    - Entrada: ``y`` y todas las covariables que dejan ``drop_covariates`` y
      ``NON_FEATURE_COLUMNS``, de los últimos ``input_size`` días hasta el corte.
      No usa covariables futuras (el parquet actual no trae Fourier).
    - ``target_transform: difference``: entrena sobre el cambio diario, agrega el
      nivel actual como covariable ``level_y`` y reconstruye el nivel con la suma
      acumulada desde el nivel del corte.
    - Validación: las ventanas cuyo objetivo cae por completo en los últimos
      ``val_size`` días paran el entrenamiento; las que lo cruzan se descartan.
    """

    name = "cnn_lstm"
    model_prefix = MODEL

    implemented = True

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self.params = params or load_cnn_lstm_config()
        self._nets: list[nn.Module] = []
        self._transform = "level"
        self._horizon = DEFAULT_HORIZON
        self.features: list[str] = []
        self._fitted = False

    @property
    def model_columns(self) -> list[str]:
        return [f"{self.model_prefix}_s{s}" for s in self.params["seeds"]]

    def _prepare(
        self, frame: pl.DataFrame, cutoff: date
    ) -> tuple[pl.DataFrame, list[str]]:
        """History up to the cutoff with the model features, one row per day."""
        history = frame.filter(pl.col("ds") <= cutoff).sort("unique_id", "ds")
        if history.is_empty():
            raise ValueError(f"No hay observaciones hasta el corte {cutoff}.")
        history = drop_covariates(history, self.params.get("drop_covariates", []))
        transform = self.params.get("target_transform", "level")
        if transform not in TRANSFORMS:
            raise ValueError(
                f"target_transform desconocido: {transform!r} {TRANSFORMS}"
            )
        covariates = [
            c for c in history.columns if c not in {*KEYS, *NON_FEATURE_COLUMNS}
        ]
        if transform == "difference":
            history = history.with_columns(
                level_y=pl.col("y"), y=pl.col("y").diff().over("unique_id")
            ).drop_nulls("y")
            covariates.append("level_y")
        features = ["y", *covariates]
        if history.select(features).null_count().sum_horizontal().item():
            raise ValueError("Hay valores nulos en las entradas del CNN-LSTM.")
        return history, features

    def _scaled_loss(
        self, net: nn.Module, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        scaled, median, iqr = robust_scale(x)
        target = (y - median[:, None]) / iqr[:, None]
        return (net(scaled) - target).abs().mean()

    def _build_net(self, n_features: int, horizon: int) -> nn.Module:
        """Network trained per seed; subclasses swap the architecture only."""
        p = self.params
        return CnnLstm(
            n_features=n_features,
            horizon=horizon,
            conv_channels=p["conv_channels"],
            kernel_size=p["kernel_size"],
            conv_layers=p["conv_layers"],
            lstm_hidden_size=p["lstm_hidden_size"],
            lstm_layers=p["lstm_layers"],
            dropout=p.get("dropout", 0.0),
        )

    def _train(
        self,
        seed: int,
        train: tuple[torch.Tensor, torch.Tensor],
        valid: tuple[torch.Tensor, torch.Tensor],
        horizon: int,
        device: torch.device,
    ) -> nn.Module:
        p = self.params
        torch.manual_seed(seed)
        net = self._build_net(train[0].shape[2], horizon).to(device)
        optimiser = torch.optim.Adam(net.parameters(), lr=p["learning_rate"])
        # Seeded shuffle of independent windows; the split itself stays chronological.
        generator = torch.Generator().manual_seed(seed)
        best, best_state, stale = float("inf"), deepcopy(net.state_dict()), 0
        for _ in range(p["max_epochs"]):
            net.train()
            order = torch.randperm(len(train[0]), generator=generator)
            for batch in order.split(p["batch_size"]):
                optimiser.zero_grad()
                loss = self._scaled_loss(net, train[0][batch], train[1][batch])
                loss.backward()
                optimiser.step()
            net.eval()
            with torch.no_grad():
                score = self._scaled_loss(net, *valid).item()
            if score < best:
                best, best_state, stale = score, deepcopy(net.state_dict()), 0
            else:
                stale += 1
                if stale >= p["early_stop_patience"]:
                    break
        net.load_state_dict(best_state)
        return net.cpu().eval()

    def fit(
        self,
        windows: pl.DataFrame,
        frame: pl.DataFrame,
        *,
        cutoff: date,
        horizon: int = DEFAULT_HORIZON,
    ) -> Self:
        validate_horizon(horizon)
        validate_windows(windows, horizon=horizon)
        p = self.params
        val_size = int(p["val_size"])
        if val_size < horizon:
            raise ValueError("La parada temprana requiere val_size >= horizonte.")
        history, features = self._prepare(frame, cutoff)
        train_x, train_y, valid_x, valid_y = [], [], [], []
        for _, group in history.group_by("unique_id", maintain_order=True):
            series = group.select(features).to_numpy().astype(np.float32)
            if val_size >= series.shape[0]:
                raise ValueError(
                    f"val_size ({val_size}) debe ser menor que las filas de "
                    f"entrenamiento ({series.shape[0]})."
                )
            x, y, ends = make_windows(series, p["input_size"], horizon)
            limit = series.shape[0] - val_size
            train = ends < limit
            valid = ends - horizon + 1 >= limit
            train_x.append(x[train])
            train_y.append(y[train])
            valid_x.append(x[valid])
            valid_y.append(y[valid])
        data = [np.concatenate(a) for a in (train_x, train_y, valid_x, valid_y)]
        if not len(data[0]) or not len(data[2]):
            raise ValueError(
                "Faltan ventanas de entrenamiento o validación: reduce input_size "
                "o val_size, o usa más historia."
            )
        device = resolve_device()
        tensors = [torch.from_numpy(a).to(device) for a in data]
        self._nets = [
            self._train(
                seed,
                (tensors[0], tensors[1]),
                (tensors[2], tensors[3]),
                horizon,
                device,
            )
            for seed in p["seeds"]
        ]
        self._transform = p.get("target_transform", "level")
        self._horizon = horizon
        self.features = features
        self._fitted = True
        return self

    def predict(
        self,
        frame: pl.DataFrame,
        *,
        cutoff: date,
        horizon: int = DEFAULT_HORIZON,
    ) -> PredictionResult:
        validate_horizon(horizon)
        if not self._fitted:
            raise RuntimeError("El runner no está ajustado; llama a fit primero.")
        if horizon != self._horizon:
            raise ValueError(
                f"El horizonte de predict ({horizon}) difiere del de fit "
                f"({self._horizon})."
            )
        history, features = self._prepare(frame, cutoff)
        if features != self.features:
            raise ValueError(
                f"Las columnas de entrada cambiaron: {features} != {self.features}."
            )
        size = self.params["input_size"]
        ids, windows = [], []
        for uid, group in history.group_by("unique_id", maintain_order=True):
            if group["ds"].max() != cutoff or group.height < size:
                raise ValueError(
                    f"La serie {uid[0]!r} no tiene {size} días hasta el corte {cutoff}."
                )
            ids.append(uid[0])
            windows.append(group.select(features).tail(size).to_numpy())
        x = torch.from_numpy(np.stack(windows).astype(np.float32))
        levels = (
            frame.filter((pl.col("ds") <= cutoff) & pl.col("y").is_not_null())
            .group_by("unique_id")
            .agg(pl.col("y").sort_by("ds").last())
        )
        y_cut = torch.tensor(
            [levels.filter(pl.col("unique_id") == uid)["y"][0] for uid in ids],
            dtype=torch.float32,
        )
        scaled, median, iqr = robust_scale(x)
        dates = [cutoff + timedelta(days=step) for step in range(1, horizon + 1)]
        rows: dict[str, list] = {
            k: []
            for k in ("model", "unique_id", "cutoff", "ds", "horizon_step", "y_pred")
        }
        for column, net in zip(self.model_columns, self._nets, strict=True):
            with torch.no_grad():
                out = net(scaled) * iqr[:, None] + median[:, None]
            if self._transform == "difference":
                out = reconstruct_levels(out, y_cut)
            for uid, values in zip(ids, out.tolist(), strict=True):
                rows["model"] += [column] * horizon
                rows["unique_id"] += [uid] * horizon
                rows["cutoff"] += [cutoff] * horizon
                rows["ds"] += dates
                rows["horizon_step"] += list(range(1, horizon + 1))
                rows["y_pred"] += values
        long = pl.DataFrame(rows).with_columns(pl.col("horizon_step").cast(pl.Int32))
        return PredictionResult.from_frame(long, runner=self.name, horizon=horizon)

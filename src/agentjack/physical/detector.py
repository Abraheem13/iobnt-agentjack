"""Receiver-side detectors.

Three, in increasing order of what they know about the channel:

* ThresholdDetector      - one number, calibrated from data. The honest floor.
* DecisionFeedbackDetector - knows the CIR, cancels ISI from past decisions.
* GRUDetector            - learns the mapping, knows nothing analytically.

Detection is infrastructure here, not the contribution. It exists so that
attacks and defenses operate on a receiver that is not gratuitously bad: if the
baseline receiver were weak, every attack would look stronger than it is.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..channel.isi import subtract_isi

__all__ = ["ThresholdDetector", "DecisionFeedbackDetector", "GRUDetector", "ber"]


def ber(true_bits: np.ndarray, pred_bits: np.ndarray) -> float:
    true_bits = np.asarray(true_bits, dtype=np.int64)
    pred_bits = np.asarray(pred_bits, dtype=np.int64)
    if len(true_bits) != len(pred_bits):
        raise ValueError("length mismatch")
    return float(np.mean(true_bits != pred_bits))


@dataclass
class ThresholdDetector:
    """Fixed-threshold detection. Threshold fitted on labelled traces."""

    threshold: float | None = None

    def fit(self, observations: np.ndarray, symbols: np.ndarray) -> ThresholdDetector:
        obs = np.asarray(observations, dtype=np.float64).ravel()
        sym = np.asarray(symbols, dtype=np.int64).ravel()
        m1 = obs[sym == 1].mean() if np.any(sym == 1) else 1.0
        m0 = obs[sym == 0].mean() if np.any(sym == 0) else 0.0
        self.threshold = 0.5 * (m0 + m1)
        return self

    def detect(self, observations: np.ndarray) -> np.ndarray:
        if self.threshold is None:
            raise RuntimeError("call fit() first")
        return (np.asarray(observations, dtype=np.float64) > self.threshold).astype(np.int64)


@dataclass
class DecisionFeedbackDetector:
    """Cancels ISI using previously decided symbols and a known CIR."""

    cir: np.ndarray
    n_molecules: float
    threshold: float | None = None

    def __post_init__(self):
        self.cir = np.asarray(self.cir, dtype=np.float64)
        if self.threshold is None:
            self.threshold = 0.5 * self.n_molecules * self.cir[0]

    def detect(self, observations: np.ndarray) -> np.ndarray:
        obs = np.asarray(observations, dtype=np.float64)
        L = len(self.cir)
        decided: list[int] = []
        for k, y in enumerate(obs):
            hist = np.array(decided[::-1][: L - 1], dtype=np.float64)
            clean = subtract_isi(y, hist, self.cir, self.n_molecules) if hist.size else y
            decided.append(int(clean > self.threshold))
        return np.array(decided, dtype=np.int64)


@dataclass
class GRUDetector:
    """Small sequence detector. Sees a causal window of counts, emits one bit.

    Deliberately tiny: this must train in minutes on a laptop, because it is
    infrastructure for the real experiments rather than a result in itself.
    """

    window: int = 12
    hidden_size: int = 32
    num_layers: int = 1
    device: str = "cpu"
    _model: object | None = field(default=None, repr=False)
    _scale: float = 1.0

    def _build(self):
        import torch.nn as nn

        class Net(nn.Module):
            def __init__(self, hidden, layers):
                super().__init__()
                self.gru = nn.GRU(1, hidden, layers, batch_first=True)
                self.head = nn.Linear(hidden, 1)

            def forward(self, x):
                out, _ = self.gru(x)
                return self.head(out[:, -1, :]).squeeze(-1)

        return Net(self.hidden_size, self.num_layers)

    @staticmethod
    def _windows(obs: np.ndarray, window: int) -> np.ndarray:
        padded = np.concatenate([np.zeros(window - 1), np.asarray(obs, dtype=np.float64)])
        return np.lib.stride_tricks.sliding_window_view(padded, window).copy()

    def fit(self, observations: np.ndarray, symbols: np.ndarray, epochs: int = 12,
            lr: float = 3e-3, batch_size: int = 256, seed: int = 0) -> GRUDetector:
        import torch

        torch.manual_seed(seed)
        obs = np.asarray(observations, dtype=np.float64).ravel()
        sym = np.asarray(symbols, dtype=np.int64).ravel()
        self._scale = float(obs.std() + 1e-9)

        X = torch.tensor(self._windows(obs, self.window) / self._scale, dtype=torch.float32).unsqueeze(-1)
        y = torch.tensor(sym, dtype=torch.float32)

        self._model = self._build().to(self.device)
        opt = torch.optim.Adam(self._model.parameters(), lr=lr)
        loss_fn = torch.nn.BCEWithLogitsLoss()

        n = len(y)
        for _ in range(epochs):
            perm = torch.randperm(n)
            for i in range(0, n, batch_size):
                idx = perm[i : i + batch_size]
                opt.zero_grad()
                loss = loss_fn(self._model(X[idx].to(self.device)), y[idx].to(self.device))
                loss.backward()
                opt.step()
        return self

    def detect(self, observations: np.ndarray) -> np.ndarray:
        import torch

        if self._model is None:
            raise RuntimeError("call fit() first")
        obs = np.asarray(observations, dtype=np.float64).ravel()
        X = torch.tensor(self._windows(obs, self.window) / self._scale, dtype=torch.float32).unsqueeze(-1)
        self._model.eval()
        with torch.no_grad():
            logits = self._model(X.to(self.device)).cpu().numpy()
        return (logits > 0).astype(np.int64)

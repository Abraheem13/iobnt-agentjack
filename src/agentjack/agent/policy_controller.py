"""The learned controller: a small GRU trained by behaviour cloning.

This is the PRIMARY quantitative agent. Every headline number in the paper comes
from it, because it is fully deterministic given a seed - unlike an LLM, whose
sampling and version drift would make the central claims unreproducible.

Trained by behaviour cloning on a safe expert rather than by reinforcement
learning. Three reasons, and the third is the important one:

1. It trains in minutes on a laptop, which the 15-day budget requires.
2. Cloning a *stated* expert makes the benign policy auditable - we can say
   exactly what safe behaviour the agent was taught.
3. RL against a reward that penalises unsafe actions would produce an agent that
   ignores the channel whenever the channel disagrees with telemetry. That agent
   is attack-proof by construction, which sounds good and is actually a broken
   experiment: it would make the paper's threat unmeasurable rather than absent.
   The controller must genuinely *use* the message, exactly as a deployed
   gateway would.

The input is the telemetry window plus the decoded message, so the agent has
both a physical view and a channel view. An attack has to make the channel view
win.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..twin.digital_twin import BioCyberTwin, Observation
from .base import BaseAgent

__all__ = ["PolicyController", "expert_action", "collect_demonstrations"]

ACTIONS = ("none", "low_dose", "med_dose", "high_dose")
ACTION_INDEX = {a: i for i, a in enumerate(ACTIONS)}


def expert_action(observation: Observation, target: float = 6.5, isf: float = 2.0,
                  low_cutoff: float = 4.0) -> str:
    """The safe expert the controller clones.

    Trusts the message when it is consistent with telemetry, and falls back to
    its own correction rule when it is not. This is a *sane* gateway, not a
    credulous one - which matters, because an attack that only beats a credulous
    agent proves very little.
    """
    g = observation.glucose
    if g < low_cutoff:
        return "none"

    needed = (g - target) / isf - observation.dose_last_hour * 0.5
    if needed >= 3.5:
        own = "high_dose"
    elif needed >= 2.0:
        own = "med_dose"
    elif needed >= 0.75:
        own = "low_dose"
    else:
        own = "none"

    asked = observation.message.get("opcode", "none")
    if asked not in ACTIONS:
        return own
    # Follow the message unless it asks for materially more than telemetry
    # justifies. One step of over-dosing is tolerated; two is not.
    if ACTION_INDEX[asked] <= ACTION_INDEX[own] + 1:
        return asked
    return own


def collect_demonstrations(twin_factory, n_episodes: int, seed: int = 0,
                           glucose_range: tuple[float, float] = (6.0, 18.0)) -> tuple[np.ndarray, np.ndarray]:
    """Roll out the expert and record (features, action index) pairs.

    Episodes start across a wide glucose range, including the severe
    hyperglycaemia that follows a missed bolus or illness. Without those starts
    the expert never issues its largest dose, that class never appears in
    training, and the controller becomes structurally incapable of emitting it -
    which would make an over-dosing attack unmeasurable for a reason that has
    nothing to do with the attack.
    """
    X, y = [], []
    rng = np.random.default_rng(seed)
    for ep in range(n_episodes):
        twin = twin_factory(seed + ep)
        obs = twin.reset(glucose0=float(rng.uniform(*glucose_range)))
        while True:
            a = expert_action(obs)
            X.append(PolicyController.featurise(obs))
            y.append(ACTION_INDEX[a])
            res = twin.step(a)
            obs = res.observation
            if res.done:
                break
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int64)


@dataclass
class PolicyController(BaseAgent):
    """GRU over the telemetry window, concatenated with message features."""

    name: str = "policy_gru"
    hidden_size: int = 64
    num_layers: int = 1
    device: str = "cpu"
    seed: int = 0
    _model: object | None = field(default=None, repr=False)
    _norm: tuple[float, float] = (7.0, 3.0)

    N_MESSAGE_FEATURES = 7

    @staticmethod
    def featurise(observation: Observation) -> np.ndarray:
        """Telemetry window ++ message features ++ dose history.

        The message contributes a one-hot opcode plus its integrity flags, so the
        controller *can* learn to distrust a frame whose checksum failed. Whether
        it does is an empirical question the attacks will answer.
        """
        tele = np.asarray(observation.telemetry, dtype=np.float32)
        msg = observation.message or {}
        op = msg.get("opcode", "none")
        onehot = np.zeros(len(ACTIONS), dtype=np.float32)
        if op in ACTION_INDEX:
            onehot[ACTION_INDEX[op]] = 1.0
        extras = np.array([
            float(msg.get("checksum_ok", False)),
            float(msg.get("value", 0)) / 200.0,
            float(observation.dose_last_hour) / 8.0,
        ], dtype=np.float32)
        return np.concatenate([tele, onehot, extras])

    def _build(self, window: int):
        import torch.nn as nn

        n_extra = self.N_MESSAGE_FEATURES

        class Net(nn.Module):
            def __init__(self, hidden, layers, n_actions):
                super().__init__()
                self.gru = nn.GRU(1, hidden, layers, batch_first=True)
                self.head = nn.Sequential(
                    nn.Linear(hidden + n_extra, hidden), nn.ReLU(),
                    nn.Linear(hidden, n_actions),
                )

            def forward(self, tele, extra):
                out, _ = self.gru(tele.unsqueeze(-1))
                return self.head(torch.cat([out[:, -1, :], extra], dim=-1))

        import torch  # noqa: F401  (referenced inside forward)

        self._window = window
        return Net(self.hidden_size, self.num_layers, len(ACTIONS))

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 30, lr: float = 3e-3,
            batch_size: int = 256, val_frac: float = 0.2, class_weighted: bool = True,
            verbose: bool = False) -> dict:
        """Behaviour cloning with class weighting.

        Dosing decisions are intrinsically rare - most 15-minute steps correctly
        do nothing - so an unweighted classifier reaches 90%+ accuracy by always
        predicting "none". That agent would appear immune to every attack in the
        paper while actually just being inert, so the loss is weighted by inverse
        class frequency and the reported metric is per-class recall rather than
        overall accuracy.
        """
        import torch

        torch.manual_seed(self.seed)
        window = X.shape[1] - self.N_MESSAGE_FEATURES
        mu, sd = float(X[:, :window].mean()), float(X[:, :window].std() + 1e-6)
        self._norm = (mu, sd)

        Xn = X.copy()
        Xn[:, :window] = (Xn[:, :window] - mu) / sd
        n_val = int(len(Xn) * val_frac)
        rng = np.random.default_rng(self.seed)
        perm = rng.permutation(len(Xn))
        val_idx, tr_idx = perm[:n_val], perm[n_val:]

        tele = torch.tensor(Xn[:, :window])
        extra = torch.tensor(Xn[:, window:])
        yy = torch.tensor(y)

        self._model = self._build(window).to(self.device)
        opt = torch.optim.Adam(self._model.parameters(), lr=lr)

        if class_weighted:
            counts = np.bincount(y, minlength=len(ACTIONS)).astype(np.float64)
            w = np.where(counts > 0, len(y) / (len(ACTIONS) * np.maximum(counts, 1)), 0.0)
            weight = torch.tensor(w, dtype=torch.float32, device=self.device)
        else:
            weight = None
        loss_fn = torch.nn.CrossEntropyLoss(weight=weight)

        history = {"train_loss": [], "val_acc": [], "val_recall": []}
        for _ in range(epochs):
            self._model.train()
            p = torch.randperm(len(tr_idx))
            for i in range(0, len(tr_idx), batch_size):
                idx = torch.tensor(tr_idx)[p[i : i + batch_size]]
                opt.zero_grad()
                loss = loss_fn(self._model(tele[idx], extra[idx]), yy[idx])
                loss.backward()
                opt.step()
            history["train_loss"].append(float(loss.item()))
            self._model.eval()
            with torch.no_grad():
                vi = torch.tensor(val_idx)
                pred = self._model(tele[vi], extra[vi]).argmax(-1)
                acc = (pred == yy[vi]).float().mean()
                recall = []
                for c in range(len(ACTIONS)):
                    mask = yy[vi] == c
                    recall.append(float((pred[mask] == c).float().mean()) if mask.any() else float("nan"))
            history["val_acc"].append(float(acc))
            history["val_recall"].append(recall)
            if verbose:
                rs = " ".join(f"{a}={r:.2f}" for a, r in zip(ACTIONS, recall))
                print(f"  loss {loss.item():.4f}  acc {acc:.4f}  recall {rs}")
        return history

    def assert_trained(self, history: dict, min_recall: float = 0.8) -> None:
        """Refuse to be used as an experimental subject if undertrained.

        An undertrained controller collapses to always-"none". It then scores
        perfectly on safety AND appears immune to every attack - for the same
        reason: it never acts. Measured directly, below 12 epochs the minimum
        per-class recall sits at 0.50 and every attack reports 0% unsafe; at 12
        epochs recall reaches 0.99 and the same attack reports 38%.

        That failure is silent and points the wrong way, so it is checked rather
        than trusted.
        """
        recall = history["val_recall"][-1]
        present = [r for r in recall if r == r]   # drop NaN for absent classes
        if not present or min(present) < min_recall:
            raise RuntimeError(
                f"controller undertrained: min per-class recall {min(present):.2f} "
                f"< {min_recall}. It will look safe and attack-proof because it "
                f"never acts. Train longer before measuring anything."
            )

    def act(self, observation: Observation) -> str:
        import torch

        if self._model is None:
            raise RuntimeError("call fit() or load() first")
        f = self.featurise(observation)
        window = len(f) - self.N_MESSAGE_FEATURES
        mu, sd = self._norm
        tele = torch.tensor(((f[:window] - mu) / sd)[None, :], dtype=torch.float32)
        extra = torch.tensor(f[window:][None, :], dtype=torch.float32)
        self._model.eval()
        with torch.no_grad():
            logits = self._model(tele.to(self.device), extra.to(self.device))
        return ACTIONS[int(logits.argmax(-1).item())]

    def save(self, path: str | Path) -> None:
        import torch

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self._model.state_dict(), "norm": self._norm,
                    "window": self._window, "hidden_size": self.hidden_size,
                    "num_layers": self.num_layers}, path)

    def load(self, path: str | Path) -> "PolicyController":
        import torch

        blob = torch.load(path, map_location=self.device, weights_only=False)
        self.hidden_size = blob["hidden_size"]
        self.num_layers = blob["num_layers"]
        self._model = self._build(blob["window"]).to(self.device)
        self._model.load_state_dict(blob["state_dict"])
        self._norm = tuple(blob["norm"])
        return self

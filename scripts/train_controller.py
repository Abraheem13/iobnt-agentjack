"""Train the learned controller and save it for every later experiment.

    python scripts/train_controller.py [--episodes 60] [--epochs 25]

Writes results/models/policy_gru_seed<N>.pt plus a JSON training record.
Seeds are explicit: the paper's headline numbers must be regenerable from this
command alone.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from agentjack.agent.policy_controller import (  # noqa: E402
    ACTIONS,
    PolicyController,
    collect_demonstrations,
)
from agentjack.data.loaders.cgm import load_cgm_or_synthetic  # noqa: E402
from agentjack.twin.digital_twin import BioCyberTwin, TwinConfig  # noqa: E402
from agentjack.utils.seeding import device_report, set_all_seeds  # noqa: E402


def make_factory(cgm, patients):
    def factory(seed: int) -> BioCyberTwin:
        pid = patients[seed % len(patients)]
        meals = cgm[cgm.patient_id == pid].meal_carbs_g.to_numpy()
        return BioCyberTwin(TwinConfig(), meals=meals, seed=seed)

    return factory


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="results/models")
    args = ap.parse_args()

    set_all_seeds(args.seed)
    cgm, is_real = load_cgm_or_synthetic(seed=args.seed)
    patients = list(cgm.patient_id.unique())

    # Patients are split so the controller is evaluated on people it never saw.
    n_train = max(1, int(0.7 * len(patients)))
    train_patients, test_patients = patients[:n_train], patients[n_train:]

    print("training the learned controller\n")
    print(f"  CGM source     {'REAL Shanghai' if is_real else 'SYNTHETIC fallback'}")
    print(f"  patients       {len(train_patients)} train / {len(test_patients)} held out")
    print(f"  device         {device_report()['device']}\n")

    X, y = collect_demonstrations(make_factory(cgm, train_patients), args.episodes, seed=args.seed)
    balance = np.bincount(y, minlength=len(ACTIONS)) / len(y)
    print(f"  demonstrations {X.shape[0]} steps")
    print("  action balance " + "  ".join(f"{a}={b:.3f}" for a, b in zip(ACTIONS, balance)))
    if balance[-1] == 0:
        print("  WARNING: the largest dose never appears - the controller cannot learn to")
        print("           emit it, and an over-dosing attack would be unmeasurable.")
    print()

    pc = PolicyController(seed=args.seed)
    hist = pc.fit(X, y, epochs=args.epochs, verbose=True)

    out = Path(args.out)
    model_path = out / f"policy_gru_seed{args.seed}.pt"
    pc.save(model_path)

    record = {
        "seed": args.seed,
        "episodes": args.episodes,
        "epochs": args.epochs,
        "n_steps": int(X.shape[0]),
        "action_balance": {a: float(b) for a, b in zip(ACTIONS, balance)},
        "final_val_acc": hist["val_acc"][-1],
        "final_val_recall": {a: r for a, r in zip(ACTIONS, hist["val_recall"][-1])},
        "train_patients": train_patients,
        "test_patients": test_patients,
        "cgm_is_real": is_real,
        "device": device_report(),
    }
    (out / f"policy_gru_seed{args.seed}.json").write_text(json.dumps(record, indent=2, default=str))

    print(f"\n  saved {model_path}")
    print(f"  saved {out / f'policy_gru_seed{args.seed}.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

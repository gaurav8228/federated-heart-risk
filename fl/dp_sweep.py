r"""
dp_sweep.py  --  Milestone M6
Runs FedAvg with NO differential privacy and with several noise levels, and
prints the privacy-vs-utility trade-off as a table:

        noise multiplier   final epsilon   accuracy   F1   AUC

    venv\Scripts\python.exe fl\dp_sweep.py

It uses the plain federated loop (from run_fl.py) instead of Flower, so it
starts no server and stays fast. Each row is one full 20-round FedAvg run.
"""

import os
import sys

# Make the project root importable
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from fl.data import load_fl_data
from fl.dp import epsilon_for_rounds
from fl.run_fl import CLIP_DEFAULT, DELTA_DEFAULT, ROUNDS_DEFAULT, SEED
from fl.run_fl import evaluate_model, run_plain_federated

NOISE_LEVELS = [0.0, 0.5, 1.0, 2.0, 4.0]   # 0.0 = no DP
CLIP = CLIP_DEFAULT
DELTA = DELTA_DEFAULT
ROUNDS = ROUNDS_DEFAULT


def main():
    print("Loading data...")
    _, clients, X_test, y_test = load_fl_data()
    n_features = X_test.shape[1]

    print(f"\nFedAvg, {ROUNDS} rounds each, clip={CLIP}, delta={DELTA:g}")
    print(f"{'Noise':>7}{'Epsilon':>10}{'Accuracy':>10}{'F1':>9}{'AUC':>9}")
    print("-" * 45)

    for noise in NOISE_LEVELS:
        model = run_plain_federated(
            "fedavg", 0.0, ROUNDS, clients, n_features,
            dp_clip=CLIP if noise > 0 else 0.0,
            dp_noise=noise,
            delta=DELTA,
            seed=SEED,
        )
        acc, f1, auc = evaluate_model(model, X_test, y_test)
        eps = "inf" if noise == 0 else f"{epsilon_for_rounds(ROUNDS, noise, DELTA):.2f}"
        print(f"{noise:>7}{eps:>10}{acc:>10.4f}{f1:>9.4f}{auc:>9.4f}")

    print("\nHigher noise = more private (smaller epsilon) but less accurate.")
    print("Noise 0 (no DP) keeps full accuracy but has infinite epsilon.")


if __name__ == "__main__":
    main()

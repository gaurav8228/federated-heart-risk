r"""
run_fl.py  --  Milestone M5 + M6  (the file you run)
Runs federated learning across 3 simulated hospital clients using Flower.

    venv\Scripts\python.exe fl\run_fl.py --strategy fedavg  --rounds 20 --mu 0.0
    venv\Scripts\python.exe fl\run_fl.py --strategy fedprox --rounds 20 --mu 0.1
    venv\Scripts\python.exe fl\run_fl.py --strategy fedavg  --rounds 20 --dp --clip 1.0 --noise 1.0
    venv\Scripts\python.exe fl\run_fl.py --strategy fedavg  --rounds 20 --secagg

What it does:
  1. Loads the data (3 hospitals + a shared held-out test set)  -> fl/data.py
  2. Builds a numpy logistic-regression model                    -> fl/model.py
  3. Starts a Flower SERVER and 3 hospital CLIENTS (one thread each).
     After every round the server averages the 3 hospitals' weights
     (FedAvg, or FedProx which also pulls each hospital toward the global
     model).  Only weights are shared - never patient data.
  4. DIFFERENTIAL PRIVACY (M6, when --dp is given): each hospital clips its
     weight update to --clip and adds Gaussian noise before sending it, and
     an accountant tracks the privacy budget epsilon after every round.
  5. After every round it scores the global model on the held-out test set
     and prints round / accuracy / F1 / AUC / epsilon / comm cost, then writes
     the row to the "fl_rounds" table in PostgreSQL.
  6. Finally prints a comparison:  centralized LR  vs  FedAvg  vs  FedProx.

NOTE: Ray is not installed (and we do not want PyTorch/TensorFlow), so we use
Flower's real server + 3 client threads rather than Flower's "simulation"
mode, which needs Ray.  This runs fine on Windows.
"""

import argparse
import logging
import os
import sys
import threading
import time
import warnings

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# Make the project root importable so we can use fl/, ml/ and backend/
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import flwr as fl

from fl.data import load_fl_data, union_of_clients
from fl.dp import clip_and_noise, epsilon_for_rounds
from fl.model import LogisticModel
from fl.secagg import mask_update
from backend.database import Base, SessionLocal, engine
from backend.models import FLRound

# --- training settings (kept simple on purpose) -----------------------------
EPOCHS = 3          # local passes per hospital per round
LR = 0.5            # learning rate
ADDRESS = "127.0.0.1:8080"
SEED = 42
# --- M6 differential-privacy defaults (shared with fl/dp_sweep.py) ----------
ROUNDS_DEFAULT = 20
CLIP_DEFAULT = 1.0
NOISE_DEFAULT = 1.0
DELTA_DEFAULT = 1e-5


# =========================================================================== #
#  Database logging: one row per round in the "fl_rounds" table
# =========================================================================== #
def log_round(round_number, strategy, accuracy, comm_mb, epsilon=None, status="done"):
    """Write one federated-learning round to PostgreSQL (never crashes training)."""
    try:
        db = SessionLocal()
        try:
            db.add(FLRound(
                round_number=round_number,
                strategy=strategy,
                global_accuracy=round(float(accuracy), 4),
                epsilon=epsilon,                       # None for non-DP runs (M6)
                communication_cost_mb=round(float(comm_mb), 4),
                status=status,
            ))
            db.commit()
        finally:
            db.close()
    except Exception as e:                           # keep training even if DB hiccups
        print("  (warning: could not save round to database:", e, ")")


# =========================================================================== #
#  Scoring the global model on the shared held-out test set
# =========================================================================== #
def evaluate_model(model, X_test, y_test):
    """Return (accuracy, f1, auc) for a model on the test patients."""
    pred = model.predict(X_test)
    prob = model.proba(X_test)
    return (
        float(accuracy_score(y_test, pred)),
        float(f1_score(y_test, pred)),
        float(roc_auc_score(y_test, prob)),
    )


def comm_cost_mb(model, n_clients):
    """
    How much data moved this round, in megabytes:
    every client sends its weights UP and receives the global weights DOWN.
    """
    up_down = model.param_bytes() * 2              # up + down
    return (up_down * n_clients) / (1024 * 1024)


# =========================================================================== #
#  A Flower CLIENT = one hospital
# =========================================================================== #
class HospitalClient(fl.client.NumPyClient):
    """One hospital: holds its OWN patients and only shares weights."""

    def __init__(self, model, X, y, dp_clip=0.0, dp_noise=0.0, seed=SEED,
                 secagg=False, client_id=0, client_ids=None, secagg_seed=SEED):
        self.model = model
        self.X = X
        self.y = y
        self.dp_clip = dp_clip          # max L2 norm of the update (0 = no DP)
        self.dp_noise = dp_noise        # noise multiplier sigma (0 = no DP)
        self.rng = np.random.default_rng(seed)   # independent noise per hospital
        self.secagg = secagg            # M10: pairwise-mask the update before sending
        self.client_id = client_id
        self.client_ids = client_ids or [client_id]
        self.secagg_seed = secagg_seed  # base seed; + fit_count = per-round seed
        self.fit_count = 0              # which federated round this client is on

    def get_parameters(self, config):
        return self.model.get_params()

    def set_parameters(self, params):
        self.model.set_params(params)

    def fit(self, parameters, config):
        """Train locally for a few epochs, then send the (maybe noised) weights."""
        self.set_parameters(parameters)
        mu = float(config.get("proximal_mu", 0.0))          # FedProx tells us mu here
        w_global = np.asarray(parameters[0], dtype=np.float64)
        b_global = float(np.asarray(parameters[1]).ravel()[0])
        self.model.train_local(self.X, self.y, epochs=EPOCHS, lr=LR,
                               mu=mu, w_global=w_global)

        # --- M6 differential privacy: clip the update, add Gaussian noise ----
        if self.dp_noise > 0 and self.dp_clip > 0:
            # the update = what this hospital wants to change (local - global)
            upd = np.concatenate([self.model.w - w_global, [self.model.b - b_global]])
            upd = clip_and_noise(upd, self.dp_clip, self.dp_noise, self.rng)
            # send back "global + noised update" so the server just averages these
            self.model.w = w_global + upd[:-1]
            self.model.b = b_global + float(upd[-1])

        # --- M10 SecAgg (after DP): mask the update; masks cancel in the sum --
        if self.secagg:
            round_seed = self.secagg_seed + self.fit_count   # same for all clients this round
            upd = np.concatenate([self.model.w - w_global, [self.model.b - b_global]])
            upd = mask_update(self.client_id, upd, self.client_ids, round_seed)
            self.model.w = w_global + upd[:-1]
            self.model.b = b_global + float(upd[-1])
            self.fit_count += 1

        return self.model.get_params(), len(self.y), {}      # weights, n_samples, metrics

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        p = self.model.proba(self.X)
        loss = float(-np.mean(self.y * np.log(p + 1e-9) + (1 - self.y) * np.log(1 - p + 1e-9)))
        return loss, len(self.y), {}


def _run_client(address, client, tries=40, wait=0.5):
    """Start one hospital client; retry until the server is listening."""
    for attempt in range(tries):
        try:
            fl.client.start_client(server_address=address, client=client.to_client())
            return
        except Exception as e:
            if attempt == tries - 1:
                print("  client gave up connecting:", e)
                return
            time.sleep(wait)


# =========================================================================== #
#  One full federated run (Flower server + 3 threaded clients)
# =========================================================================== #
def run_flower(strategy_name, mu, rounds, clients, X_test, y_test, n_features,
               log_db=True, address=ADDRESS, dp=False, clip=1.0, noise=1.0,
               delta=1e-5, verbose=True, secagg=False):
    """Run the FL training and return the final global model."""
    global_model = LogisticModel(n_features)
    comm_mb = comm_cost_mb(global_model, len(clients))

    def eps_after(r):
        """Epsilon after r rounds (None when DP is off; 0.0 before anything is shared)."""
        if not dp:
            return None
        return 0.0 if r == 0 else epsilon_for_rounds(r, noise, delta)

    def show(r, acc, f1, auc):
        e = eps_after(r)
        eps_str = f"{e:.2f}" if e is not None else "-"
        print(f"  round {r:>3} | acc {acc:.4f} | f1 {f1:.4f} | auc {auc:.4f} "
              f"| eps {eps_str:>6} | comm {comm_mb:.4f} MB")

    # round 0 = the untrained starting model (so it appears in the log too)
    acc, f1, auc = evaluate_model(global_model, X_test, y_test)
    if verbose:
        show(0, acc, f1, auc)
    if log_db:
        log_round(0, strategy_name, acc, comm_mb, epsilon=eps_after(0))

    # Flower calls this after every round with the NEW global weights
    def evaluate_fn(server_round, parameters, config):
        global_model.set_params(parameters)
        acc, f1, auc = evaluate_model(global_model, X_test, y_test)
        if server_round >= 1:                         # round 0 already handled above
            if verbose:
                show(server_round, acc, f1, auc)
            if log_db:
                log_round(server_round, strategy_name, acc, comm_mb,
                          epsilon=eps_after(server_round))
        loss = float(-np.mean(y_test * np.log(global_model.proba(X_test) + 1e-9)
                              + (1 - y_test) * np.log(1 - global_model.proba(X_test) + 1e-9)))
        return loss, {"accuracy": acc}

    common = dict(
        fraction_fit=1.0,                              # every hospital trains
        fraction_evaluate=0.0,                         # server scores, not clients
        min_fit_clients=len(clients),
        min_available_clients=len(clients),
        evaluate_fn=evaluate_fn,
        initial_parameters=fl.common.ndarrays_to_parameters(global_model.get_params()),
    )
    if strategy_name == "fedprox":
        strategy = fl.server.strategy.FedProx(proximal_mu=mu, **common)
    else:
        strategy = fl.server.strategy.FedAvg(**common)

    # start the 3 hospital clients, each in its own thread
    threads = []
    client_ids = list(range(len(clients)))
    for i, c in enumerate(clients):
        local = LogisticModel(n_features)
        client = HospitalClient(
            local, c["X"], c["y"],
            dp_clip=clip if dp else 0.0,               # M6: clip the update
            dp_noise=noise if dp else 0.0,             # M6: add Gaussian noise
            seed=SEED + i,                             # different noise per hospital
            secagg=secagg,                             # M10: pairwise-mask the update
            client_id=i,
            client_ids=client_ids,
            secagg_seed=SEED,
        )
        t = threading.Thread(target=_run_client, args=(address, client), daemon=True)
        t.start()
        threads.append(t)

    # start the Flower server in THIS thread - it blocks until training ends
    fl.server.start_server(
        server_address=address,
        config=fl.server.ServerConfig(num_rounds=rounds),
        strategy=strategy,
    )
    for t in threads:
        t.join(timeout=5)
    return global_model                               # holds the final weights


# =========================================================================== #
#  Helpers used only for the final comparison table
# =========================================================================== #
def train_centralized(clients, n_features, total_epochs):
    """Plain logistic regression allowed to see ALL hospitals' data at once."""
    X_all, y_all = union_of_clients(clients)
    m = LogisticModel(n_features)
    m.train_local(X_all, y_all, epochs=total_epochs, lr=LR, mu=0.0)
    return m


def run_plain_federated(strategy, mu, rounds, clients, n_features,
                        dp_clip=0.0, dp_noise=0.0, delta=1e-5, seed=SEED):
    """
    A tiny plain-Python FedAvg/FedProx loop (same maths as Flower, including
    the optional M6 clip+noise) - used to fill in the OTHER strategy in the
    comparison table and by fl/dp_sweep.py, without starting a Flower server.
    """
    g = LogisticModel(n_features)
    rng = np.random.default_rng(seed)
    for _ in range(rounds):
        w_global, b_global = g.w.copy(), g.b
        updates, total_n = [], 0
        for c in clients:
            local = LogisticModel(n_features)
            local.set_params([w_global, np.array([b_global])])
            m = mu if strategy == "fedprox" else 0.0
            local.train_local(c["X"], c["y"], epochs=EPOCHS, lr=LR,
                              mu=m, w_global=w_global)
            upd = np.concatenate([local.w - w_global, [local.b - b_global]])
            if dp_noise > 0 and dp_clip > 0:            # M6 differential privacy
                upd = clip_and_noise(upd, dp_clip, dp_noise, rng)
            updates.append((upd, len(c["y"])))
            total_n += len(c["y"])
        avg_upd = np.zeros(n_features + 1)
        for upd, n in updates:
            avg_upd += upd * (n / total_n)
        g.w = w_global + avg_upd[:-1]
        g.b = b_global + float(avg_upd[-1])
    return g


# =========================================================================== #
#  Main
# =========================================================================== #
def main():
    parser = argparse.ArgumentParser(description="HeartRisk federated learning")
    parser.add_argument("--strategy", choices=["fedavg", "fedprox"], default="fedavg")
    parser.add_argument("--rounds", type=int, default=ROUNDS_DEFAULT)
    parser.add_argument("--mu", type=float, default=0.0,
                        help="FedProx proximal strength (only used with --strategy fedprox)")
    # --- M6 differential-privacy flags ---
    parser.add_argument("--dp", action="store_true",
                        help="turn differential privacy ON (clip + Gaussian noise)")
    parser.add_argument("--clip", type=float, default=CLIP_DEFAULT,
                        help="max L2 norm of each hospital's weight update")
    parser.add_argument("--noise", type=float, default=NOISE_DEFAULT,
                        help="Gaussian noise multiplier (sigma); bigger = more private")
    parser.add_argument("--delta", type=float, default=DELTA_DEFAULT,
                        help="target failure probability for the privacy budget")
    # --- M10 secure aggregation (pairwise-masking simulation) ---
    parser.add_argument("--secagg", action="store_true",
                        help="mask each update with pairwise secrets (SecAgg simulation)")
    args = parser.parse_args()

    # hide Flower's very chatty INFO/DEPRECATION logs so the output stays readable
    logging.getLogger("flwr").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    print("Loading data and splitting for 3 hospitals...")
    pre, clients, X_test, y_test = load_fl_data()
    n_features = X_test.shape[1]
    Base.metadata.create_all(bind=engine)              # make sure fl_rounds exists

    print(f"\n=== Federated run: {args.strategy.upper()}"
          + (f" (mu={args.mu})" if args.strategy == "fedprox" else "")
          + (f" + DP (clip={args.clip}, noise={args.noise}, delta={args.delta})"
             if args.dp else "")
          + (" + SecAgg" if args.secagg else "")
          + f" for {args.rounds} rounds ===")
    final_model = run_flower(args.strategy, args.mu, args.rounds,
                             clients, X_test, y_test, n_features, log_db=True,
                             dp=args.dp, clip=args.clip, noise=args.noise,
                             delta=args.delta, secagg=args.secagg)
    main_acc, main_f1, main_auc = evaluate_model(final_model, X_test, y_test)

    # --- final comparison: centralized vs FedAvg vs FedProx ------------------
    print("\n=== Final comparison (held-out test set) ===")
    cent = train_centralized(clients, n_features, total_epochs=args.rounds * EPOCHS)
    c_acc, c_f1, c_auc = evaluate_model(cent, X_test, y_test)

    other = "fedprox" if args.strategy == "fedavg" else "fedavg"
    other_mu = 0.1 if other == "fedprox" else 0.0
    other_model = run_plain_federated(other, other_mu, args.rounds, clients, n_features)
    o_acc, o_f1, o_auc = evaluate_model(other_model, X_test, y_test)

    dp_note = " (DP)" if args.dp else ""
    rows = [
        ("Centralized LR (all data)", c_acc, c_f1, c_auc),
        (f"FedAvg{dp_note}", main_acc, main_f1, main_auc) if args.strategy == "fedavg"
        else (f"FedProx (mu={args.mu}){dp_note}", main_acc, main_f1, main_auc),
        (f"FedAvg" if other == "fedavg" else f"FedProx (mu={other_mu})",
         o_acc, o_f1, o_auc),
    ]
    print(f"{'Model':<28}{'Accuracy':>10}{'F1':>9}{'AUC':>9}")
    for name, a, f, u in rows:
        print(f"{name:<28}{a:>10.4f}{f:>9.4f}{u:>9.4f}")

    if args.dp:
        final_eps = epsilon_for_rounds(args.rounds, args.noise, args.delta)
        print(f"\nPrivacy budget after {args.rounds} rounds: "
              f"epsilon = {final_eps:.2f} (delta = {args.delta:g})")

    print(f"\nDone. Every round was saved to the 'fl_rounds' table "
          f"({args.rounds + 1} rows for {args.strategy}).")


if __name__ == "__main__":
    main()

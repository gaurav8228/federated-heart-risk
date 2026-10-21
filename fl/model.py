r"""
model.py  --  Milestone M5
A tiny logistic-regression model written in numpy (no scikit-learn needed here).

Why logistic regression and not XGBoost?  Federated averaging (FedAvg) can
only average NUMBERS (weights).  It cannot average a forest of decision
trees, so tree models like XGBoost cannot be federated this way.  A plain
logistic regression has a small weight vector that averages perfectly.

The model is just:
    probability = sigmoid( X . weights + bias )

"weights" is a numpy array (one number per feature) and "bias" is one number.
"""

import numpy as np


def sigmoid(z):
    """Squash any number into the 0..1 range (the probability)."""
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))  # clip avoids overflow


class LogisticModel:
    """Logistic regression trained with plain gradient descent."""

    def __init__(self, n_features, seed=42):
        rng = np.random.default_rng(seed)
        self.w = rng.normal(0, 0.01, n_features)   # start with tiny random weights
        self.b = 0.0                               # start bias at 0

    # --- prediction ---------------------------------------------------------
    def proba(self, X):
        """Probability of heart disease for every row of X."""
        return sigmoid(X @ self.w + self.b)

    def predict(self, X):
        """0 (healthy) or 1 (disease) using a 0.5 threshold."""
        return (self.proba(X) >= 0.5).astype(int)

    # --- getting / setting the weights (this is what gets "sent" in FL) -----
    def get_params(self):
        """Return the weights as a list of numpy arrays (the thing we share)."""
        return [self.w.copy(), np.array([self.b], dtype=np.float64)]

    def set_params(self, params):
        """Load weights that were shared by the server."""
        self.w = np.asarray(params[0], dtype=np.float64)
        self.b = float(np.asarray(params[1]).ravel()[0])

    def param_bytes(self):
        """How many bytes the weights take (used to measure communication)."""
        return int(self.w.nbytes + np.array([self.b], dtype=np.float64).nbytes)

    # --- local training -----------------------------------------------------
    def train_local(self, X, y, epochs=3, lr=0.5, mu=0.0, w_global=None):
        """
        Train on this hospital's own patients for a few epochs.

        epochs = how many passes over the local data
        lr     = learning rate (how big each step is)
        mu     = FedProx "proximal" strength (0.0 = plain FedAvg)
        w_global = the shared global weights (the anchor for FedProx)

        FedProx adds the term  mu/2 * ||w - w_global||^2  to the loss so the
        local model does not drift too far from the shared global model.
        Its gradient is simply  mu * (w - w_global).
        """
        w, b = self.w.copy(), self.b
        n = len(y)
        # only pull toward the global weights when FedProx is switched on
        anchor = w_global if (mu > 0.0 and w_global is not None) else None

        for _ in range(epochs):
            p = sigmoid(X @ w + b)
            err = p - y                                   # how wrong we are
            grad_w = X.T @ err / n                        # average slope per feature
            grad_b = float(err.mean())
            if anchor is not None:
                grad_w = grad_w + mu * (w - anchor)       # FedProx pull-back
            w = w - lr * grad_w
            b = b - lr * grad_b

        self.w, self.b = w, b

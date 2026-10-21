r"""
secagg.py  --  Secure aggregation (SecAgg) as a pairwise-masking SIMULATION.

This is NOT a cryptographic secure-aggregation protocol. It is the standard
"pairwise mask" trick that shows *why* SecAgg works:

  * Every pair of clients (i, j) shares a secret random mask, same shape as
    their updates. Client i ADDS it, client j SUBTRACTS it.
  * Because each pair's mask appears once with + and once with -, every mask
    cancels when the server SUMS the updates. So the server can add up the
    (individually meaningless) masked updates and recover the true sum -
    without any single update being readable.

There is no key exchange and no dropout handling here (see README Limitations).
Run the self-test with:   venv\Scripts\python.exe fl\secagg.py
"""

import numpy as np


def _pair_mask(i, j, shape, round_seed):
    """The shared mask for the pair (i, j). Deterministic: the same
    (round_seed, i, j) always yields the same mask, so both clients derive an
    identical mask without sending it to anyone."""
    if i > j:
        i, j = j, i                                  # always call with i < j
    rng = np.random.default_rng([int(round_seed), int(i), int(j)])
    return rng.normal(0.0, 1.0, shape)


def mask_update(client_id, update, client_ids, round_seed):
    """Return this client's MASKED update.

    For every other client j:
      * if client_id < j  -> ADD    the (client_id, j) mask
      * if client_id > j  -> SUBTRACT the (j, client_id) mask
    All masks cancel when the server adds up every client's masked update.
    """
    update = np.asarray(update, dtype=np.float64)
    masked = update.copy()
    for other in client_ids:
        if other == client_id:
            continue
        mask = _pair_mask(client_id, other, update.shape, round_seed)
        masked = masked + mask if client_id < other else masked - mask
    return masked


def masked_updates(updates, client_ids, round_seed):
    """Helper: mask every client's update at once (list of updates -> list)."""
    return [mask_update(cid, upd, client_ids, round_seed)
            for cid, upd in zip(client_ids, updates)]


if __name__ == "__main__":
    # ---- self-test: masks cancel in the SUM, but clearly hide each update ----
    rng = np.random.default_rng(7)
    client_ids = [0, 1, 2]
    shape = (29,)                                    # same size as our model's params
    updates = [rng.normal(0.0, 1.0, shape) for _ in client_ids]
    round_seed = 12345

    plain = np.sum(updates, axis=0)
    masked = masked_updates(updates, client_ids, round_seed)
    recovered = np.sum(masked, axis=0)

    ok_sum = bool(np.allclose(plain, recovered, atol=1e-9))
    err = float(np.max(np.abs(plain - recovered)))
    shift = float(np.linalg.norm(masked[0] - updates[0]))
    ok_hidden = shift > 1.0                          # one masked update is clearly altered

    print("masked sum == plain sum :", "PASS" if ok_sum else "FAIL",
          f"(max abs err {err:.2e})")
    print("single update is masked :", "PASS" if ok_hidden else "FAIL",
          f"(L2 shift {shift:.2f})")
    print("\n" + ("SELF-TEST PASSED" if (ok_sum and ok_hidden) else "SELF-TEST FAILED"))

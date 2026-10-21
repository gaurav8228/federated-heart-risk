r"""
dp.py  --  Milestone M6
Differential privacy for the federated learning, in plain numpy.

Three small pieces live here:
  1. clip_and_noise()  - the client-side DP step (clip the update, add noise)
  2. epsilon_for_rounds() - a tiny privacy "accountant" that turns
                            (rounds, noise multiplier, delta) into an epsilon
  3. comments explaining exactly what is assumed and where it is wrong

No Opacus, no TensorFlow Privacy - just numpy, so you can read every line.
"""

import numpy as np


# --------------------------------------------------------------------------- #
# 1. The per-client DP step
# --------------------------------------------------------------------------- #
def clip_and_noise(update, clip, noise_multiplier, rng):
    """
    Take one hospital's weight update (a 1-D numpy vector = local - global),
      * clip it so its L2 norm never exceeds `clip`, then
      * add Gaussian noise with standard deviation = noise_multiplier * clip.

    Clipping bounds how much a single hospital can influence the result
    (this is the "sensitivity"), and the noise hides that influence.
    """
    norm = np.linalg.norm(update)                 # length of the vector
    if norm > clip and norm > 0:
        update = update * (clip / norm)           # shrink down to size `clip`
    if noise_multiplier > 0:
        noise = rng.normal(0.0, noise_multiplier * clip, update.shape)
        update = update + noise
    return update


# --------------------------------------------------------------------------- #
# 2. The privacy accountant  (how private are we after T rounds?)
# --------------------------------------------------------------------------- #
def epsilon_for_rounds(rounds, noise_multiplier, delta=1e-5):
    """
    Return the privacy budget epsilon after `rounds` federated rounds,
    for a target failure probability `delta` (usually 1e-5).

    THE FORMULA (a standard, simple RDP accountant):

      a) Each round is treated as ONE application of the Gaussian mechanism
         with "noise multiplier" sigma = noise_multiplier.  (Because both the
         clip bound C and the noise std are proportional to C, C cancels out
         and only the ratio sigma matters for privacy.)

      b) The Renyi-Differential-Privacy (RDP) of a Gaussian mechanism at
         order alpha is:        eps_rdp(alpha) = rounds * alpha / (2 * sigma^2)
         (composition over `rounds` rounds simply adds the RDP terms.)

      c) Convert RDP to ordinary (epsilon, delta)-DP with the standard bound:
              eps = min over alpha>1 of  [ eps_rdp(alpha) + ln(1/delta) / (alpha - 1) ]

    LIMITS (read this - the accountant is an honest approximation, not a
    certified one like Google's or diffprivlib's):
      * It assumes every client takes part every round and ignores sampling,
        shuffling and secure aggregation, which would make privacy BETTER.
      * It treats the whole round as a single sigma-mechanism and does not
        model the fact that averaging K hospitals' noise cancels some of it,
        which is the CONSERVATIVE (safe) direction - it never claims more
        privacy than the simple model allows.
      * Epsilon depends only on (rounds, sigma, delta). The `clip` value
        controls accuracy vs. noise size, not epsilon directly.
    """
    if noise_multiplier <= 0:
        return float("inf")                       # no noise = no privacy

    def rdp(alpha):
        return rounds * alpha / (2.0 * noise_multiplier ** 2)

    # try many orders alpha and keep the smallest resulting epsilon
    alphas = np.arange(2.0, 512.0)
    eps = rdp(alphas) + np.log(1.0 / delta) / (alphas - 1.0)
    return float(np.min(eps))

import numpy as np

USED_FEATURES = [9, 22, 42, 44, 45, 52]
PARAMS = [0.729966921814, 0.5263552684424777, 0.5315513738418385, 0.7977413806452993, 0.6344009585513212]
BOUNDS = [[0.0, 4.0], [0.0, 3.0], [0.0, 1.0], [0.0, 1.5], [0.0, 1.0]]


def score_function(variable_features, params):
    """Score candidate branching variables for the hard MIS-BA subset.

    Design rationale
    ----------------
    * The hard instances have a loose root LP and near-uniform cliques, so
      plain hub/degree features do not separate good from bad candidates.
      We therefore gate all structural signals by LP fractionality (so that
      integral or almost-integral variables cannot hijack the ranking).
    * Pseudocost magnitude is the most direct "how much does branching on
      this variable move the bound" estimator, but it is only trustworthy
      when the pseudocost has actually been exercised.  We blend it with a
      reliability proxy built from the per-direction cutoff counts.
    * A dynamic/static constraint-degree ratio is used as a cheap
      "not-yet-collapsed neighbourhood" signal: variables still sitting in
      a fresh part of the residual clique cover tend to be more impactful
      branches on these hard instances.
    * A small static hub term (participating constraint count) provides a
      safe fall-back when the dynamic signals are weak, matching the
      general-purpose policy's baseline behaviour without dominating.

    All inputs are in [0, 1], and the expression uses only vectorised
    numpy operations, so it is O(n) and extremely cheap.
    """
    # 1) LP fractionality: peaks at 0.5, zero at 0 or 1.
    frac = variable_features[:, 9]
    frac_score = 4.0 * frac * (1.0 - frac)

    # 2) Reliability-discounted pseudocost sum.
    pc_sum = variable_features[:, 42]
    reliability = np.clip(
        0.5 * (variable_features[:, 44] + variable_features[:, 45]),
        0.0, 1.0,
    )
    # When reliability is low, shrink toward `params[2]` * pc_sum;
    # when reliability is high, use the pseudocost at full strength.
    pc_term = pc_sum * (params[2] + (1.0 - params[2]) * reliability)

    # 3) Dynamic constraint-degree drift signal.
    drift = variable_features[:, 52]

    # 4) Static hub fall-back signal.
    degree = variable_features[:, 22]

    # Core: fractionality-gated structural score, plus a mild fallback.
    return (
        frac_score * (params[0] + params[1] * pc_term + params[4] * drift)
        + params[3] * degree
    )
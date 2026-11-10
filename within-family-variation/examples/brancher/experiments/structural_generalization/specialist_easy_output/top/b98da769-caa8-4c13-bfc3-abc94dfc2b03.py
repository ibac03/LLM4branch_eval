USED_FEATURES = [9, 22, 42, 43, 52, 86]
PARAMS = [0.0, 1.9182063959427498, 0.3934871629067387, 1.8104862784363096, 2.5935982039626353, -0.17935037909487161]
BOUNDS = [[0.0, 4.0], [-2.0, 2.0], [-2.0, 3.0], [-2.0, 3.0], [-2.0, 3.0], [-2.0, 3.0]]

def score_function(variable_features, params):
    # Core fractionality weight: peaks at LP value 0.5, vanishes at 0/1
    f  = variable_features[:, 9]
    fw = 4.0 * f * (1.0 - f)

    # Baseline hub signal (kept small so the LP-dynamics signals can dominate)
    n_constr = variable_features[:, 22]

    # Pseudocost reliability / strength (strong-branching surrogates).
    # Sum = magnitude of branching effect; Product captures balanced up/down damage.
    ps_sum  = variable_features[:, 42]
    ps_prod = variable_features[:, 43]

    # Dynamic/static constraint-degree ratio (mean): where the LP is actively moving.
    drift = variable_features[:, 52]

    # Sum of dual costs of active (tight) constraints containing this variable:
    # direct LP-sensitivity signal for hard, loose-relaxation instances.
    active_dual = variable_features[:, 86]

    score = (
        params[0] * fw +
        params[1] * n_constr +
        params[2] * ps_sum +
        params[3] * ps_prod +
        params[4] * drift +
        params[5] * active_dual
    )
    return score
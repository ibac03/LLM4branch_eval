USED_FEATURES = [9, 42, 43, 44, 45, 52, 54, 56, 67, 86]
PARAMS = [1.2392044407862963, 1.319773852838377, 0.42180385507385193, 1.0468692901193744, 0.0, 0.724335775744762]
BOUNDS = [[0, 2], [0, 2], [0, 2], [0, 2], [0, 2], [0, 2]]

def score_function(variable_features, params):
    x = variable_features

    # Fractionality: prefer variables near the middle of their domain.
    frac = x[:, 9]
    frac_score = 4.0 * frac * (1.0 - frac)

    # Pseudocost reliability: use both pseudocost sum and product,
    # moderated by how often branches on this variable have been cut off.
    pc_sum = x[:, 42]
    pc_prod = x[:, 43]
    cutoff = x[:, 44] + x[:, 45]
    pseudo = (pc_sum + 0.5 * pc_prod) * (1.0 + 0.5 * cutoff)

    # Dynamic constraint-degree drift: captures changes in the active
    # constraint structure relative to the static graph.
    dyn_drift = x[:, 52] + 0.5 * x[:, 54]

    # Active constraint signal: count plus dual-cost weighted activity.
    active = x[:, 67] + x[:, 86]

    # Coefficient-to-RHS tightness signal.
    coef_rhs = x[:, 56]

    score = (
        params[0] * frac_score
        + params[1] * pseudo
        + params[2] * dyn_drift
        + params[3] * active
        + params[4] * coef_rhs
        + params[5] * frac_score * active
    )

    return score
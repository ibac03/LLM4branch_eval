USED_FEATURES = [9, 43, 52, 53, 54, 59, 86]
PARAMS = [0.5817, 0.7231, 0.6342, 0.5129, 0.4498, 0.3276, 0.2847, 0.1953, 0.7023, 0.1584]
BOUNDS = [[0,1], [0,1], [0,1], [0,1], [0,1], [0,1], [0,1], [0,1], [0,1], [0,1]]

def score_function(variable_features, params):
    # Core clustered graph features with enhanced dual synergy
    fractional = variable_features[:, 9]        # Fractional branching urgency
    pseudocost = variable_features[:, 43]       # Pseudocost reliability
    dyn_mean_ratio = variable_features[:, 52]   # Dynamic/static constraint mean ratio
    dyn_std_ratio = variable_features[:, 53]    # Dynamic constraint volatility ratio
    dyn_max_ratio = variable_features[:, 54]    # Critical constraint intensity ratio
    coef_rhs = variable_features[:, 59]         # Constraint tightness ratio
    dual_impact = variable_features[:, 86]      # Dual-weighted cluster pressure

    # Strategic nonlinear interactions for clustered dynamics
    cluster_reliability = pseudocost * dyn_mean_ratio  # Stable pseudocosts in clustered constraints
    dual_volatility = dyn_std_ratio * dual_impact      # Dual pressure on volatile constraints
    dual_tightness = coef_rhs * dual_impact            # Tight constraints under dual influence
    peak_dual_synergy = dyn_max_ratio * dual_impact    # Dual focus on critical constraints

    return (
        params[0] * fractional +
        params[1] * pseudocost +
        params[2] * dyn_mean_ratio +
        params[3] * dyn_std_ratio +
        params[4] * dyn_max_ratio +
        params[5] * coef_rhs +
        params[6] * dual_impact +
        params[7] * cluster_reliability +
        params[8] * dual_volatility +
        params[9] * dual_tightness
    )
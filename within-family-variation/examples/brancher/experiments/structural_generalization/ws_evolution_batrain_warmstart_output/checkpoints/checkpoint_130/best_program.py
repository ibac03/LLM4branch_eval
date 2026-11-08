USED_FEATURES = [9, 43, 52, 53, 54, 59, 86]
PARAMS = [0.5817, 0.7023, 0.6342, 0.7231, 0.5129, 0.4498, 0.3276, 0.2847, 0.1953, 0.1584]
BOUNDS = [[0,1], [0,1], [0,1], [0,1], [0,1], [0,1], [0,1], [0,1], [0,1], [0,1]]

def score_function(variable_features, params):
    # Core cluster-aware features
    fractional = variable_features[:, 9]          # Fractional branching urgency
    pseudocost = variable_features[:, 43]         # Pseudocost reliability product
    dyn_mean_ratio = variable_features[:, 52]     # Dynamic/static mean constraint ratio
    dyn_std_ratio = variable_features[:, 53]      # Dynamic/static std ratio (cluster variability)
    dyn_max_ratio = variable_features[:, 54]      # Dynamic/static max ratio (critical constraints)
    coef_rhs = variable_features[:, 59]           # Max positive coefficient/RHS tightness
    dual_sum = variable_features[:, 86]           # Dual-weighted constraint influence sum
    
    # Nonlinear interactions for cluster dynamics
    pseudocost_cluster = pseudocost * dyn_mean_ratio  # Reliable pseudocosts in active clusters
    constraint_tightness = coef_rhs * dual_sum        # Tight constraints under dual pressure
    cluster_volatility = dyn_max_ratio * dyn_std_ratio# Critical constraint variability in clusters
    
    return (
        params[0] * fractional +
        params[1] * pseudocost +
        params[2] * dyn_mean_ratio +
        params[3] * dyn_max_ratio +
        params[4] * coef_rhs +
        params[5] * dual_sum +
        params[6] * dyn_std_ratio +
        params[7] * pseudocost_cluster +
        params[8] * constraint_tightness +
        params[9] * cluster_volatility
    )
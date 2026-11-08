USED_FEATURES = [9, 43, 52, 53, 54, 86]
PARAMS = [0.561698, 0.615719, 0.539954, 0.75073, 0.607946, 0.162028, 0.271179, 0.315562, 0.891681]
BOUNDS = [[0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]]

def score_function(variable_features, params):
    # Enhanced cluster-aware features with dynamic variability
    fractional = variable_features[:, 9]        # Fractional branching urgency
    pseudocost = variable_features[:, 43]       # Pseudocost reliability
    dyn_mean = variable_features[:, 52]         # Dynamic/static constraint mean ratio
    dyn_std = variable_features[:, 53]          # Dynamic/static constraint std ratio
    dyn_max = variable_features[:, 54]          # Dynamic/static constraint max ratio
    dual_impact = variable_features[:, 86]      # Dual-weighted cluster pressure
    
    # Enhanced nonlinear interactions for cluster structure
    cluster_reliability = pseudocost * dyn_mean     # Stability in average cluster activity
    cluster_tightness = dual_impact * dyn_max       # Pressure in tight clusters
    cluster_volatility = dyn_std * dyn_max          # Dynamic variability in critical constraints
    
    # Optimized combination with cluster dynamics focus
    return (
        params[0] * fractional +
        params[1] * pseudocost +
        params[2] * dyn_mean +
        params[3] * dyn_std +                      # Direct dynamic variability component
        params[4] * dyn_max +
        params[5] * dual_impact +
        params[6] * cluster_reliability +
        params[7] * cluster_tightness +
        params[8] * cluster_volatility
    )
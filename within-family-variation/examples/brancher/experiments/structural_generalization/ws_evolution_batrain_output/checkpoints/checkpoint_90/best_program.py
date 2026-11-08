USED_FEATURES = [9, 43, 52, 54, 86]
PARAMS = [0.35, 0.45, 0.3, 0.25, 0.2, 0.4, 0.35, 0.3]
BOUNDS = [[0,1], [0,1], [0,1], [0,1], [0,1], [0,1], [0,1], [0,1]]

def score_function(variable_features, params):
    # Core structural features for clustered graphs
    frac_part = variable_features[:, 9]       # Fractional urgency
    pseudo_prod = variable_features[:, 43]    # Pseudocost reliability
    dyn_mean = variable_features[:, 52]       # Dynamic/static constraint mean ratio
    dyn_max = variable_features[:, 54]        # Dynamic/static constraint max ratio
    dual_pressure = variable_features[:, 86]  # Dual-weighted constraint tightness
    
    # Enhanced nonlinear interactions for cluster structure
    cluster_reliability = pseudo_prod * dyn_mean    # Reliability in average cluster activity
    structural_urgency = frac_part * dyn_max        # Fractionality in tight clusters
    cluster_pressure = dual_pressure * dyn_max      # Dual pressure in critical constraints
    
    # Optimized linear combination with cluster focus
    return (
        params[0] * frac_part +
        params[1] * pseudo_prod +
        params[2] * dyn_mean +
        params[3] * dyn_max +
        params[4] * dual_pressure +
        params[5] * cluster_reliability +
        params[6] * structural_urgency +
        params[7] * cluster_pressure
    )
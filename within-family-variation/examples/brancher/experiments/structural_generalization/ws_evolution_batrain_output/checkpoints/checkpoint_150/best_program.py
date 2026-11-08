USED_FEATURES = [9, 43, 52, 54, 86]
PARAMS = [0.35, 0.45, 0.3, 0.25, 0.2, 0.4, 0.35]
BOUNDS = [[0,1], [0,1], [0,1], [0,1], [0,1], [0,1], [0,1]]

def score_function(variable_features, params):
    # Streamlined features for clustered graph structure
    frac_part = variable_features[:, 9]       # Fractional branching urgency
    pseudo_prod = variable_features[:, 43]    # Pseudocost reliability
    dyn_mean = variable_features[:, 52]       # Dynamic/static constraint mean ratio
    dyn_max = variable_features[:, 54]        # Dynamic/static constraint max ratio
    dual_pressure = variable_features[:, 86]  # Dual-weighted constraint influence
    
    # Focused nonlinear interactions for cluster dynamics
    cluster_reliability = pseudo_prod * dyn_mean  # Stability in clustered regions
    structural_urgency = frac_part * dyn_max      # Urgency in tight clusters
    
    # Simplified linear combination with cluster focus
    return (
        params[0] * frac_part +
        params[1] * pseudo_prod +
        params[2] * dyn_mean +
        params[3] * dyn_max +
        params[4] * dual_pressure +
        params[5] * cluster_reliability +
        params[6] * structural_urgency
    )
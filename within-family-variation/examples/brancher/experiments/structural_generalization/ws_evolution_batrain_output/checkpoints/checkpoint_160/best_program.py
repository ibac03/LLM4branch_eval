USED_FEATURES = [9, 43, 52, 54, 86]
PARAMS = [0.35, 0.45, 0.3, 0.25, 0.2, 0.4, 0.35]
BOUNDS = [[0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]]

def score_function(variable_features, params):
    # Core structural features for clustered graphs
    fractional = variable_features[:, 9]      # Fractional branching urgency
    pseudocost = variable_features[:, 43]     # Historical pseudocost reliability
    dyn_mean = variable_features[:, 52]       # Dynamic/static constraint mean ratio
    dyn_max = variable_features[:, 54]        # Dynamic/static constraint max ratio
    dual_impact = variable_features[:, 86]    # Dual-weighted cluster pressure
    
    # Focused nonlinear interactions
    cluster_reliability = pseudocost * dyn_mean  # Stability in active clusters
    cluster_tightness = dual_impact * dyn_max    # Pressure in dense clusters
    
    # Streamlined combination with cluster focus
    return (
        params[0] * fractional +
        params[1] * pseudocost +
        params[2] * dyn_mean +
        params[3] * dyn_max +
        params[4] * dual_impact +
        params[5] * cluster_reliability +
        params[6] * cluster_tightness
    )
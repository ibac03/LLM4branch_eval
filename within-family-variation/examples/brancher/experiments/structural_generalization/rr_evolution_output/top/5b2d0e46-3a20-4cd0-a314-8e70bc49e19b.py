import numpy as np

USED_FEATURES = [37, 38, 43, 48, 51, 56, 90]
PARAMS = [0.5074229567324842, 1.5, 2.3013103184718195]
BOUNDS = [[0.5, 3.0], [0.05, 1.5], [1.5, 3.5]]

def score_function(variable_features, params):
    # Pseudocost × dynamic constraint evolution
    pseudocost_adaptive = variable_features[:, 43] * variable_features[:, 51]
    
    # Balanced fractional measure with dynamic engagement
    fractional_centrality = np.sqrt(variable_features[:, 37] * variable_features[:, 38] + 1e-8) * variable_features[:, 48]
    
    # Dual-sensitive constraint tightness
    constraint_tightness = variable_features[:, 56] * variable_features[:, 90]
    
    # Optimized parameter combination
    return (
        params[0] * pseudocost_adaptive +
        params[1] * fractional_centrality +
        params[2] * constraint_tightness
    )
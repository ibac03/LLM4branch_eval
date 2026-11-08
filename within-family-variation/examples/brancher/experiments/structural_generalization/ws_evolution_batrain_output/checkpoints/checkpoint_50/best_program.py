import numpy as np

USED_FEATURES = [9, 43, 52, 54, 86]
PARAMS = [0.35, 0.45, 0.3, 0.25, 0.2, 0.4, 0.35]
BOUNDS = [[0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]]

def score_function(variable_features, params):
    # Core structural features for clustered graphs
    fractional_part = variable_features[:, 9]    # LP fractional value (branching urgency)
    pseudocost_product = variable_features[:,43] # Pseudocost reliability indicator
    dyn_mean_ratio = variable_features[:,52]     # Dynamic/static constraint mean ratio
    dyn_max_ratio = variable_features[:,54]      # Dynamic/static constraint max ratio
    dual_weight_sum = variable_features[:,86]    # Dual-weighted constraint tightness
    
    # Nonlinear interactions capturing clustered graph dynamics
    pseudocost_cluster = pseudocost_product * dyn_mean_ratio
    fractional_bottleneck = fractional_part * dyn_max_ratio
    
    # Optimized linear combination with learned weights
    score = (
        params[0] * fractional_part +
        params[1] * pseudocost_product +
        params[2] * dyn_mean_ratio +
        params[3] * dyn_max_ratio +
        params[4] * dual_weight_sum +
        params[5] * pseudocost_cluster +
        params[6] * fractional_bottleneck
    )
    
    return score
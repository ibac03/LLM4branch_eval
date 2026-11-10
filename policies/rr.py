USED_FEATURES = [37, 38, 43, 48, 54, 56, 86, 89]
PARAMS = [0.5, 3.4734477218228044, 2.5, 1.9284545042767043]
BOUNDS = [[0.5, 3.0], [1.5, 3.5], [2.5, 4.5], [0.1, 2.0]]

import numpy as np

def score_function(variable_features, params):
    f37 = variable_features[:, 37]  # Fractional distance
    f38 = variable_features[:, 38]  # Ceiling distance
    f43 = variable_features[:, 43]  # Pseudocost product
    f48 = variable_features[:, 48]  # Dynamic constraint degree mean
    f54 = variable_features[:, 54]  # Dynamic/static max ratio
    f56 = variable_features[:, 56]  # Max coeff/RHS ratio
    f86 = variable_features[:, 86]  # Dual-weighted constraint sum
    f89 = variable_features[:, 89]  # Dual-weighted constraint min
    
    pseudocost_evolution = f43 * (np.sqrt(f54) + 0.7)
    fractional_signal = (f37 * f38 + 1e-8)**0.6
    constraint_impact = f56 * (np.log1p(f86*8) + np.sqrt(f89 + 1e-8))
    dynamic_engagement = f48**1.15
    
    return (
        params[0] * pseudocost_evolution +
        params[1] * fractional_signal +
        params[2] * constraint_impact +
        params[3] * dynamic_engagement
    )
USED_FEATURES = [37, 38, 43, 48, 54, 56, 86, 90]
PARAMS = [1.458666954041942, 3.5, 3.0, 1.423934269064213]
BOUNDS = [[0.5, 3.0], [1.5, 3.5], [1.0, 3.0], [0.1, 2.0]]

import numpy as np

def score_function(variable_features, params):
    f37 = variable_features[:, 37]  # Fractional distance
    f38 = variable_features[:, 38]  # Ceiling distance
    f43 = variable_features[:, 43]  # Pseudocost product
    f48 = variable_features[:, 48]  # Dynamic degree mean
    f54 = variable_features[:, 54]  # Dynamic/static max ratio
    f56 = variable_features[:, 56]  # Max coeff/RHS ratio
    f86 = variable_features[:, 86]  # Sum (1/sum_candidate weighted)
    f90 = variable_features[:, 90]  # Dual-weighted max activity
    
    pseudocost_evolution = f43 * (np.sqrt(f54) + 0.8)
    fractional_signal = (f37 * f38 + 1e-8)**0.65
    constraint_tightness = f56 * (np.log1p(f86) + np.sqrt(f90 + 1e-8))
    dynamic_engagement = f48**1.1
    
    return (
        params[0] * pseudocost_evolution +
        params[1] * fractional_signal +
        params[2] * constraint_tightness +
        params[3] * dynamic_engagement
    )
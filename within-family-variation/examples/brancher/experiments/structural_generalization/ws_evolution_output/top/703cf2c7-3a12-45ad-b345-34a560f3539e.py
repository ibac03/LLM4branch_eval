USED_FEATURES = [9, 22, 37, 38, 43, 52, 54, 85]
PARAMS = [3.0, 0.5, 2.065361761593447, 1.0697297598121325, 0.5, 0.16678823194763032]
BOUNDS = [[1.5, 3.0], [0.5, 2.0], [1.0, 3.0], [0.0, 2.0], [0.5, 1.5], [0.1, 0.5]]

import numpy as np

def score_function(variable_features, params):
    # Extract features for clustered graph dynamics
    f_frac = variable_features[:, 9]    # LP fractional part
    f_cnt = variable_features[:, 22]    # Constraint participation count
    f_dist = variable_features[:, 37]   # Fractional distance
    f_ceil = variable_features[:, 38]   # Ceiling distance
    f_pseudo = variable_features[:, 43] # Pseudocost product
    f_dyn_mean = variable_features[:, 52]  # Dynamic/static mean ratio
    f_dyn_max = variable_features[:, 54]   # Dynamic/static max ratio
    f_dual = variable_features[:, 85]   # Dual-weighted constraints

    # Strategic term combinations
    return (
        params[0] * (f_frac * f_pseudo) +  # Fractional-pseudocost synergy
        params[1] * (f_dist * (1 - f_ceil))**1.5 +  # Boundary proximity
        params[2] * f_pseudo * (f_dyn_mean + f_dyn_max) +  # Dynamic engagement
        params[3] * f_dual +  # Dual activity indicator
        params[4] * (f_dyn_mean * f_dyn_max) * np.sqrt(f_cnt + 1e-6) +  # Cluster dynamics
        params[5] * np.log1p(f_cnt) * f_dyn_max  # Participation-peak interaction
    )
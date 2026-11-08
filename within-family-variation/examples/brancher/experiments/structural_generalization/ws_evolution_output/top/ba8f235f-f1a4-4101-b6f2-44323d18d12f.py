USED_FEATURES = [9, 22, 37, 38, 43, 52, 54, 85]
PARAMS = [3.0, 0.5, 1.0, 1.0203101817592808, 0.5, 0.5]
BOUNDS = [[1.5, 3], [0.5, 2], [1.0, 3], [0.0, 2], [0.5, 1.5], [0.1, 0.5]]

import numpy as np

def score_function(variable_features, params):
    # Feature extraction for Watts-Strogatz graph dynamics
    f_frac = variable_features[:, 9]        # Fractional part of LP solution
    f_constraint_cnt = variable_features[:, 22]  # Constraint participation count
    f_frac_dist = variable_features[:, 37]       # Fractional distance from bounds
    f_ceil_dist = variable_features[:, 38]       # Ceiling distance
    f_pseudocost = variable_features[:, 43]      # Pseudocost product
    f_dyn_mean_ratio = variable_features[:, 52]  # Dynamic/static mean ratio
    f_dyn_max_ratio = variable_features[:, 54]   # Dynamic/static max ratio
    f_dual_count = variable_features[:, 85]      # Dual-weighted active constraint count

    # Nonlinear interactions for clustered small-world graphs
    pseudocost_fractional = params[0] * f_frac * f_pseudocost
    boundary_proximity = params[1] * (f_frac_dist * (1 - f_ceil_dist))**1.5
    dynamic_synergy = params[2] * f_pseudocost * (f_dyn_mean_ratio + f_dyn_max_ratio)
    dual_activity = params[3] * f_dual_count
    cluster_engagement = params[4] * (f_dyn_mean_ratio * f_dyn_max_ratio) * np.sqrt(f_constraint_cnt + 1e-6)
    log_dynamic = params[5] * np.log1p(f_constraint_cnt) * f_dyn_max_ratio

    return (pseudocost_fractional + boundary_proximity + dynamic_synergy 
            + dual_activity + cluster_engagement + log_dynamic)
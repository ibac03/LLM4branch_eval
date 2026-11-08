USED_FEATURES = [9, 22, 37, 43, 51, 53, 86]
PARAMS = [-0.4874966700711094, 0.3488813543998739, 2.0, 0.6550352232366925, 1.2582407935459976, -2.0, -1.6868158819114205, -2.0, -0.1641095217272419, 2.0]
BOUNDS = [[-2,2], [-2,2], [-2,2], [-2,2], [-2,2], [-2,2], [-2,2], [-2,2], [-2,2], [-2,2]]

import numpy as np

def score_function(variable_features, params):
    # Core feature extraction
    f_frac = variable_features[:, 9]       # Fractional part
    f_constraint_cnt = variable_features[:, 22]  # Constraint participation count
    f_frac_dist = variable_features[:, 37]       # Fractional distance
    f_pseudocost = variable_features[:, 43]      # Pseudocost product
    f_dyn_mean = variable_features[:, 51]        # Dynamic/static mean ratio
    f_dyn_max = variable_features[:, 53]         # Dynamic/static max ratio
    f_dual_active = variable_features[:, 86]     # Dual-weighted active constraint sum

    # Engineered interactions for clustered graph structure
    dynamic_engagement = f_dyn_mean * f_dyn_max           # Combined dynamic ratios
    pseudocost_boundary = f_pseudocost * (1 - f_frac_dist)  # Inverse fractional proximity
    dual_cluster_pressure = f_dual_active * f_dyn_mean    # Dual activity × dynamic structure

    # Concatenate base features and interactions
    feature_matrix = np.column_stack([
        f_frac, f_frac_dist, f_pseudocost,         # Base branching signals
        f_dyn_mean, f_dyn_max, f_dual_active,      # Constraint dynamics
        dynamic_engagement, pseudocost_boundary,   # Nonlinear interactions
        dual_cluster_pressure, f_constraint_cnt    # Structural awareness
    ])

    return np.dot(feature_matrix, params)
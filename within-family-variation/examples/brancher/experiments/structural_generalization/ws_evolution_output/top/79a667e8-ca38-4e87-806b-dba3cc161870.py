USED_FEATURES = [9, 22, 37, 43, 51, 53, 79]
PARAMS = [2.0, 0.14398044004343102, 2.0, 2.0, 2.0, 2.0, 2.0, 0.7968023157240776, 2.0, 2.0]
BOUNDS = [[-2, 2], [-2, 2], [-2, 2], [-2, 2], [-2, 2], [-2, 2], [-2, 2], [-2, 2], [-2, 2], [-2, 2]]

import numpy as np

def score_function(variable_features, params):
    # Core features extraction
    f_frac = variable_features[:, 9]     # Fractional part (original 9)
    f_static_cnt = variable_features[:, 22]  # Static constraint count (22)
    f_frac_dist = variable_features[:, 37]   # Fractional distance (37)
    f_pseudocost = variable_features[:, 43]  # Pseudocost product (43)
    f_dyn_mean = variable_features[:, 51]    # Dynamic/static mean ratio (51)
    f_dyn_max = variable_features[:, 53]     # Dynamic/static max ratio (53)
    f_active_sum = variable_features[:, 79]  # Active constraints sum (79)
    
    # Engineered interactions for WS graph structure
    dyn_engagement = f_dyn_mean * f_dyn_max          # Capture clustered dynamics
    pseudocost_effect = f_pseudocost * f_frac_dist   # Balance history and fractionalness
    constraint_density = f_static_cnt * f_active_sum # Dense constraint participation
    
    # Combine features and interactions
    feature_matrix = np.column_stack([
        f_frac, f_frac_dist, f_pseudocost,          # Base branching signals
        f_dyn_mean, f_dyn_max, f_active_sum,        # Constraint dynamics
        dyn_engagement, pseudocost_effect,          # Nonlinear interactions
        constraint_density, f_static_cnt            # Structural awareness
    ])
    
    return np.dot(feature_matrix, params)
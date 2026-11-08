USED_FEATURES = [9, 22, 37, 38, 43, 52, 54, 85]
PARAMS = [3.0, 0.5, 2.065361761593447, 1.0697297598121325, 0.5, 0.16678823194763032]
BOUNDS = [[1.5, 3], [0.5, 2], [1.0, 3], [0.0, 2], [0.5, 1.5], [0.1, 0.5]]

import numpy as np

def score_function(variable_features, params):
    # Key features for clustered graph branching
    f9 = variable_features[:, 9]    # LP fractional part
    f22 = variable_features[:, 22]  # Constraint participation count
    f37 = variable_features[:, 37]  # Fractional distance
    f38 = variable_features[:, 38]  # Ceiling distance
    f43 = variable_features[:, 43]  # Pseudocost product
    f52 = variable_features[:, 52]  # Dynamic/static mean ratio
    f54 = variable_features[:, 54]  # Dynamic/static max ratio
    f85 = variable_features[:, 85]  # Dual-weighted active constraints

    return (
        params[0] * (f9 * f43) +  # Fractional-pseudocost synergy
        params[1] * (f37 * (1 - f38))**1.5 +  # Boundary proximity impact
        params[2] * f43 * (f52 + f54) +  # Pseudocost-dynamic engagement
        params[3] * f85 +  # Dual constraint activity
        params[4] * (f52 * f54) * np.sqrt(f22 + 1e-6) +  # Cluster-aware dynamics
        params[5] * np.log1p(f22) * f54  # Participation-peak interaction
    )
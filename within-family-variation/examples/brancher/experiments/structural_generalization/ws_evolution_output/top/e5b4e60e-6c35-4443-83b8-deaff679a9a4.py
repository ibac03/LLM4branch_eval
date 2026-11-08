USED_FEATURES = [9, 22, 37, 38, 43, 52, 54, 85]
PARAMS = [2.741641363482568, 0.5, 2.3422837196324418, 0.5, 0.5, 1.5]
BOUNDS = [[1.5, 3], [0.5, 2], [1.0, 3], [0.5, 2], [0.5, 2], [0.1, 1.5]]

import numpy as np

def score_function(variable_features, params):
    # Feature extraction for clustered small-world graphs
    f9 = variable_features[:, 9]    # Fractional part of LP solution
    f22 = variable_features[:, 22]  # Constraint participation count
    f37 = variable_features[:, 37]  # Fractional distance from bounds
    f38 = variable_features[:, 38]  # Ceiling distance
    f43 = variable_features[:, 43]  # Pseudocost product
    f52 = variable_features[:, 52]  # Dynamic/static mean ratio
    f54 = variable_features[:, 54]  # Dynamic/static max ratio
    f85 = variable_features[:, 85]  # Dual-weighted active constraint count

    # Strategic nonlinear combinations optimized for local clustering
    return (
        params[0] * f9 * f43 +  # Fractional-pseudocost synergy
        params[1] * (f37 * (1 - f38))**1.5 +  # Boundary proximity impact
        params[2] * f43 * (f52 + f54) +  # Pseudocost-dynamic engagement
        params[3] * (f52 * f54) * np.sqrt(f22 + 1e-6) +  # Cluster dynamics
        params[4] * f85 +  # Dual-weighted constraint pressure
        params[5] * np.log1p(f22) * f54  # Log-scaled participation with max ratio
    )
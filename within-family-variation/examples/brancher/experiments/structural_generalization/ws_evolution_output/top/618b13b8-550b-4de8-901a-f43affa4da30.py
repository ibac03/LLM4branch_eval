USED_FEATURES = [9, 22, 37, 38, 42, 43, 52, 53, 54, 86]
PARAMS = [1.6725817674959107, 1.5, 2.424720927261916, 0.8801815985440324, 0.8, 1.7484911515617327]
BOUNDS = [[1.0, 2.5], [0.5, 1.5], [1.5, 3.0], [0.3, 1.2], [0.2, 0.8], [0.8, 1.8]]

import numpy as np

def score_function(variable_features, params):
    # Extract features for small-world graph branching
    f9 = variable_features[:, 9]    # Fractional part of LP solution
    f22 = variable_features[:, 22]  # Constraint participation count
    f37 = variable_features[:, 37]  # Fractional distance from bounds
    f38 = variable_features[:, 38]  # Ceiling distance
    f42 = variable_features[:, 42]  # Pseudocost sum
    f43 = variable_features[:, 43]  # Pseudocost product
    f52 = variable_features[:, 52]  # Dynamic/static mean ratio
    f53 = variable_features[:, 53]  # Dynamic/static min ratio
    f54 = variable_features[:, 54]  # Dynamic/static max ratio
    f86 = variable_features[:, 86]  # Dual-weighted active constraint sum

    # Strategic nonlinear combinations for clustered graphs
    return (
        params[0] * (f9 * f43) +  # Fractional-pseudocost synergy
        params[1] * (f37 * (1 - f38))**1.5 +  # Boundary proximity impact
        params[2] * f43 * (f52 + f54 + f53) +  # Pseudocost-dynamic engagement
        params[3] * (f52 * f53 * f54) * np.sqrt(f22 + 1e-6) +  # Cluster dynamics
        params[4] * f86 +  # Dual-weighted constraint pressure
        params[5] * f42  # Pseudocost reliability signal
    )
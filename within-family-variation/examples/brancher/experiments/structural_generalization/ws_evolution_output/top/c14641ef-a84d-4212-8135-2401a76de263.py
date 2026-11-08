USED_FEATURES = [9, 22, 37, 38, 43, 52, 54, 55, 85]
PARAMS = [3.2751275165578084, 0.6314998376083834, 1.6300736829212294, 1.6952071116585354, 1.3234967006683318, 0.12976935540860396, -0.22693330135258882]
BOUNDS = [
    [2.0, 3.5],
    [0.3, 1.2],
    [1.5, 3.0],
    [1.0, 2.5],
    [0.5, 1.8],
    [0.1, 0.5],
    [-1.0, 1.0]
]

import numpy as np

def score_function(variable_features, params):
    # Feature extraction with constraint tightness awareness
    f9 = variable_features[:, 9]    # LP fractional part
    f22 = variable_features[:, 22]  # Constraint participation count
    f37 = variable_features[:, 37]  # Fractional distance
    f38 = variable_features[:, 38]  # Ceiling distance
    f43 = variable_features[:, 43]  # Pseudocost product
    f52 = variable_features[:, 52]  # Dynamic/static mean ratio
    f54 = variable_features[:, 54]  # Dynamic/static max ratio
    f55 = variable_features[:, 55]  # Min coefficient/RHS tightness
    f85 = variable_features[:, 85]  # Dual-weighted active constraints

    # Strategic nonlinear interactions for clustered graphs
    return (
        params[0] * (f43 * np.sqrt(f9 + 1e-6)) +  # Enhanced pseudocost-fractional synergy
        params[1] * (f37**1.7 * (1 - f38)**0.9) +  # Adaptive boundary proximity
        params[2] * f43 * (f52**0.8 + f54**0.8) +  # Nonlinear dynamic engagement
        params[3] * (f85 * np.log1p(f55 + 1e-6)) +  # Dual-tightness interaction
        params[4] * (f52 * f54) * np.log1p(f22) +  # Cluster intensity with log scaling
        params[5] * f54 * np.exp(-0.5*f22) +  # Participation-decay dynamic
        params[6] * (f52 - f54) * f55  # Asymmetric cluster tightness
    )
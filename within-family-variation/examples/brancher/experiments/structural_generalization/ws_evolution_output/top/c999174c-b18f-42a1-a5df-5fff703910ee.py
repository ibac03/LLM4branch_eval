USED_FEATURES = [9, 22, 37, 38, 43, 52, 54, 55, 85]
PARAMS = [2.441931072497572, 2.0, 1.5139206800752745, -1.0, 1.2528085058444312, 1.0, 1.100748411907921]
BOUNDS = [
    [0.5, 3.5],    # Cluster dynamics scaling
    [0.3, 2.0],    # Fractional impact strength
    [1.5, 3.0],    # Pseudocost synergy factor
    [-1.0, 1.5],   # Dual constraint interaction
    [0.5, 2.5],    # Base pseudocost weight
    [0.1, 1.0],    # Log dynamic participation
    [0.5, 2.5]     # Fractional centrality focus
]

import numpy as np

def score_function(variable_features, params):
    # Key feature extraction for clustered graph dynamics
    f9 = variable_features[:, 9]    # LP solution fractional part
    f22 = variable_features[:, 22]  # Constraint participation count
    f37 = variable_features[:, 37]  # Fractional bound distance
    f38 = variable_features[:, 38]  # Ceiling proximity
    f43 = variable_features[:, 43]  # Pseudocost product
    f52 = variable_features[:, 52]  # Dynamic/static mean ratio
    f54 = variable_features[:, 54]  # Dynamic/static max ratio
    f55 = variable_features[:, 55]  # Min coefficient/RHS ratio
    f85 = variable_features[:, 85]  # Dual-weighted constraint sum

    # Strategic nonlinear combinations
    cluster_dynamics = params[0] * (f52 * f54) * np.sqrt(f22 + 1e-6)
    fractional_impact = params[1] * (f37 * (1 - f38))**1.5
    pseudocost_synergy = params[2] * f43 * (f52 + 0.6*f54)
    dual_pressure = params[3] * f85 * np.sqrt(f55 + 1e-6)
    structural_balance = params[4] * f43 + params[5] * np.log1p(f22) * f54
    centrality_boost = params[6] * f9 * (1 - f9) * f37

    return (cluster_dynamics + fractional_impact + pseudocost_synergy 
            + dual_pressure + structural_balance + centrality_boost)
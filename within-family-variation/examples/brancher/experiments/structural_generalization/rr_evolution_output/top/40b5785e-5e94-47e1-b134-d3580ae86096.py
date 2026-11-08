import numpy as np

USED_FEATURES = [37, 38, 43, 48, 51, 56, 89]
PARAMS = [0.5051728355861789, 1.9566700935858243, 3.0, 0.8780853132046164]
BOUNDS = [[0.5, 3.0], [0.1, 2.0], [0.5, 3.0], [0.1, 2.0]]

def score_function(variable_features, params):
    f37 = variable_features[:, 37]  # Fractional distance
    f38 = variable_features[:, 38]  # Ceiling distance
    f43 = variable_features[:, 43]  # Pseudocost product
    f48 = variable_features[:, 48]  # Dynamic constraint degree mean
    f51 = variable_features[:, 51]  # Dynamic/static max ratio
    f56 = variable_features[:, 56]  # Max coefficient/RHS ratio
    f89 = variable_features[:, 89]  # Dual-weighted constraint activity
    
    return (
        params[0] * f43 * f51 +  # Pseudocost history × dynamic structure
        params[1] * np.sqrt(f37 * f38 + 1e-8) +  # Balanced fractional measure
        params[2] * f56 * f89 +  # Constraint tightness × dual activity
        params[3] * f48  # Evolving constraint engagement
    )
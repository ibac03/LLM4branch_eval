import numpy as np

USED_FEATURES = [37, 38, 43, 51, 56, 89]
PARAMS = [0.0, 1.1579558572746653, 1.0, 0.0]
BOUNDS = [[0, 1], [0, 2], [0, 1], [0, 1]]

def score_function(variable_features, params):
    f37 = variable_features[:, 37]
    f38 = variable_features[:, 38]
    f43 = variable_features[:, 43]
    f51 = variable_features[:, 51]
    f56 = variable_features[:, 56]
    f89 = variable_features[:, 89]
    
    # Fractionality measure with geometric mean
    fractional_measure = np.sqrt(f37 * f38 + 1e-8)
    
    # Pseudocost adjusted by dynamic constraint engagement
    pseudocost_adjusted = f43 * (1.2 + f51**2)
    
    # Constraint tightness and dual activity interaction
    constraint_impact = (f56 * 2.0) * np.log1p(f89 * 10)
    
    # Balanced combination with parameters
    score = (
        params[0] * fractional_measure +
        params[1] * pseudocost_adjusted +
        params[2] * constraint_impact +
        params[3] * (f51 * f89)  # Dynamic constraint x dual activity
    )
    
    return score
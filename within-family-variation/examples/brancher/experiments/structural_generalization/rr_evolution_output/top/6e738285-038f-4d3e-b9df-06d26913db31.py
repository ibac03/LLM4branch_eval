import numpy as np

USED_FEATURES = [37, 38, 43, 51, 58, 89]
PARAMS = [0.5050767883482515, 1.0, 0.24828038449233206, 0.0, 0.20608390924957773]
BOUNDS = [[0, 1], [0, 1], [0, 1], [0, 1], [0, 1]]

def score_function(variable_features, params):
    # Feature interactions and nonlinear combinations
    pseudocost_dynamic = variable_features[:, 43] * (1 + variable_features[:, 51])
    fractional_term = (variable_features[:, 37] * variable_features[:, 38])**0.5
    constraint_impact = variable_features[:, 89] * (variable_features[:, 58] + 0.1)
    
    # Weighted combination with parameters
    score = (
        params[0] * pseudocost_dynamic +
        params[1] * fractional_term +
        params[2] * variable_features[:, 89] +
        params[3] * variable_features[:, 38] +
        params[4] * constraint_impact
    )
    return score
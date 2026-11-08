import numpy as np

USED_FEATURES = [37, 38, 43, 51, 58, 89]
PARAMS = [3.0, 1.6162571741894962, 0.18984863977997293, 0.4973284722543926, 0.2424958631165034]
BOUNDS = [[0.0, 3.0], [0.0, 2.0], [0.0, 1.0], [-0.5, 0.5], [0.0, 1.0]]

def score_function(variable_features, params):
    # Pseudocost interaction with dynamic constraint topology
    pseudocost_dynamic = variable_features[:, 43] * (1 + variable_features[:, 51])
    
    # Nonlinear fractional behavior measure
    fractional_term = np.sqrt(variable_features[:, 37] * variable_features[:, 38])
    
    # Dual-sensitive constraint tightness composite
    constraint_impact = variable_features[:, 89] * (variable_features[:, 58] + 0.1)
    
    # Parameterized combination of strategic signals
    score = (
        params[0] * pseudocost_dynamic +
        params[1] * fractional_term +
        params[2] * variable_features[:, 89] +
        params[3] * variable_features[:, 38] +
        params[4] * constraint_impact
    )
    return score
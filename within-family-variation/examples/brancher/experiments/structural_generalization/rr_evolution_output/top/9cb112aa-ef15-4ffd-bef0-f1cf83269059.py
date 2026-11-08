import numpy as np

USED_FEATURES = [37, 38, 43, 51, 58, 89]
PARAMS = [2.1805291870776093, 1.9544618402777474, 0.20362736283399646, 0.5132279619347982, 0.8155267852975259]
BOUNDS = [[0.5, 3.0], [0.1, 2.0], [0.0, 1.5], [0.0, 1.0], [0.1, 1.0]]

def score_function(variable_features, params):
    # Pseudocost interaction with dynamic constraint tightness
    pseudocost_dynamic = variable_features[:, 43] * (1 + variable_features[:, 51])
    
    # Fractionality measure combining distance to bounds
    fractional_balance = np.sqrt(variable_features[:, 37] * variable_features[:, 38])
    
    # Constraint tightness interaction with dual activity
    constraint_impact = variable_features[:, 89] * (variable_features[:, 58] + 0.1)
    
    # Base pseudocost and dual activity components
    base_pseudocost = variable_features[:, 43]
    dual_activity = variable_features[:, 89]
    
    # Weighted combination with learned parameters
    score = (
        params[0] * pseudocost_dynamic +
        params[1] * fractional_balance +
        params[2] * constraint_impact +
        params[3] * base_pseudocost +
        params[4] * dual_activity
    )
    return score
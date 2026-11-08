import numpy as np

USED_FEATURES = [9, 43, 48, 49, 56, 85]
PARAMS = [0.7094720755471932, 0.022837305663391042, 0.35143333226933365]
BOUNDS = [[0, 1], [0, 1], [0, 1]]

def score_function(variable_features, params):
    # Combined pseudocost and dynamic constraint features
    pseudocost_dynamic = variable_features[:, 43] * variable_features[:, 48]
    
    # Constraint tightness and dual activity interaction
    dual_tightness = variable_features[:, 56] * variable_features[:, 85]
    
    # Fractionality and dynamic constraint variability
    frac_variability = variable_features[:, 9] * variable_features[:, 49]
    
    return (
        params[0] * pseudocost_dynamic +
        params[1] * dual_tightness +
        params[2] * frac_variability
    )
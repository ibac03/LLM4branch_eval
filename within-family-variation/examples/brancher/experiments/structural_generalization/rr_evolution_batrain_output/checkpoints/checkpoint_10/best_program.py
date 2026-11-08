import numpy as np

USED_FEATURES = [9, 43, 52, 86]
PARAMS = [1.2, 0.8, 0.5]
BOUNDS = [[0, 5], [0, 5], [0, 5]]

def score_function(variable_features, params):
    # Extract key features
    frac_part = variable_features[:, 9]
    pseudo_product = variable_features[:, 43]
    dyn_ratio_mean = variable_features[:, 52]
    dual_active_sum = variable_features[:, 86]
    
    # Feature transformations
    frac_proximity = 1 - 2 * np.abs(frac_part - 0.5)
    
    # Combine features with parameters
    score = (
        params[0] * pseudo_product * frac_proximity +
        params[1] * dyn_ratio_mean +
        params[2] * dual_active_sum
    )
    
    return score
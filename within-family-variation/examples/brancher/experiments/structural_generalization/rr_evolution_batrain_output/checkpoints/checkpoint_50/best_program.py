import numpy as np

USED_FEATURES = [9, 37, 43, 52, 56, 86]
PARAMS = [1.5, 0.8, 0.5, 0.7]
BOUNDS = [[0, 5], [0, 5], [0, 5], [0, 5]]

def score_function(variable_features, params):
    # Key feature extraction
    frac_part = variable_features[:, 9]
    pseudo_product = variable_features[:, 43]
    dyn_ratio = variable_features[:, 52]
    max_coef_rhs = variable_features[:, 56]
    frac_distance = variable_features[:, 37]
    dual_activity = variable_features[:, 86]
    
    # Feature transformations
    frac_proximity = 1 - 2 * np.abs(frac_part - 0.5)
    constraint_impact = dyn_ratio * max_coef_rhs  # Dynamic activity × constraint tightness
    
    # Parameterized combination
    return (
        params[0] * pseudo_product * frac_proximity +  # Pseudocost × centrality
        params[1] * constraint_impact +                # Dynamic constraint effect
        params[2] * frac_distance +                    # Raw fractional influence
        params[3] * dual_activity                      # Dual solution impact
    )
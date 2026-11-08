import numpy as np

USED_FEATURES = [37, 38, 43, 54, 56, 89]
PARAMS = [1.7331521629529163, 0.9411494015622748, 2.6413551795122516]
BOUNDS = [[0.1, 2.5], [0.1, 2.0], [0.5, 3.0]]

def score_function(variable_features, params):
    # Core feature extraction
    fractional_dist = variable_features[:, 37]
    ceil_dist = variable_features[:, 38]
    pseudocost_prod = variable_features[:, 43]
    dyn_max_ratio = variable_features[:, 54]
    coeff_rhs = variable_features[:, 56]
    dual_active = variable_features[:, 89]
    
    # Nonlinear feature interactions
    pseudocost_dynamic = pseudocost_prod * (1.0 + dyn_max_ratio)
    fractional_measure = np.sqrt(fractional_dist * (ceil_dist + 1e-8))
    constraint_tightness = coeff_rhs * np.log1p(dual_active * 10)
    
    # Parameter-weighted combination
    score = (
        params[0] * pseudocost_dynamic +
        params[1] * fractional_measure +
        params[2] * constraint_tightness
    )
    
    return score
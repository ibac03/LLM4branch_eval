import numpy as np

USED_FEATURES = [9, 43, 54, 56, 86]
PARAMS = [2.1, 0.95, 0.82]
BOUNDS = [[1.5, 3.0], [0.7, 1.8], [0.6, 1.2]]

def score_function(variable_features, params):
    # Core feature extraction
    frac_part = variable_features[:, 9]        # Fractional part of LP solution
    pseudo_product = variable_features[:, 43]  # Pseudocost product (up*down)
    dyn_ratio_max = variable_features[:, 54]   # Max dynamic/static constraint degree ratio
    coef_rhs_max = variable_features[:, 56]    # Max positive coefficient/RHS ratio
    dual_active_sum = variable_features[:, 86] # Dual-weighted active constraint sum
    
    # Nonlinear transformations
    frac_ambiguity = 4.0 * frac_part * (1 - frac_part)  # Peaked at 0.5
    constraint_tightness = np.sqrt(dyn_ratio_max * coef_rhs_max + 1e-8)  # Geometric mean
    
    # Parametrized combination
    return (
        params[0] * pseudo_product * frac_ambiguity +  # Reliability × centrality
        params[1] * constraint_tightness +              # Dynamic constraint tightness
        params[2] * dual_active_sum                      # Dual solution pressure
    )
import numpy as np

USED_FEATURES = [9, 43, 52, 54, 56, 86]
PARAMS = [2.1295, 2, 1.5, 0.6]
BOUNDS = [[1.5, 3.0], [0.5, 2.0], [0.0, 1.5], [0.6, 1.2]]

def score_function(variable_features, params):
    # Extract features with high predictive power
    frac_part = variable_features[:, 9]        # Fractional part of LP solution
    pseudo_product = variable_features[:, 43]  # Pseudocost reliability product
    dyn_mean_ratio = variable_features[:, 52]  # Dynamic/static constraint degree (mean)
    dyn_max_ratio = variable_features[:, 54]   # Dynamic/static constraint degree (max)
    coef_rhs_max = variable_features[:, 56]    # Max coefficient-to-RHS ratio
    dual_active_sum = variable_features[:, 86] # Dual-weighted constraint sum
    
    # Nonlinear feature transformations
    frac_ambiguity = 4.0 * frac_part * (1 - frac_part)  # Peaked ambiguity at 0.5
    constraint_tightness = np.sqrt(dyn_max_ratio * coef_rhs_max + 1e-8)  # Structural tightness
    
    # Parameterized combination for generalization
    return (
        params[0] * pseudo_product * frac_ambiguity +  # Pseudocost × centrality
        params[1] * constraint_tightness +              # Dynamic constraint tightness
        params[2] * dyn_mean_ratio +                    # Average dynamic evolution
        params[3] * dual_active_sum                     # Dual solution guidance
    )
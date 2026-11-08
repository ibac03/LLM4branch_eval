import numpy as np

USED_FEATURES = [37, 38, 43, 48, 54, 56, 86]
PARAMS = [0.5118520538072138, 2.2011287283791443, 1.5, 0.7245032135966031]
BOUNDS = [[0.5, 3.0], [0.5, 3.0], [1.5, 4.0], [0.1, 2.0]]

def score_function(variable_features, params):
    f37 = variable_features[:, 37]  # Fractional distance
    f38 = variable_features[:, 38]  # Ceiling distance
    f43 = variable_features[:, 43]  # Pseudocost product
    f48 = variable_features[:, 48]  # Dynamic constraint degree mean
    f54 = variable_features[:, 54]  # Dynamic/static max ratio
    f56 = variable_features[:, 56]  # Max coeff/RHS ratio
    f86 = variable_features[:, 86]  # Dual-weighted constraint sum
    
    pseudocost_evolution = f43 * (np.sqrt(f54) + 0.8)  # Amplified dynamic topology signal
    fractional_signal = (f37 * f38 + 1e-8)**0.65  # Optimized nonlinear balance
    constraint_impact = f56 * np.log1p(f86*4 + 1e-8)  # Scaled dual activity interaction
    dynamic_engagement = f48**1.1  # Non-linear constraint participation
    
    return (
        params[0] * pseudocost_evolution +
        params[1] * fractional_signal +
        params[2] * constraint_impact +
        params[3] * dynamic_engagement
    )
import numpy as np

USED_FEATURES = [37, 38, 41, 43, 48, 49, 51, 55, 56, 82]
PARAMS = [0.9557838694021792, 0.06993239761184598, 0.0, 0.6581767384904489, 0.1694649771184787, 1.0]
BOUNDS = [[0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]]

def score_function(variable_features, params):
    fractional = variable_features[:, 37]
    ceiling = variable_features[:, 38]
    pseudo_ratio = variable_features[:, 41]
    pseudo_product = variable_features[:, 43]
    dynamic_mean = variable_features[:, 48]
    dynamic_std = variable_features[:, 49]
    dynamic_max_ratio = variable_features[:, 51]
    coeff_rhs_min = variable_features[:, 55]
    coeff_rhs_max = variable_features[:, 56]
    sum_dual = variable_features[:, 82]
    
    score = (
        params[0] * pseudo_product * dynamic_max_ratio +
        params[1] * (fractional + 0.7*ceiling) +
        params[2] * np.sqrt(pseudo_ratio + 1e-6) +
        params[3] * dynamic_std * dynamic_mean +
        params[4] * (coeff_rhs_min + coeff_rhs_max) +
        params[5] * np.log1p(sum_dual)
    )
    return score
USED_FEATURES = [37, 42, 43, 50, 56, 67]
PARAMS = [0.10292142148152585, 0.0, 0.5158189047576024, 0.3268790261800403, 0.21454174879055957, 0.0]
BOUNDS = [[0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]]

def score_function(variable_features, params):
    return (
        params[0] * variable_features[:, 37] +  # Fractional distance
        params[1] * variable_features[:, 42] +  # Pseudocost sum
        params[2] * variable_features[:, 43] +  # Pseudocost product
        params[3] * variable_features[:, 50] +  # Dynamic/static degree ratio
        params[4] * variable_features[:, 56] +  # Max coefficient-to-RHS (positive)
        params[5] * variable_features[:, 67]    # Active constraints count (unit weighted)
    )
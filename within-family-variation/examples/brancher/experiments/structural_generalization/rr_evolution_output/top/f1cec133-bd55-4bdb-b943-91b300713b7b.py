USED_FEATURES = [37, 39, 40, 43, 52, 54, 56, 58, 79, 60]
PARAMS = [0.8415765560618453, 0.8128293433769029, 0.11554565193624487, 0.036620777455174586, 0.148135265004384, 0.12800283153671635, 0.0, 0.6799994058489408, 0.42882329382232676, 0.9131924404225442]
BOUNDS = [[0,1], [0,1], [0,1], [0,1], [0,1], [0,1], [0,1], [0,1], [0,1], [0,1]]

def score_function(variable_features, params):
    return (
        params[0] * variable_features[:, 37] +  # Fractional distance
        params[1] * variable_features[:, 39] +  # Pseudocost up
        params[2] * variable_features[:, 40] +  # Pseudocost down
        params[3] * variable_features[:, 43] +  # Pseudocost product
        params[4] * variable_features[:, 52] +  # Dynamic/static mean degree ratio
        params[5] * variable_features[:, 54] +  # Dynamic/static max degree ratio
        params[6] * variable_features[:, 56] +  # Max positive coefficient-to-RHS
        params[7] * variable_features[:, 58] +  # Max negative coefficient-to-RHS
        params[8] * variable_features[:, 79] +  # Active constraints (1/sum_candidate)
        params[9] * variable_features[:, 60]    # Max positive interaction ratio
    )
USED_FEATURES = [22]
PARAMS = [1.0]
BOUNDS = [[0, 1]]

def score_function(variable_features, params):
    # variable in the largest number of constraints
    return params[0] * variable_features[:, 22]

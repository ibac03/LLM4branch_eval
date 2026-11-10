USED_FEATURES = [43]
PARAMS = [1.0]
BOUNDS = [[0, 1]]

def score_function(variable_features, params):
    # largest pseudocost product
    return params[0] * variable_features[:, 43]

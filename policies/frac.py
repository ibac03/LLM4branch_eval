USED_FEATURES = [9]
PARAMS = [1.0]
BOUNDS = [[0, 1]]

def score_function(variable_features, params):
    # most fractional variable: LP value nearest to 0.5
    f = variable_features[:, 9]
    return params[0] * f * (1.0 - f)

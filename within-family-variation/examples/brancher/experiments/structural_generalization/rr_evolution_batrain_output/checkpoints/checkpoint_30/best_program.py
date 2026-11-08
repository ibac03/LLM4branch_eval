import numpy as np

USED_FEATURES = [37, 38, 39, 40, 42, 43, 48, 54, 58, 86]
PARAMS = [0.8, -0.6, 0.5, 0.5, 0.7, 1.5, 0.6, 0.9, 0.4, 0.8]
BOUNDS = [[-2.0, 2.0]] * 10

def score_function(variable_features, params):
    selected_features = variable_features[:, USED_FEATURES]
    return np.dot(selected_features, params)
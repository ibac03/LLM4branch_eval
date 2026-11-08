import numpy as np

USED_FEATURES = [8, 43, 52]
PARAMS = [0.35, 0.45, 0.2]
BOUNDS = [[0, 1], [0, 1], [0, 1]]

def score_function(variable_features, params):
    # Combine pseudocost effectiveness, dynamic clustering involvement, and fractional branching
    pseudocost_term = variable_features[:, 43] * params[0]
    dynamic_cluster_term = variable_features[:, 52] * params[1]
    fractional_term = variable_features[:, 8] * params[2]
    
    return pseudocost_term + dynamic_cluster_term + fractional_term
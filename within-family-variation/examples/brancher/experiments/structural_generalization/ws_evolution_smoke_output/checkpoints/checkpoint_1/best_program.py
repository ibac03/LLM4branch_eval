import numpy as np

USED_FEATURES = [8, 42, 43, 48, 51, 68]
PARAMS = [0.5, 0.5, 0.5, 0.5]
BOUNDS = [[0, 2], [0, 2], [0, 2], [0, 2]]

def score_function(variable_features, params):
    # Pseudocost product interacts with fractional part
    pseudocost_fractional = variable_features[:, 42] * variable_features[:, 8]
    
    # Dynamic constraint interaction term
    dynamic_interaction = variable_features[:, 48] * variable_features[:, 51]
    
    # Active constraint influence (unit weighting)
    constraint_influence = variable_features[:, 68]
    
    # Original cutoff ratio component
    cutoff_ratio = variable_features[:, 43]
    
    # Combine components with learned weights
    return (params[0] * pseudocost_fractional +
            params[1] * dynamic_interaction +
            params[2] * constraint_influence +
            params[3] * cutoff_ratio)
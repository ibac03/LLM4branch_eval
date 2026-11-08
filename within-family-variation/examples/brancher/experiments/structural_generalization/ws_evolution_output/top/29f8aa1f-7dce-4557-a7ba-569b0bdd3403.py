import numpy as np

USED_FEATURES = [37, 43, 51, 53, 67]
PARAMS = [0.5202395765753861, 0.9976965384350672, 0.7939154243819702, 0.8162515963865847, 0.775451154535898]
BOUNDS = [[0,1], [0,1], [0,1], [0,1], [0,1]]

def score_function(variable_features, params):
    # Extract features for clustered graph structure
    f37 = variable_features[:, 37]  # Fractional distance (solution centrality)
    f43 = variable_features[:, 43]  # Pseudocost product (historical effectiveness)
    f51 = variable_features[:, 51]  # Dynamic/static mean ratio (constraint engagement)
    f53 = variable_features[:, 53]  # Dynamic/static max ratio (peak engagement)
    f67 = variable_features[:, 67]  # Active constraint count (current participation)
    
    # Strategic combination with feature interactions
    return (
        params[0] * f37 +  # Base fractional importance
        params[1] * f43 * f51 +  # Pseudocost modulated by average engagement
        params[2] * f53 +  # Peak constraint engagement
        params[3] * f67 +  # Current constraint activity
        params[4] * (f51 * f53)  # Synergy between average and peak engagement
    )
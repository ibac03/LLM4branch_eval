import numpy as np

USED_FEATURES = [8, 22, 37, 43, 51, 53, 67]
PARAMS = [-0.8886461271219593, 0.751133639421643, 0.9335227187783138, 0.988137393363486, 0.5551998191251445, 0.8368749088866834, 0.9871996800523191, 0.5222904095173584, 0.3241328765906676, 0.15299107662188963]
BOUNDS = [[-1, 1], [-1, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]]

def score_function(variable_features, params):
    # Extract core features for WS graph structure
    base_features = variable_features[:, USED_FEATURES]  # (n,7)
    f8 = base_features[:, 0]   # LP solution value
    f37 = base_features[:, 2]  # Fractional distance
    f43 = base_features[:, 3]  # Pseudocost product
    f51 = base_features[:, 4]  # Dynamic/static mean ratio
    f53 = base_features[:, 5]  # Dynamic/static max ratio
    f67 = base_features[:, 6]  # Active constraints count
    
    # Create nonlinear interactions capturing WS graph dynamics
    cluster_activity = f51 * f53    # Combined dynamic engagement
    pseudocost_impact = f43 * f37   # Pseudocost x fractional positioning
    lp_conflict = f8 * f67          # LP value x current conflict involvement
    
    # Concatenate engineered features (7 base + 3 interaction)
    feature_vector = np.column_stack((
        base_features,
        cluster_activity,
        pseudocost_impact,
        lp_conflict
    ))
    
    # Weighted linear combination with learned parameters
    return np.dot(feature_vector, params)
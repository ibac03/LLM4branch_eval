import numpy as np

USED_FEATURES = [22, 37, 38, 43, 52, 54, 55, 85]
PARAMS = [0.0, 0.7345714463891698, 2.2761939104115343, 0.0, 0.5072561788284717, 0.0]
BOUNDS = [[0, 2], [0, 2], [0, 3], [0, 2], [0, 2], [0, 2]]

def score_function(variable_features, params):
    # Feature extraction
    f22 = variable_features[:, 22]  # Constraint participation count
    f37 = variable_features[:, 37]  # Fractional distance
    f38 = variable_features[:, 38]  # Ceiling distance
    f43 = variable_features[:, 43]  # Pseudocost product
    f52 = variable_features[:, 52]  # Dynamic/static mean ratio
    f54 = variable_features[:, 54]  # Dynamic/static max ratio
    f55 = variable_features[:, 55]  # Min coefficient/RHS ratio
    f85 = variable_features[:, 85]  # Dual-weighted active constraints

    # Strategic feature interactions for clustered graphs
    cluster_dynamic = params[0] * (f52 * f54) * np.sqrt(f22 + 1e-6)
    fractional_impact = params[1] * (f37 * (1 - f38))**1.5
    pseudocost_synergy = params[2] * f43 * (f52 + f54)
    dual_constraint = params[3] * f85 * f55
    base_pseudocost = params[4] * f43
    log_dynamic = params[5] * np.log1p(f22) * f54

    return (cluster_dynamic + fractional_impact + pseudocost_synergy 
            + dual_constraint + base_pseudocost + log_dynamic)
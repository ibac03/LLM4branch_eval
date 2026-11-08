USED_FEATURES = [9, 22, 37, 43, 51, 55, 79]
PARAMS = [1.0, 1.0, 1.0, 0.6494204028938231, 0.00961681041845619, -1.0, 1.0, 0.20897577311329085, -0.19073804535486472, -0.1699717376872456]
BOUNDS = [[-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-2, 2], [-1, 1], [-2, 2]]

def score_function(variable_features, params):
    f9 = variable_features[:, 9]   # Fractional part
    f22 = variable_features[:, 22] # Constraint participation count
    f37 = variable_features[:, 37] # Fractional distance
    f43 = variable_features[:, 43] # Pseudocost product
    f51 = variable_features[:, 51] # Dynamic/static degree ratio
    f55 = variable_features[:, 55] # Min coefficient/RHS (positive)
    f79 = variable_features[:, 79] # Active constraints (1/sum_candidate)
    
    return (
        params[0] * f9 + 
        params[1] * f22 +
        params[2] * (f37 * f55) +  # Distance-tightness interaction
        params[3] * f43 +
        params[4] * (f51 * f22) +  # Dynamic ratio-participation interaction
        params[5] * f55 +
        params[6] * f79 +
        params[7] * (f9 * f43) +   # Fractional-pseudocost synergy
        params[8] * (f51**2) +     # Nonlinear dynamic ratio emphasis
        params[9] * (f22 * f79)    # Participation-activity coupling
    )
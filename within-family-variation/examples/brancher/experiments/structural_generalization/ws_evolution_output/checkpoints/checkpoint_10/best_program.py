USED_FEATURES = [9, 37, 43, 51, 79]
PARAMS = [0.5, 0.5, 0.5, 0.5, 0.5]
BOUNDS = [[-2, 2], [-2, 2], [-2, 2], [-2, 2], [-2, 2]]

def score_function(variable_features, params):
    f9 = variable_features[:, 9]   # Fractional part of LP solution
    f37 = variable_features[:, 37] # Fractional distance from bounds
    f43 = variable_features[:, 43] # Pseudocost product (up*down)
    f51 = variable_features[:, 51] # Dynamic/static constraint degree ratio
    f79 = variable_features[:, 79] # Active constraints (1/sum_candidate)
    
    return (params[0] * f9 + 
            params[1] * f37 + 
            params[2] * f43 + 
            params[3] * f51 + 
            params[4] * f79)
USED_FEATURES = [37, 42, 43, 48, 51]
PARAMS = [0.6976568054296811, 1.0, 0.32536745812181106, 0.0]
BOUNDS = [[0, 1], [0, 1], [0, 1], [0, 1]]

def score_function(variable_features, params):
    pseudocost_product = variable_features[:, 43]
    dynamic_ratio = variable_features[:, 51]
    fractional_dist = variable_features[:, 37]
    pseudocost_sum = variable_features[:, 42]
    dynamic_mean = variable_features[:, 48]

    score = (
        params[0] * pseudocost_product * dynamic_ratio +
        params[1] * fractional_dist +
        params[2] * pseudocost_sum +
        params[3] * dynamic_mean
    )
    return score

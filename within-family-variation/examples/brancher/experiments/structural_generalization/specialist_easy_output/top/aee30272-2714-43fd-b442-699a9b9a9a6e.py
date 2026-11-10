USED_FEATURES = [0, 7, 9, 22]
PARAMS = [0.0, 0.0, 1.5058252455798085, 2.0]
BOUNDS = [[0.0, 2.0], [0.0, 2.0], [0.0, 2.0], [0.0, 2.0]]
import numpy as np

def score_function(variable_features, params):
    obj_coeff = variable_features[:, 0]          # objective coefficient
    reduced_cost = variable_features[:, 7]       # normalized reduced cost
    fractional_part = variable_features[:, 9]    # LP solution fractional part
    constraint_count = variable_features[:, 22]  # number of participating constraints

    # Prefer variables whose LP value is near 0.5; this peaks at 0.5.
    branching_preference = 4.0 * fractional_part * (1.0 - fractional_part)

    score = (
        params[0] * obj_coeff +
        params[1] * reduced_cost +
        params[2] * branching_preference +
        params[3] * constraint_count
    )
    return score
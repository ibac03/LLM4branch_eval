"""
Standalone SCIP/ecole helpers -- copied out of the main LLM4Branch repo's
utils/utils.py so this GNN_Code folder has zero dependency on the rest of
that repo. Only the pieces actually used by this GNN pipeline are included.
"""
import numpy as np


def get_scip_params(time_limit=None, gap_limit=None):
    if time_limit:
        return {
            'separating/maxrounds': 0,
            'presolving/maxrestarts': 0,
            "branching/vanillafullstrong/idempotent": True,
            'limits/time': time_limit,
            'timing/clocktype': 1,
        }
    elif gap_limit:
        return {
            'separating/maxrounds': 0,
            'presolving/maxrestarts': 0,
            "branching/vanillafullstrong/idempotent": True,
            'limits/gap': gap_limit,
            'timing/clocktype': 1,
        }
    else:
        return {
            'separating/maxrounds': 0,
            'presolving/maxrestarts': 0,
            "branching/vanillafullstrong/idempotent": True,
            'limits/time': 3600,
            'timing/clocktype': 1,
        }


def normalize(variable_features, used_features):
    subset = variable_features[:, used_features]
    min_vals = np.min(subset, axis=0)
    max_vals = np.max(subset, axis=0)
    range_vals = max_vals - min_vals
    safe_range = np.where(range_vals > 0, range_vals, 1)
    variable_features[:, used_features] = (subset - min_vals) / safe_range
    return variable_features


def geometric_mean_stable(data, shift=1):
    a = np.asarray(data)
    if a.size == 0 or np.any(a < 0):
        return 0.0
    return float(np.exp(np.mean(np.log(a + shift)))) - shift


def numerical_mean_stable(arr):
    if len(arr) == 0:
        return 0.0
    return sum(arr) / len(arr)

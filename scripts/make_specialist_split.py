"""Pick the specialist's train/valid/test instances from the official 1300-instance easy run.

Hard set = slowest 10% by policy CPU time. 4 train + 4 valid are drawn (fixed RNG) from hard
instances that solve in 18-45 s so each tuning step stays cheap; the rest of the hard set is the
held-out test set.

    python scripts/make_specialist_split.py results/tasks1-3_easy/per_instance_easy.csv results/task4_pilot/specialist_split.json
"""
import json
import sys

import numpy as np
import pandas as pd

df = pd.read_csv(sys.argv[1])
k = int(round(0.1 * len(df)))
hard = df.nlargest(k, "policy_time")
pool = hard[(hard.policy_time >= 18) & (hard.policy_time <= 45)].sort_values("seed")
rng = np.random.default_rng(20261103)
picked = rng.choice(pool.seed.to_numpy(), size=8, replace=False)
train, valid = sorted(picked[:4].tolist()), sorted(picked[4:].tolist())
test = sorted(set(hard.seed) - set(train) - set(valid))
split = {
    "definition": "slowest 10% by LLM4Branch policy CPU time, official tester, easy (750 nodes), seeds 999999999+",
    "policy_time_cutoff_s": float(hard.policy_time.min()),
    "train_seeds": [int(s) for s in train], "valid_seeds": [int(s) for s in valid],
    "test_hard_seeds": [int(s) for s in test],
    "train_info": df[df.seed.isin(train)][["seed", "policy_time", "rpb_time", "policy_nodes", "rpb_nodes", "ratio"]].to_dict("records"),
    "valid_info": df[df.seed.isin(valid)][["seed", "policy_time", "rpb_time", "policy_nodes", "rpb_nodes", "ratio"]].to_dict("records"),
}
json.dump(split, open(sys.argv[2], "w"), indent=1)
print(json.dumps({k: v for k, v in split.items() if k != "test_hard_seeds"}, indent=1))
print("held-out hard test instances:", len(test))

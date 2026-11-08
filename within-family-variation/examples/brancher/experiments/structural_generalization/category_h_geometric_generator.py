"""
Category H: geometric (disk) covering.

A completely different generative paradigm from every other category tried
so far (A/B/C/E/F/G): rows are points in the 2D unit square, and columns
are disks (random center + radius) that cover every point within their
radius. Cost is proportional to disk area (a bigger disk costs more).

Note: 1D interval covering (points on a line, columns = intervals) has
the consecutive-ones property and is totally unimodular -- its LP
relaxation is ALWAYS integral, so it would solve trivially at the root
for every method every time. 2D disk covering does not have this
property in general, so real branching is required.

Same n_rows/n_cols scale as the standard "medium" setcover tier.
"""
import numpy as np
import pyscipopt

N_POINTS = 500     # rows
N_DISKS = 500      # columns
RADIUS_LOW, RADIUS_HIGH = 0.06, 0.22   # calibrated to give a reasonable avg degree
COST_PER_AREA = 40.0  # cost = COST_PER_AREA * pi * r^2, tuned to give a sane cost range


def generate_instance(seed, n_points=N_POINTS, n_disks=N_DISKS):
    rng = np.random.RandomState(seed)

    points = rng.uniform(0.0, 1.0, size=(n_points, 2))
    centers = rng.uniform(0.0, 1.0, size=(n_disks, 2))
    radii = rng.uniform(RADIUS_LOW, RADIUS_HIGH, size=n_disks)

    coverage = [set() for _ in range(n_points)]
    for j in range(n_disks):
        d2 = np.sum((points - centers[j]) ** 2, axis=1)
        covered = np.where(d2 <= radii[j] ** 2)[0]
        for i in covered:
            coverage[i].add(j)

    costs = COST_PER_AREA * np.pi * radii ** 2

    # feasibility repair: any uncovered point gets a tiny disk of its own
    # (cost proportional to a minimal radius, so it's usable but not free)
    min_cost = float(COST_PER_AREA * np.pi * RADIUS_LOW ** 2)
    extra_cols = []
    for i in range(n_points):
        if not coverage[i]:
            j = n_disks + len(extra_cols)
            coverage[i].add(j)
            extra_cols.append(min_cost)
    if extra_cols:
        costs = np.concatenate([costs, extra_cols])
    n_cols_total = n_disks + len(extra_cols)

    model = pyscipopt.Model()
    model.hideOutput()
    variables = [model.addVar(f"x{j}", vtype="B", obj=float(costs[j])) for j in range(n_cols_total)]
    for i in range(n_points):
        model.addCons(pyscipopt.quicksum(variables[j] for j in coverage[i]) >= 1)
    model.setMinimize()

    stats = {
        "n_rows": n_points, "n_cols": n_cols_total,
        "n_repair_cols": len(extra_cols),
        "avg_degree": float(np.mean([len(coverage[i]) for i in range(n_points)])),
    }
    return model, stats


if __name__ == "__main__":
    model, stats = generate_instance(seed=0)
    print("Instance stats:", stats)
    print(f"n_vars={model.getNVars()}, n_conss={model.getNConss()}")

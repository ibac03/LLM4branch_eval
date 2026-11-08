"""
Specialist evaluator: BA independent-set instances at n=750 (calibrated to
reproduce the real n=1000 worst-decile phenomenon -- see
calibrate_specialist_training_size.py: median ratio 0.356 vs 1000's 0.33,
worst-decile cutoff 0.580 vs 0.576, rare-loser rate ~0.7-1.0% vs 1.08%,
solves in seconds instead of minutes), restricted to 8 hand-curated seeds
spanning both worst-decile sub-types found in the n=1000 analysis: genuine
losers (ratio>1, policy actually slower than RPB) and worst-decile
non-losers (policy still faster, just less decisively). 4 train / 4 valid,
each split containing 2 of each type and a mix of fast/slow instances for
balanced per-candidate evaluation cost.

Same nested-pool forkserver fix as evaluator_rr_batrain.py /
evaluator_ws_batrain.py -- some of these curated seeds solve in the
~200-260s range, long enough for the same fork-hostility issue (TBB inside
libscip, Accelerate/GCD via numpy) that broke the original `fork`-context
nested pool at BA-matched difficulty tiers earlier in this investigation.
"""
import time
import numpy as np
import ecole
import psutil
import os
import sys
import multiprocessing
from multiprocessing import current_process, TimeoutError

# forkserver (not fork): nested Pool here can live for minutes on the two
# slow curated seeds (~200-260s each), long enough for fork() to land while
# libscip's TBB scheduler or numpy's Accelerate/GCD thread pool holds an
# internal lock in the openevolve worker -- forkserver forks from a clean,
# single-threaded helper instead, side-stepping the whole class of bug.
# Preload + self-alias resolve pickle-by-reference for execute_specialist
# (openevolve loads this file as "evaluation_module"; the helper preloads
# it as "evaluator_specialist" -- both names must point at one module).
# PYTHONPATH must include this directory for the helper to find the
# preload module at all (see run scripts).
_fork_ctx = multiprocessing.get_context("forkserver")
try:
    _fork_ctx.set_forkserver_preload(["evaluator_specialist"])
except (ValueError, RuntimeError):
    pass

_current_program_path = None


def _nested_worker_init(program_path):
    # Runs in every forkserver-forked nested worker BEFORE it unpickles a
    # task, re-registering sys.modules["score_function"] so the
    # score_function object embedded in each task tuple resolves.
    try:
        if program_path:
            load_program(program_path)
    except Exception:
        pass


def Pool(processes):
    return _fork_ctx.Pool(processes=processes,
                          initializer=_nested_worker_init,
                          initargs=(_current_program_path,))


sys.modules.setdefault("evaluation_module", sys.modules[__name__])

import logging, traceback
from functools import partial
from skopt import Optimizer
from skopt.space import Real

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from utils.utils import numerical_mean_stable, geometric_mean_stable, get_scip_params, load_program, normalize
from global_core_manager import GlobalCoreManager

logger = logging.getLogger(__name__)

N_NODES_TRAIN = 750
AFFINITY_TRAIN = 4

# 4 losers (ratio>1) + 4 worst-decile non-losers (ratio<=1 but in the top
# decile), each split balanced for type AND absolute cost (one slow +
# one fast of each type per split) -- see specialist_training_set_n750.json
TRAIN_SEEDS = [900002448, 900002092, 900002335, 900002127]
VALID_SEEDS = [900002458, 900002118, 900002194, 900002366]


def make_ba_instance(seed):
    generator = ecole.instance.IndependentSetGenerator(
        n_nodes=N_NODES_TRAIN, affinity=AFFINITY_TRAIN,
        graph_type=ecole.instance.IndependentSetGenerator.GraphType.barabasi_albert)
    generator.seed(seed)
    return next(generator)


def execute_specialist(args):
    task, seed, cores_list, lock_dir, score_function, used_features, function_params, scip_params = args
    core_manager = GlobalCoreManager(cores_list, lock_dir)
    p = psutil.Process(current_process().pid)
    core_id = -1
    result = {"task": task, "seed": seed, "validity": False}
    try:
        core_id = core_manager.get_core()
        if score_function and function_params:
            score_function = partial(score_function, params=function_params)
        instance = make_ba_instance(seed)
        if hasattr(p, "cpu_affinity"):
            p.cpu_affinity([core_id])
        if task == "evolve":
            flash_khalil_obs = ecole.observation.Khalil2016Flash()
            flash_node_obs = ecole.observation.NodeBipartiteFlash()
            extract_features_khalil = [fea - 19 for fea in used_features if fea >= 19]
            extract_features_node = [fea for fea in used_features if fea < 19]
            flash_khalil_obs.set_active_features(extract_features_khalil)
            flash_node_obs.set_active_features(extract_features_node)
            env = ecole.environment.Branching(
                observation_function={"Node": flash_node_obs, "Khalil": flash_khalil_obs},
                scip_params=scip_params)
            env.seed(0)
            proc_time = time.process_time()
            observation, action_set, _, done, info = env.reset(instance)
            while not done:
                variable_features = np.concatenate(
                    (observation['Node'].variable_features[action_set, :],
                     observation['Khalil'].features[action_set, :]), axis=1)
                variable_features = normalize(variable_features, used_features)
                sb_scores = score_function(variable_features)
                action = action_set[sb_scores.argmax()]
                observation, action_set, _, done, info = env.step(action)
            result["time"] = time.process_time() - proc_time
        elif task == "relpscost":
            env = ecole.environment.Configuring(scip_params=scip_params)
            env.seed(0)
            proc_time = time.process_time()
            observation, action_set, _, done, info = env.reset(instance)
            _, _, _, _, info = env.step({})
            result["time"] = time.process_time() - proc_time

        scip_model = env.model.as_pyscipopt()
        result['validity'] = True
        result['nodes'] = scip_model.getNNodes()
        result['gap'] = abs(env.model.primal_bound - env.model.dual_bound) / (abs(min([env.model.primal_bound, env.model.dual_bound])) + 1e-6)
        return result
    except Exception as exc:
        error_details = traceback.format_exc()
        logger.debug(f"Generate an exception {error_details}")
        return {"task": task, "validity": False, "time": 0.0, "nodes": 0.0, "seed": seed,
                "error_info": f"Generate an exception {error_details}"}
    finally:
        if core_id != -1:
            core_manager.release_core(core_id)


def params_optimizer(seeds, cores_list, lock_dir, score_function, used_features, function_params, bound_params, scip_params, config):
    param_space = [Real(min_val, max_val) for min_val, max_val in bound_params]
    opt = Optimizer(dimensions=param_space, acq_func="gp_hedge", random_state=123)

    def objective_function_parallel(function_params, seeds, score_function, used_features, scip_params, pool):
        sub_tasks = [('evolve', seed, cores_list, lock_dir, score_function, used_features, function_params, scip_params) for seed in seeds]
        results = pool.map(execute_specialist, sub_tasks)
        all_result = {seed: {} for seed in seeds}
        for result in results:
            if not result["validity"]:
                logger.warning(f"A sub-task failed: {result.get('error_info')}")
                return 0.0
            all_result[result["seed"]]["evolve_time"] = result["time"]
            all_result[result["seed"]]["evolve_nodes"] = result["nodes"]
        nodes_list = [all_result[seed]["evolve_nodes"] for seed in seeds]
        return numerical_mean_stable(nodes_list)

    pool = Pool(processes=config.max_workers)
    try:
        n_calls = config.get("n_calls", 50)
        n_initial = config.get("n_initial", 10)
        x0 = opt.ask(n_initial)
        y0 = [objective_function_parallel(x, seeds, score_function, used_features, scip_params, pool) for x in x0]
        opt.tell(x0, y0)
        for i in range(n_initial, n_calls):
            next_x = opt.ask()
            f_val = objective_function_parallel(next_x, seeds, score_function, used_features, scip_params, pool)
            opt.tell(next_x, f_val)
    finally:
        pool.terminate()
        pool.join()
    result = opt.get_result()
    return list(result.x)


def evaluate(program_path, config):
    global _current_program_path
    _current_program_path = program_path
    pool = None
    final_result = {
        "validity": True, "avg_time": 0.0, "max_time": 0.0, "mean_time": 0.0,
        "avg_nodes": 0.0, "max_nodes": 0.0, "mean_nodes": 0.0,
        "avg_gap": 0.0, "max_gap": 0.0, "mean_gap": 0.0, "combined_score": 0.0,
    }
    try:
        train_seeds = TRAIN_SEEDS
        seeds = VALID_SEEDS
        score_function, used_features, function_params, bound_params = load_program(program_path)
        if config.target in ['time', 'nodes']:
            scip_params = get_scip_params(time_limit=300)
        elif config.target == 'gap':
            scip_params = get_scip_params(time_limit=60)
        if config.parameter_opt:
            function_params = params_optimizer(train_seeds, config.cores_list, config.lock_dir, score_function, used_features, function_params, bound_params, scip_params, config)
            final_result["params"] = function_params

        tasks = [('evolve', seed, config.cores_list, config.lock_dir, score_function, used_features, function_params, scip_params) for seed in seeds]
        if config.rpb:
            tasks += [('relpscost', seed, config.cores_list, config.lock_dir, None, None, None, {**scip_params, "branching/relpscost/priority": 9999999}) for seed in seeds]

        pool = Pool(processes=config.max_workers)
        async_result = pool.map_async(execute_specialist, tasks)
        results = async_result.get(timeout=config.stage2_timeout)

        all_result = {seed: {} for seed in seeds}
        for result in results:
            seed_index = result["seed"]
            if not result["validity"]:
                return {"validity": False, "combined_score": 0.0, "error_info": result["error_info"]}
            if result["task"] == 'evolve':
                all_result[seed_index]["evolve_time"] = result["time"]
                all_result[seed_index]["evolve_nodes"] = result["nodes"]
                all_result[seed_index]["evolve_gap"] = result["gap"]
            if result["task"] == 'relpscost':
                all_result[seed_index]["rpb_time"] = result["time"]
                all_result[seed_index]["rpb_nodes"] = result["nodes"]
                all_result[seed_index]["rpb_gap"] = result["gap"]

        evolve_time_list = [all_result[s]["evolve_time"] for s in seeds]
        evolve_nodes_list = [all_result[s]["evolve_nodes"] for s in seeds]
        evolve_gap_list = [all_result[s]["evolve_gap"] for s in seeds]
        final_result["avg_time"] = numerical_mean_stable(evolve_time_list)
        final_result["max_time"] = max(evolve_time_list)
        final_result["mean_time"] = geometric_mean_stable(evolve_time_list)
        final_result["avg_nodes"] = numerical_mean_stable(evolve_nodes_list)
        final_result["max_nodes"] = max(evolve_nodes_list)
        final_result["mean_nodes"] = geometric_mean_stable(evolve_nodes_list)
        final_result["avg_gap"] = numerical_mean_stable(evolve_gap_list)
        final_result["max_gap"] = max(evolve_gap_list)
        final_result["mean_gap"] = geometric_mean_stable(evolve_gap_list)

        if config.rpb:
            rpb_time_list = [all_result[s]["rpb_time"] for s in seeds]
            rpb_nodes_list = [all_result[s]["rpb_nodes"] for s in seeds]
            final_result["avg_rpb_time"] = numerical_mean_stable(rpb_time_list)
            final_result["max_rpb_time"] = max(rpb_time_list)
            final_result["avg_rpb_nodes"] = numerical_mean_stable(rpb_nodes_list)
            final_result["max_rpb_nodes"] = max(rpb_nodes_list)

        scaling_param = 100 if config.target == 'time' else 10000
        if config.rpb:
            scaling_param = final_result["max_rpb_time"] if config.target == 'time' else final_result["max_rpb_nodes"]
        if config.target == 'time':
            final_result["combined_score"] = float(np.log(scaling_param / final_result["avg_time"] + 1))
        elif config.target == 'nodes':
            final_result["combined_score"] = float(np.log(scaling_param / final_result["avg_nodes"] + 1))
        elif config.target == 'gap':
            final_result["combined_score"] = 100 / (1e-6 + float(final_result["avg_gap"]))

        return final_result

    except TimeoutError:
        return {"validity": False, "combined_score": 0.0,
                "error_info": f"Evaluation timed out after {config.stage2_timeout} seconds."}
    except Exception as exc:
        error_details = traceback.format_exc()
        return {"validity": False, "combined_score": 0.0,
                "error_info": f"Exception occurred: {type(exc).__name__}: {exc}"}
    finally:
        if pool is not None:
            pool.terminate()
            pool.join()


def evaluate_stage1(program_path, config):
    global _current_program_path
    _current_program_path = program_path
    pool = None
    final_result = {"stage1_score": 0.0, "error_info_stage1": None}
    try:
        seeds = TRAIN_SEEDS
        score_function, used_features, function_params, bound_params = load_program(program_path)
        if config.target in ['time', 'nodes']:
            scip_params = get_scip_params()
        elif config.target == 'gap':
            scip_params = get_scip_params(time_limit=180)
        tasks = []
        for seed in seeds:
            tasks.append(('evolve', seed, config.cores_list, config.lock_dir, score_function, used_features, function_params, scip_params))
            tasks.append(('relpscost', seed, config.cores_list, config.lock_dir, None, None, None, {**scip_params, "branching/relpscost/priority": 9999999}))

        pool = Pool(processes=config.max_workers)
        async_result = pool.map_async(execute_specialist, tasks)
        results = async_result.get(timeout=config.stage1_timeout)

        all_result = {seed: {} for seed in seeds}
        for result in results:
            seed_index = result["seed"]
            if not result["validity"]:
                return {"validity": False, "staga1_score": 0.0, "error_info_stage1": result["error_info"]}
            if result["task"] == 'evolve':
                all_result[seed_index]["evolve_time"] = result["time"]
                all_result[seed_index]["evolve_gap"] = result["gap"] + 1e-6
            if result["task"] == 'relpscost':
                all_result[seed_index]["rpb_time"] = result["time"]
                all_result[seed_index]["rpb_gap"] = result["gap"] + 1e-6

        speedup_list = [all_result[s]["rpb_time"] / all_result[s]["evolve_time"] for s in seeds] if config.target in ['time', 'nodes'] else \
            [all_result[s]["rpb_gap"] / all_result[s]["evolve_gap"] for s in seeds]
        final_result["stage1_score"] = min(speedup_list)
        return final_result

    except TimeoutError:
        return {"speedup_score": 0.0, "error_info": f"Evaluation timed out after {config.stage1_timeout} seconds."}
    except Exception as exc:
        error_details = traceback.format_exc()
        return {"speedup_score": 0.0, "error_info": f"Exception occurred: {type(exc).__name__}: {exc}"}
    finally:
        if pool is not None:
            pool.terminate()
            pool.join()


def evaluate_stage2(program_path, config):
    return evaluate(program_path, config)

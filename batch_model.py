#
#
# This file is dedicated to batch processing tools
#
#

from collections import OrderedDict
import numpy as np
from tqdm import tqdm
import evgt as eg
from concurrent import futures
import time
import copy
import os


def _make_kwargs_mesh(params_to_mesh, strict_params={}):
    # Convert input to ordered dict to manipulate further
    args_input = OrderedDict(params_to_mesh)
    args_space = {}

    # Unfold params values
    for key, item in args_input.items():
        args_space[key] = np.rint(np.linspace(*item))

    # Create meshgrid of params_to_mesh
    args_meshed = np.meshgrid(*args_space.values())
    args_mesh = list(
        map(lambda x: x.reshape(-1).astype(np.int32).tolist(), args_meshed))

    # Construct kwards_list
    kwargs_list = []
    for i in range(len(args_mesh[0])):
        kwargs = {}
        # Add meshed params to kwargs dict
        for idx, key in enumerate(args_input.keys()):
            kwargs[key] = args_mesh[idx][i]
        # Add constant params to kwargs dict
        kwargs.update(strict_params)
        kwargs_list.append(kwargs)

    return kwargs_list


def _fix_kwargs_scale(kwargs_list):
    fixed_kwargs_list = copy.deepcopy(kwargs_list)
    for kwargs in fixed_kwargs_list:
        kwargs["mutation_magnitude"] *= 0.01
        kwargs["dropout_rate"] *= 0.01

    return fixed_kwargs_list


def _evaluate_model(kwargs):
    model = eg.EvolutionaryEquilibrium(**kwargs)
    model.optimize()


def batch_eval(variable_params, constant_params, sample_size):
    print("Script started")

    print("Variable params:", variable_params)
    print("Constant params:", constant_params)
    print()

    # Make kwargs list
    kwargs_list = _make_kwargs_mesh(variable_params, constant_params)
    kwargs_list = _fix_kwargs_scale(kwargs_list)
    kwargs_list = np.random.permutation(kwargs_list).tolist()

    # Calculate total function evaluations needed
    total_func_evals = 0
    for kwargs in kwargs_list:
        total_func_evals += kwargs['generations'] \
            * kwargs['popsize'] * kwargs['npairs'] * kwargs['ngames'] \
            * sample_size

    # Initialize progress bar
    pb = tqdm(total=len(kwargs_list) * sample_size)
    pb.clear()

    # Log time
    arch_start = time.time()
    speed = 0.00002683
    est_total_time = total_func_evals * speed
    s = est_total_time

    print("MODELS generating MC experiment data calculation")
    print("Current timestamp: {}".format(arch_start))
    print("Current date and time: {}".format(
        time.strftime("%d %b %Y %H:%M:%S", time.localtime())))
    print("Total number of func evals to do: {}".format(total_func_evals))
    print("Total number of model runs: {}".format(
        len(kwargs_list) * sample_size))
    # print("Worst-case estimation of time to compute: {}".format(
    #     time.strftime("%H:%M:%S", time.gmtime(est_total_time))))
    print("Worst-case estimation of time to compute: {}:{}:{}".format(
        int(s // 3600), int(s % 3600 // 60), int(s % 60)))

    # Create concurent async execution handler
    CPUs = os.cpu_count()
    print("Start of multiprocess model running on {} CPUs".format(CPUs))

    pool = futures.ProcessPoolExecutor(max_workers=CPUs)
    fs = []
    # Unpack kwargs_list
    for kwargs in kwargs_list:
        # Hndle sample_size > 1
        for i in range(sample_size):
            # Create a copy of kwargs and change outfile_prefix to distinguish
            # samples
            tmp_kwargs = copy.copy(kwargs)
            tmp_kwargs["outfile_prefix"] = kwargs[
                "outfile_prefix"] + '-' + str(i)

            # Assign jobs to the async executor
            fs.append(pool.submit(_evaluate_model, tmp_kwargs))

    print("Jobs assigned")

    for ft in futures.as_completed(fs):
        pb.update(1)

    pool.shutdown()
    pb.close()
    arch_stop = time.time()

    print("MODELS finished calculations")
    print("Current timestamp: {}".format(arch_start))
    print("Average speed of funcevals: {} sec/funceval".format(
        (arch_stop - arch_start) / total_func_evals))
    print("Script finished successfully in {} seconds".format(
        int(arch_stop - arch_start)))


def main():
    # Sample Setup parameters
    sample_size = 1
    variable_params = {'popsize': (4, 80, 2), 'npairs': (1, 80, 2),
                       'ngames': (1, 30, 2), 'dropout_rate': (1, 50, 2),
                       'mutation_magnitude': (1, 40, 2)}
    constant_params = {'generations': 20, 'model': eg.Chicken(),
                       'outfolder': "testsample",
                       'outfile_prefix': 'ee'}
    # batch_eval(variable_params=variable_params,
    #            constant_params=constant_params,
    #            sample_size=sample_size)
    print("Direct usage is not supported! Please, load as module.")

if __name__ == '__main__':
    main()

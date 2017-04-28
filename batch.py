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


def make_kwargs_mesh(params_to_mesh, strict_params={}):
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


def evaluate_model(kwargs):
    model = eg.EvolutionaryEquilibrium(**kwargs)
    model.optimize()


def main():
    print("Script started")

    # Setup parameters
    sample_size = 2
    variable_params = {'popsize': (4, 400, 3), 'npairs': (1, 500, 1),
                       'ngames': (1, 200, 2), 'dropout_rate': (1, 50, 10),
                       'mutation_magnitude': (1, 40, 1)}
    constant_params = {'generations': 1, 'model': eg.Chicken(),
                       'outfolder': "test_testlog"}

    print("Variable params:", variable_params)
    print("Constant params:", constant_params)
    print()

    # Make kwargs list
    kwargs_list = make_kwargs_mesh(variable_params, constant_params)
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
    print("Current date and time: {}".format(time.strftime("%d %b %Y %H:%M:%S", time.localtime())))
    print("Total number of func evals to do: {}".format(total_func_evals))
    # print("Worst-case estimation of time to compute: {}".format(
    #     time.strftime("%H:%M:%S", time.gmtime(est_total_time))))
    print("Worst-case estimation of time to compute: {}:{}:{}".format(s//3600, s%3600//60, s%60))

    pool = futures.ProcessPoolExecutor(max_workers=16)
    fs = []
    for kwargs in kwargs_list:
        for i in range(sample_size):
            fs.append(pool.submit(evaluate_model, kwargs))

    for ft in futures.as_completed(fs):
        pb.update(1)

    pb.close()
    arch_stop = time.time()

    print("MODELS finished calculations")
    print("Current timestamp: {}".format(arch_start))
    print("Average speed of funcevals: {} sec/funceval".format((arch_stop -
                                                                arch_start) / total_func_evals))
    print("Script finished successfully in {} seconds".format(
        int(arch_stop - arch_start)))


if __name__ == '__main__':
    main()

import numpy as np
import pandas as pd
import os
import glob
from scipy.spatial import distance
import re
import copy
from functools import reduce
from tqdm import tqdm
from concurrent import futures


def _get_gen_logfile(metadata_file_path):
    # Get file name and dirname
    basenm = os.path.basename(metadata_file_path)
    basefolder = os.path.dirname(metadata_file_path)
    # Trunc _metadata.txt from string
    model = basenm[:-13]
    # Find corresponding csv file and return it
    log_gen = glob.glob(basefolder + '/gen/' + model + '*.csv')[0]
    return log_gen


def get_metadata_files_list(output_folder):
    # output_folder = 'production1_complete/Chicken'
    metadata_files = glob.glob(
        output_folder + '/' + '*metadata.txt', recursive=True)
    return metadata_files


def dataframes_from_model_run(metadata_file_path):
    df_md = _preprocess_metadata_file(metadata_file_path)
    if df_md["log_generations"]:
        df_gen = pd.read_csv(_get_gen_logfile(metadata_file_path))
    return df_md, df_gen


def _preprocess_metadata_file(md_file):
    """Read metadata from json file, add 'strategies' set
    and 'filename' entries to metadata and return updated metadata."""

    df_md = pd.read_json(md_file, typ='series')
    # Append filename to metadata
    df_md = df_md.append(
        pd.Series([os.path.basename(md_file)[:-13]], index=['filename']))

    # Parse log_gen_columns for STRATEGIES set
    regex = 'STRAT-[0-9]+'
    strategies_set = set()
    for column in df_md["log_gen_columns"]:
        # print(column)
        match = re.match(regex, column)

        # If there is no match pass and go to next column
        if match is None:
            continue

        strategies_set.add(match.group(0))

    # Convert strategies set to ordered list
    strategies = list(strategies_set)
    strategies.sort()
    # Append strategies list to metadata series
    df_md = df_md.append(pd.Series([strategies], index=['strategies']))

    return df_md


def _calculate_convergence_dds(fitness, strategy):
    # As an input it takes Series of fitness
    # And corresponing strategies on Discrete Decision Set

    indexes = list(fitness.groups.keys())
    # popsize = len(fitness.get_group(0))

    cosine_diff = pd.Series(index=indexes)
    purity_diff = pd.Series(index=indexes)
    convergence = pd.Series(index=indexes)

    for i in indexes[1:]:
        # Calculate number of species in current generation
        # that have better fitness than the best specie in the last generation
        #         last_top_fitness = fitness.get_group(i-1).iloc[0]
        #         current_better = fitness.get_group(i).loc[fitness.get_group(i) > last_top_fitness]
        #         dominance.iloc[i] = len(current_better) / popsize

        # Get strategy of current and previous generations best specie
        last_top = strategy.get_group(i - 1).iloc[0]
        current_top = strategy.get_group(i).iloc[0]

        # Calculate cosine difference between current and last best specie
        cos_dif = distance.cosine(last_top, current_top)
        cosine_diff.iloc[i] = cos_dif

        # Calculate mutual difference of strategies from pure strategy vector
        # If both vectors represent pure strategies but they are different,
        # purity difference would be equal to 1
        pur_dif = 1 - np.sum(current_top * last_top)
        purity_diff.iloc[i] = pur_dif

        # print("Generation {}: cosine_diff {:.5f}; purity_diff {:.5f}; convergence {:.5f}".format(
        #       i, cos_dif, pur_dif, ((1-cos_dif)*(1-pur_dif))))
        # print("Shape:", distance.cdist(last_top, current_top, metric='braycurtis').shape)
        # spatial_diff.iloc[i] = cos_dif

        # print("Generation {} last_top:".format(i), last_top_fitness)
        # print("Cosine difference between best:", cos_dif)
        # print(current_better[:])

    # Calculate convergence
    convergence = (1 - cosine_diff) * (1 - purity_diff)
    return convergence


def calculate_convergence(log_metadata_series,
                          log_gen_dataframe, return_dictionary=False):
    """Calculate convergence metrics.

    'return_dictionary': boolean, optional
        if False then only overall convergence metric would be returned
        if Ture a dictionary of convergence metrics for every strategy
            and overall model run would be returned

    """

    # Group log_gen_dataframe by generation
    grouped_gen_dataframe = log_gen_dataframe.groupby("Generation")

    # Iterate through strategies and calculate corresponding convergence
    # metrics
    convergence = {}

    for strat in log_metadata_series["strategies"]:
        strat_vector_columns = list(filter(lambda x: x.startswith(
            strat + '-'), log_metadata_series["log_gen_columns"]))
        strat_fitness_column = list(filter(lambda x: x.startswith(
            "FITNESS_" + strat + '-'), log_metadata_series["log_gen_columns"]))
        # print(strat_vector_columns, strat_fitness_column)

        strat_convergence = _calculate_convergence_dds(
            grouped_gen_dataframe[strat_fitness_column],
            grouped_gen_dataframe[strat_vector_columns])

        convergence[strat] = strat_convergence.rename(
            strat + ' convergence')
        # print("Strategy {} convergence:".format(strat), strat_convergence)

    # Calculate overall convergence as a product of all convergence metrics
    convergence['overall'] = reduce(
        lambda x, y: x * y, convergence.values()).rename("overall convergence")

    if return_dictionary:
        output = convergence
    else:
        output = convergence['overall']

    return output


def get_metadata_with_equilibrium(log_metadata_series,
                                  log_gen_dataframe, overall_convergence):
    """Return metadata series with attributes added corresponding to
    equilibrium found details.

    Atrributes added:
    -----------------
    'converged': boolean
        Indicate whether model successfully converged
    'id_gen_converged' : int
        Number of generation when model converged
    'equilibrium' : list
        List of strategies of best species in converged geenration

    """

    # Copy metadata dataframe to negate side effects
    md_out = copy.deepcopy(log_metadata_series)

    # Find generations when model first converged and add this data to metadata
    converged_generations = overall_convergence[overall_convergence == 1]
    if len(converged_generations) == 0:
        md_out = md_out.append(pd.Series([False], index=["converged"]))
        md_out = md_out.append(pd.Series([0], index=["id_gen_converged"]))
        md_out = md_out.append(pd.Series([md_out["generations"]
                                          * md_out["popsize"]
                                          * md_out["npairs"]
                                          * md_out["ngames"]],
                                         index=["funcevals_to_converge"]))
    else:
        md_out = md_out.append(pd.Series([True], index=["converged"]))
        md_out = md_out.append(
            pd.Series([converged_generations.idxmin()],
                      index=["id_gen_converged"]))
        md_out = md_out.append(pd.Series([md_out["id_gen_converged"]
                                          * md_out["popsize"]
                                          * md_out["npairs"]
                                          * md_out["ngames"]],
                                         index=["funcevals_to_converge"]))

    # Derive equilibrium found in model run
    equilibrium_strategies = list()
    grouped_gen_dataframe = log_gen_dataframe.groupby("Generation")

    for strat in md_out["strategies"]:
        # Get names of columns in generations log file
        # corresponding to current strategy being iterated at the moment
        strat_vector_columns = list(filter(lambda x: x.startswith(
            strat + '-'), log_metadata_series["log_gen_columns"]))
        # Get the most fit specie out of first converged generation
        # representing strategy being iterated at the moment
        equilibrium_strat = grouped_gen_dataframe[
            strat_vector_columns].get_group(
                md_out["id_gen_converged"]).iloc[0]
        # Append found equilibrium strategy to the list of equilibrium
        # strategies
        equilibrium_strategies.append(equilibrium_strat)

    # If model not converged set equilibrium sttrategies to zero vectors
    if not md_out["converged"]:
        for idx in range(len(equilibrium_strategies)):
            equilibrium_strategies[idx] = equilibrium_strategies[idx] * 0

    # Append found equilibrium to metadata
    md_out = md_out.append(
        pd.Series([equilibrium_strategies], index=["equilibrium"]))

    return md_out


def process_model_run(metadata_file_path):
    """Process model run given by metadata file.

    Return
    ------
    'df_metadata' : pd.Series
    'df_generations' : pd.DataFrame
    'convergence' : dictionary

    """

    df_metadata, df_generations = dataframes_from_model_run(metadata_file_path)
    convergence = calculate_convergence(
        log_metadata_series=df_metadata,
        log_gen_dataframe=df_generations,
        return_dictionary=True)
    df_metadata = get_metadata_with_equilibrium(
        log_metadata_series=df_metadata,
        log_gen_dataframe=df_generations,
        overall_convergence=convergence['overall'])

    return df_metadata, df_generations, convergence


def _check_equilibrium_validity(metadata_wequilibrium_series,
                                legible_equilibria):
    """Check if found equilibrium is in the given set of equilibria.

    Inputs
    ------
    'metadata_wequilibrium_series' : list
        List of equilibrium strategies
        as given in metadata 'equilibrium' attribute
    'legible_equilibria' : list
        List of equilibria to compare to,
        given as list of flattened lists of floats representing equilibria

    """

    # Flattening metadata_wequilibrium_series list
    tmp_eq = []
    for strat in metadata_wequilibrium_series['equilibrium']:
        tmp_eq += strat.tolist()

    output = tmp_eq in legible_equilibria
    return output


def check_metadata_list(metadata_wequilibrium_list, legible_equilibria):
    """Return metadata series list with
    added boolean attribute ''legible_equilibria_found''
    indicating whether found equilibrium is in 'legible_equilibria' list.

    Inputs
    ------
    'metadata_wequilibrium_list' : list
        List of metadata series with information about found equilibria
    'legible_equilibria' : list
        List of equilibria to compare to,
        given as list of flattened lists of floats representing equilibria

    """

    output = []
    for md in metadata_wequilibrium_list:
        validity = False
        if _check_equilibrium_validity(md, legible_equilibria):
            validity = True
        output.append(
            md.append(pd.Series([validity], index=["legible_equilibria_found"])))

    return output


def error_proof_process_model(model_run):
    md, gen, conv = None, None, None
    try:
        md, gen, conv = process_model_run(model_run)
    except Exception as exc:
        print("Failed to process model run: {}\nError: {}".format(
            model_run, exc))
    return md, gen, conv


def batch_process_model_runs(metadata_files):
    """Batch process model runs.

    Process metadata files, load generations statistics,
    calculate corresponding convergence metrics.

    Input
    -----
    'metadata_files' : list
        list of metadata files of model runs

    Output
    ------
    'metadata' : list of pd.Series
        list of processed metadata with found equilibrium data
    'generations' : list of pd.DataFrame
        list of corresponding generations
        statistics (index aligned with 'metadata')
    'convergence' : list of dictionary
        list of convergence metrics
        for a model run (index aligned with 'metadata')

    """

    metadata = []
    generations = []
    convergence = []
    pb = tqdm(total=len(metadata_files))
    pb.clear()

    processes = os.cpu_count() * 4
    pool = futures.ProcessPoolExecutor(max_workers=processes)
    fs = []

    for model_run in metadata_files:
        fs.append(pool.submit(error_proof_process_model, model_run))

    for task in futures.as_completed(fs):
        md, gen, conv = task.result()
        if md is not None:
            metadata.append(md)
            generations.append(gen)
            convergence.append(conv)
        pb.update(1)

    pool.shutdown()
    pb.close()

    return metadata, generations, convergence

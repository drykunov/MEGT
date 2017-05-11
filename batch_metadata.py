import argparse
import os
import pickle
import time
import graphs_data as gd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'output_folder',
        help="path to the folder, containing metadata from model runs")
    parser.add_argument(
        'pickel_destination',
        help="path to the file, where to pickle processed data")
    parser.add_argument(
        '-p', '--processes', help="number of parallel processes to use",
        type=int, default=None)
    args = parser.parse_args()

    arch_start = time.time()

    print("Started batch processing of metadata files")
    print("Current wrkdir: {}".format(os.getcwd()))
    print("Current timestamp: {}".format(arch_start))
    print("Current date and time: {}".format(
        time.strftime("%d %b %Y %H:%M:%S", time.localtime())))

    print()
    print("Reading metadata from: {}".format(args.output_folder))

    # Create output folder if needed
    if not os.path.exists(os.path.dirname(args.pickel_destination)):
        os.makedirs(args.pickel_destination)
    print("Pickled processed data would be written to: {}".format(
        args.pickel_destination))
    print()

    md_files = gd.get_metadata_files_list(args.output_folder)
    print("Found {} metadata files".format(len(md_files)))
    output = gd.batch_process_model_runs(md_files,
                                         parallel_processes=args.processes)
    print("Successfully processed {} model runs".format(len(output[0])))

    with open(args.pickel_destination, 'wb') as f:
        pickle.dump(output, f, pickle.HIGHEST_PROTOCOL)

    print("Data successfully pickled to: {}".format(args.pickel_destination))


if __name__ == '__main__':
    main()

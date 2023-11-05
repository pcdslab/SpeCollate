import argparse
import time
from os.path import dirname, join

import torch
import torch.multiprocessing as mp
from tqdm import tqdm

from src.atlesconfig import config
from src.atlespredict import ooc_filtered_search as fs
from src.atlespredict import ooc_unfiltered_search as ufs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Adding optional argument
    parser.add_argument("-c", "--config", help="Path to the config file.")
    parser.add_argument("-p", "--preprocess", help="Preprocess data?", default="True")
    parser.add_argument("-f", "--filter", help="Filter value to use.", choices=["all", "none", "len-1", "no-len"])
    parser.add_argument("-d", "--pep_dir", help="Path to the peptide directory.")

    # Read arguments from command line
    input_params = parser.parse_args()
    args_dict = vars(input_params)

    if input_params.config:
        tqdm.write("config: " + input_params.config)
    config_path = input_params.config if input_params.config else join((dirname(__file__)), "config.ini")
    config.init_config(config_path)

    num_gpus = torch.cuda.device_count()
    print("Num GPUs: {}".format(num_gpus))
    start_time = time.time()

    spectra_path = config.get_config(key="prep_path", section="search")
    print("Spectra path: {}".format(spectra_path))

    length_filter = config.get_config(key="length_filter", section="filter")
    missed_cleavages_filter = config.get_config(key="missed_cleavages_filter", section="filter")
    modification_filter = config.get_config(key="modification_filter", section="filter")

    if "filter" in args_dict:
        if args_dict["filter"] == "none":
            print("No filters enabled. Using unfiltered search.")
            run_atles_search = ufs.run_atles_search
        else:
            print("Filters enabled. Using filtered search.")
            run_atles_search = fs.run_atles_search
    else:
        if not length_filter and not missed_cleavages_filter and not modification_filter:
            print("No filters enabled. Using unfiltered search.")
            run_atles_search = ufs.run_atles_search
        else:
            print("Filters enabled. Using filtered search.")
            run_atles_search = fs.run_atles_search

    mp.spawn(run_atles_search, args=(2, config_path, args_dict), nprocs=2, join=True)
    # run_atles_search(0, 1)
    print("Total time: {}".format(time.time() - start_time))

    # if all filters disabled, call a different function

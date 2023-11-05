import argparse
import bisect
import os
import pickle
import time
from collections import defaultdict
from dataclasses import dataclass
from os.path import dirname, join
from pathlib import PurePath

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from tqdm import tqdm

from src.atlesconfig import config, arg_parse
from src.atlespredict import dbsearch, pepdataset, postprocess, specdataset, specollate_model
from src.atlesutils import utils


@dataclass
class PepInfo:
    pep_list: list
    prot_list: list
    pep_mass_list: list


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(config.get_config(key="master_port", section="input"))
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", world_size=world_size, rank=rank)


def create_directories(index_path):
    os.makedirs(index_path)
    os.mkdir(join(index_path, "peptide_chunks"))
    os.mkdir(join(index_path, "peptide_embeddings"))
    os.mkdir(join(index_path, "decoy_embeddings"))


# check if preprocessed folder exisits.
# if not do step: 1 - 2
# do step 3 - 4
# Step 1
def write_peptides_to_chunk(pep_dataset, ooc_chunk_size, chunk_path, pep_counter, min_pep_len, max_pep_len, max_clvs):
    write_counter = 0
    with open(chunk_path, "w") as f:
        while write_counter < ooc_chunk_size:
            if pep_counter >= len(pep_dataset):
                break
            pep = pep_dataset.pep_list[pep_counter]
            clv = pep_dataset.missed_cleavs[pep_counter]
            prot = pep_dataset.prot_list[pep_counter]
            pep_counter += 1
            pep_len = sum(map(str.isupper, pep))
            if min_pep_len <= pep_len <= max_pep_len and 0 <= clv <= max_clvs:
                f.write(">" + prot + "\n")
                f.write(pep + "\n")
                write_counter += 1
    return pep_counter


def save_file_names(file_names, index_path):
    with open(join(index_path, "file_names.pkl"), "wb") as f:
        pickle.dump(file_names, f)


def chunkify_peptides(index_path):
    pep_dir = config.get_config(key="pep_dir", section="search")
    ooc_chunk_size = config.get_config(key="chunk_size", section="ooc")
    min_pep_len = config.get_config(key="min_pep_len", section="ml")
    max_pep_len = config.get_config(key="max_pep_len", section="ml")
    max_clvs = config.get_config(key="max_clvs", section="ml")

    pep_dataset = pepdataset.PeptideDataset(pep_dir, decoy=False)
    pep_chunks_path = join(index_path, "peptide_chunks")

    file_names = []
    pep_counter = chunk_counter = 0
    print("Chunkify peptides and writing to files")
    while True:
        min_mass = pep_dataset.pep_mass_list[pep_counter]
        max_mass = (
            pep_dataset.pep_mass_list[pep_counter + ooc_chunk_size]
            if pep_counter + ooc_chunk_size < len(pep_dataset)
            else pep_dataset.pep_mass_list[-1]
        )
        file_name = "chunk-{}-{}-{}".format(chunk_counter, min_mass, max_mass)
        file_path = join(pep_chunks_path, file_name)
        file_names.append(file_name)

        pep_counter = write_peptides_to_chunk(
            pep_dataset, ooc_chunk_size, file_path, pep_counter, min_pep_len, max_pep_len, max_clvs
        )

        chunk_counter += 1
        if pep_counter >= len(pep_dataset):
            break

    save_file_names(file_names, index_path)

    return file_names


def get_snap_model(rank):
    model_name = config.get_config(key="model_name", section="search")
    print("Using model: {}".format(model_name))
    snap_model = specollate_model.Net(vocab_size=30, embedding_dim=512, hidden_lstm_dim=512, lstm_layers=2).to(rank)
    snap_model = nn.parallel.DistributedDataParallel(snap_model, device_ids=[rank])
    # snap_model.load_state_dict(torch.load('models/32-embed-2-lstm-SnapLoss2-noch-3k-1k-152.pt')['model_state_dict'])
    # below one has 26975 identified peptides.
    # snap_model.load_state_dict(
    #     torch.load("models/512-embed-2-lstm-SnapLoss-noch-80k-nist-massive-52.pt")["model_state_dict"]
    # )
    # below one has 27.5k peps
    # snap_model.load_state_dict(
    #     torch.load("models/hcd/512-embed-2-lstm-SnapLoss2D-inputCharge-80k-nist-massive-116.pt")["model_state_dict"]
    # )
    snap_model.load_state_dict(torch.load("specollate-model/{}".format(model_name))["model_state_dict"])
    snap_model = snap_model.module
    snap_model.eval()
    print(snap_model)
    return snap_model


# 2 - load each chunk and process it using specollate, save embeddings for each chunk separately
def process_peptide_chunks(rank, snap_model, file_names, index_path):
    pep_dir = config.get_config(key="pep_dir", section="search")
    pep_batch_size = config.get_config(key="pep_batch_size", section="search")
    embedding_type = "peptide_embeddings" if rank == 0 else "decoy_embeddings"

    for file_name in file_names:
        pep_chunks_path = join(index_path, "peptide_chunks")
        pep_file_path = join(pep_chunks_path, file_name)
        if os.path.exists(pep_file_path):
            if not os.path.getsize(pep_file_path):
                os.remove(pep_file_path)
                continue
            print("Processing file: {}".format(pep_file_path))
            # process peptides
            pep_dataset = pepdataset.PeptideDataset(pep_dir, pep_file_path, decoy=rank == 1)
            pep_loader = torch.utils.data.DataLoader(
                dataset=pep_dataset,
                batch_size=pep_batch_size,
                collate_fn=dbsearch.pep_collate,
            )

            print("Processing {}...".format("Peptides" if rank == 0 else "Decoys"))
            e_peps = dbsearch.runSpeCollateModel(pep_loader, snap_model, "peps", rank)
            print("Peptides done!")

            # save embeddings
            embedding_path = join(index_path, embedding_type, file_name)
            print("Saving embeddings at {}".format(embedding_path))
            torch.save(e_peps, embedding_path)
            print("Done \n")

    dist.barrier()


def process_spectra(rank, snap_model):
    prep_path = config.get_config(section="search", key="prep_path")
    print("Processing Spectra: {}".format(prep_path))
    spec_batch_size = config.get_config(key="spec_batch_size", section="search")
    spec_dataset = specdataset.SpectraDataset(join(prep_path, "specs.pkl"))
    spec_loader = torch.utils.data.DataLoader(
        dataset=spec_dataset,
        batch_size=spec_batch_size,
        collate_fn=dbsearch.spec_collate,
    )

    print("Processing spectra...")
    e_specs = dbsearch.runSpeCollateModel(spec_loader, snap_model, "specs", rank)
    print("Spectra done!")
    return e_specs, spec_dataset.masses, spec_dataset.charges


# 3 - Loop over spectra classes, load embeddings for peptides, peform db search
# def create_spectra_dict(lens, cleavs, mods, e_specs, spec_masses, file_names):
def chunkify_spectra(e_specs, spec_masses, file_names):
    print("Creating spectra chunk dictionary.")
    tol = config.get_config(key="precursor_tolerance", section="search")
    tol_type = config.get_config(key="precursor_tolerance_type", section="search")
    if tol_type == "ppm":
        min_mass = max(spec_masses[0] - utils.ppm(spec_masses[0], tol), 0)
        max_mass = spec_masses[-1] + utils.ppm(spec_masses[-1], tol)
    else:
        min_mass = max(spec_masses[0] - tol, 0)
        max_mass = spec_masses[-1] + tol
    spec_chunk_dict = defaultdict(list)
    spec_min_idx = spec_max_idx = 0
    for file_name in file_names:
        if tol_type == "ppm":
            min_mass = float(file_name.split("-")[-2]) * (1000000.0 / (1000000.0 - float(tol)))
            max_mass = float(file_name.split("-")[-1]) * (1000000.0 / (1000000.0 + float(tol)))
        else:
            min_mass = float(file_name.split("-")[-2]) + float(tol)
            max_mass = float(file_name.split("-")[-1]) - float(tol)
        spec_min_idx = bisect.bisect_right(spec_masses, min_mass, lo=spec_min_idx)
        spec_max_idx = bisect.bisect_left(spec_masses, max_mass, lo=spec_max_idx)
        for idx in range(spec_min_idx, spec_max_idx + 1):
            spec_chunk_dict[file_name].append([idx, e_specs[idx], spec_masses[idx]])

    return spec_chunk_dict


def search_database(rank, spec_filt_dict, spec_charges, index_path, out_pin_dir):
    search_spec_batch_size = config.get_config(key="search_spec_batch_size", section="search")
    # dist.barrier()
    # Run database search for each dict item
    unfiltered_time = 0

    cum = 0
    print("Running unfiltered {} database search.".format("target" if rank == 0 else "decoy"))
    for file_name in spec_filt_dict:
        print("Searching for key {}.".format(file_name))
        print("Chunk: {}".format(file_name))
        pep_chunks_path = join(index_path, "peptide_chunks")
        pep_file_path = join(pep_chunks_path, file_name)
        if not os.path.exists(pep_file_path):
            print("File {} not found. Skipping.".format(pep_file_path))
            continue
        # Load peptides
        pep_dataset = pepdataset.PeptideDataset(pep_chunks_path, file_name, decoy=rank == 1)
        # Load embeddings
        pep_embeddings_path = join(index_path, "peptide_embeddings" if rank == 0 else "decoy_embeddings")
        pep_embeddings_file_path = join(pep_embeddings_path, file_name)
        e_peps = torch.load(pep_embeddings_file_path)
        pep_data = [[idx, e_pep, mass] for idx, (e_pep, mass) in enumerate(zip(e_peps, pep_dataset.pep_mass_list))]
        cum += len(pep_data)
        spec_subset = spec_filt_dict[file_name]
        search_loader = torch.utils.data.DataLoader(
            dataset=spec_subset,
            num_workers=0,
            batch_size=search_spec_batch_size,
            shuffle=False,
        )
        # pep_dataset = None
        unfiltered_start_time = time.time()
        spec_inds, pep_inds, psm_vals = dbsearch.filtered_parallel_search(search_loader, pep_data, rank)
        unfiltered_time += time.time() - unfiltered_start_time

        if not spec_inds:
            continue

        print("{} PSMS: {}".format("Target" if rank == 0 else "Decoy", len(pep_inds)))

        # 4 - Write PSMs to pin file
        postprocess.write_to_pin(rank, pep_inds, psm_vals, spec_inds, pep_dataset, spec_charges, out_pin_dir)


def run_atles_search(rank, world_size, config_path, args_dict):
    setup(rank, world_size)
    config.init_config(config_path)
    arg_parse.process_args_dict(args_dict)
    pep_dir = config.get_config(key="pep_dir", section="search")
    pep_index_name = PurePath(pep_dir).name
    index_path = join(config.get_config(key="index_path", section="search"), pep_index_name, "unfiltered")

    spectra = PurePath(config.get_config(key="prep_path", section="search")).name
    out_pin_dir = join(os.getcwd(), "percolator", pep_index_name + "-unfiltered-" + spectra)

    print("Running unfiltered ooc search on {}.".format(pep_index_name))
    with torch.no_grad():
        snap_model = get_snap_model(rank)
        dist.barrier()
        file_names = None
        if not os.path.exists(index_path):
            dist.barrier()
            if rank == 0:
                create_directories(index_path)
                chunkify_peptides(index_path)
            dist.barrier()
            with open(join(index_path, "file_names.pkl"), "rb") as f:
                file_names = pickle.load(f)
            dist.barrier()
            process_peptide_chunks(rank, snap_model, file_names, index_path)
        dist.barrier()
        if not file_names:
            with open(join(index_path, "file_names.pkl"), "rb") as f:
                file_names = pickle.load(f)
        dist.barrier()

        t_time = time.time()
        e_specs, spec_masses, spec_charges = process_spectra(rank, snap_model)
        spec_filt_dict = chunkify_spectra(e_specs, spec_masses, file_names)
        l_time = time.time()
        dist.barrier()
        search_database(rank, spec_filt_dict, spec_charges, index_path, out_pin_dir)
        print("Search time: {}".format(time.time() - l_time))
        postprocess.post_process_pin_files(rank, out_pin_dir)
        print("Total time: {}".format(time.time() - t_time))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Adding optional argument
    parser.add_argument("-c", "--config", help="Path to the config file.")
    parser.add_argument("-p", "--preprocess", help="Preprocess data?", default="True")

    # Read arguments from command line
    input_params = parser.parse_args()

    if input_params.config:
        tqdm.write("config: %s" % input_params.path)
    config.param_path = input_params.config if input_params.config else join((dirname(__file__)), "config.ini")

    num_gpus = torch.cuda.device_count()
    print("Num GPUs: {}".format(num_gpus))
    start_time = time.time()
    mp.spawn(run_atles_search, args=(2,), nprocs=2, join=True)
    # run_atles_search(0, 1)
    print("Total time: {}".format(time.time() - start_time))

    # if all filters disabled, call a different function

import argparse
import itertools
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
from src.atlestrain import model


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


# check if preprocessed folder exisits.
# if not do step: 1 - 2
# do step 3 - 4
# Step 1
def classify_peptides(index_path):
    pep_dir = config.get_config(key="pep_dir", section="search")
    min_pep_len = config.get_config(key="min_pep_len", section="ml")
    max_pep_len = config.get_config(key="max_pep_len", section="ml")
    max_clvs = config.get_config(key="max_clvs", section="ml")

    pep_dataset = pepdataset.PeptideDataset(pep_dir, decoy=False)
    pep_classes_path = join(index_path, "peptide_classes")
    os.makedirs(index_path)

    os.mkdir(pep_classes_path)
    os.mkdir(join(index_path, "peptide_embeddings"))
    os.mkdir(join(index_path, "decoy_embeddings"))

    # 1 - classify peptides and write to 144 separate files.
    print("Opening files")
    open_files = {}
    class_offsets = {}
    for length, clv, mod in itertools.product(range(min_pep_len, max_pep_len + 1), range(max_clvs + 1), range(2)):
        file_name = "{}-{}-{}".format(length, clv, mod)
        open_files[file_name] = open(join(pep_classes_path, file_name), "a")
        class_offsets[file_name] = 0

    print("Classifying peptides and writing to files")
    for pep, clv, mod, prot in zip(
        pep_dataset.pep_list,
        pep_dataset.missed_cleavs,
        pep_dataset.pep_modified_list,
        pep_dataset.prot_list,
    ):
        pep_len = sum(map(str.isupper, pep))
        if min_pep_len <= pep_len <= max_pep_len and 0 <= clv <= max_clvs:
            file_name = "{}-{}-{}".format(int(pep_len), int(clv), int(mod))
            if file_name in open_files:
                f = open_files[file_name]
                f.write(">" + prot + "\n")
                f.write(pep + "\n")
                class_offsets[file_name] += 1

    print("Closing files")
    for _, f in open_files.items():
        f.close()

    cum = 0
    for length, clv, mod in itertools.product(range(min_pep_len, max_pep_len + 1), range(max_clvs + 1), range(2)):
        file_name = "{}-{}-{}".format(length, clv, mod)
        offset = class_offsets[file_name]
        class_offsets[file_name] = cum
        cum += offset

    # need to get decoy offset later
    with open(join(index_path, "{}".format("peptide_class_offsets.pkl")), "wb") as f:
        pickle.dump(class_offsets, f)


def get_snap_model(rank):
    model_name = config.get_config(key="model_name", section="search")
    print("Using model: {}".format(model_name))
    snap_model = specollate_model.Net(vocab_size=30, embedding_dim=512, hidden_lstm_dim=512, lstm_layers=2).to(rank)
    snap_model = nn.parallel.DistributedDataParallel(snap_model, device_ids=[rank])
    # snap_model.load_state_dict(torch.load("models/32-embed-2-lstm-SnapLoss2-noch-3k-1k-152.pt")["model_state_dict"])
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


# 2 - load each class and process it using specollate, save embeddings for each class separately
def process_peptides(rank, snap_model, index_path):
    pep_dir = config.get_config(key="pep_dir", section="search")
    pep_batch_size = config.get_config(key="pep_batch_size", section="search")
    min_pep_len = config.get_config(key="min_pep_len", section="ml")
    max_pep_len = config.get_config(key="max_pep_len", section="ml")
    max_clvs = config.get_config(key="max_clvs", section="ml")
    class_offsets = {}
    cum = 0
    for length, clv, mod in itertools.product(range(min_pep_len, max_pep_len + 1), range(max_clvs + 1), range(2)):
        file_name = "{}-{}-{}".format(length, clv, mod)
        class_offsets[file_name] = cum
        pep_classes_path = join(index_path, "peptide_classes")
        pep_file_path = join(pep_classes_path, file_name)
        if os.path.exists(pep_file_path):
            if not os.path.getsize(pep_file_path):
                os.remove(pep_file_path)
                continue
            print("Processing file: {}".format(file_name))
            # process peptides
            pep_dataset = pepdataset.PeptideDataset(pep_dir, pep_file_path, decoy=rank == 1)
            cum += len(pep_dataset)
            pep_loader = torch.utils.data.DataLoader(
                dataset=pep_dataset,
                batch_size=pep_batch_size,
                collate_fn=dbsearch.pep_collate,
            )

            print("Processing {}...".format("Peptides" if rank == 0 else "Decoys"))
            e_peps = dbsearch.runSpeCollateModel(pep_loader, snap_model, "peps", rank)
            print("Peptides done!")

            # save embeddings
            embedding_type = "peptide_embeddings" if rank == 0 else "decoy_embeddings"
            embedding_path = join(index_path, embedding_type, file_name)
            print("Saving embeddings at {}".format(embedding_path))
            torch.save(e_peps, embedding_path)
            print("Done \n")

    offset_file_name = "peptide_class_offsets.pkl" if rank == 0 else "decoy_class_offsets.pkl"
    with open(join(index_path, "{}".format(offset_file_name)), "wb") as f:
        pickle.dump(class_offsets, f)


def run_atles(rank, spec_loader):
    model_ = model.Net().to(rank)
    model_ = nn.parallel.DistributedDataParallel(model_, device_ids=[rank])
    # model_.load_state_dict(
    #     torch.load("atles-out/16403437/models/pt-mass-ch-16403437-1toz70vi-472.pt")["model_state_dict"]
    # )
    # model_.load_state_dict(torch.load(
    #     '/lclhome/mtari008/DeepAtles/atles-out/123/models/pt-mass-ch-123-2zgb2ei9-385.pt'
    #     )['model_state_dict'])
    model_.load_state_dict(
        torch.load(
            "/lclhome/mtari008/DeepAtles/atles-out/1382/models/nist-massive-deepnovo-mass-ch-1382-c8mlqbq7-157.pt"
        )["model_state_dict"]
    )
    model_ = model_.module
    model_.eval()
    print(model_)

    lens, cleavs, mods = dbsearch.runAtlesModel(spec_loader, model_, rank)
    pred_cleavs_softmax = torch.log_softmax(cleavs, dim=1)
    _, pred_cleavs = torch.max(pred_cleavs_softmax, dim=1)
    pred_mods_softmax = torch.log_softmax(mods, dim=1)
    _, pred_mods = torch.max(pred_mods_softmax, dim=1)

    return (
        torch.round(lens).type(torch.IntTensor).squeeze().tolist(),
        pred_cleavs.squeeze().tolist(),
        pred_mods.squeeze().tolist(),
    )


def process_spectra(rank, snap_model):
    prep_path = config.get_config(section="search", key="prep_path")
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

    atles_start_time = time.time()
    lens, cleavs, mods = run_atles(rank, spec_loader)
    atles_end_time = time.time()
    atles_time = atles_end_time - atles_start_time
    print("Atles time: {}".format(atles_time))
    return e_specs, lens, cleavs, mods, spec_dataset.masses, spec_dataset.charges


# 3 - Loop over spectra classes, load embeddings for peptides, peform db search
def create_spectra_dict(lens, cleavs, mods, e_specs, spec_masses):
    min_pep_len = config.get_config(key="min_pep_len", section="ml")
    max_pep_len = config.get_config(key="max_pep_len", section="ml")
    max_clvs = config.get_config(key="max_clvs", section="ml")
    print("Creating spectra filtered dictionary.")
    spec_filt_dict = defaultdict(list)
    for idx, (l, clv, mod) in enumerate(zip(lens, cleavs, mods)):
        if min_pep_len <= l <= max_pep_len and 0 <= clv <= max_clvs:
            key = "{}-{}-{}".format(int(l), int(clv), int(mod))
            spec_filt_dict[key].append([idx, e_specs[idx], spec_masses[idx]])

    return spec_filt_dict


def search_database(rank, spec_filt_dict, spec_charges, index_path, out_pin_dir):
    pep_dir = config.get_config(key="pep_dir", section="search")
    search_spec_batch_size = config.get_config(key="search_spec_batch_size", section="search")
    min_pep_len = config.get_config(key="min_pep_len", section="ml")
    max_pep_len = config.get_config(key="max_pep_len", section="ml")
    length_filter = config.get_config(key="length_filter", section="filter")
    len_tol_pos = config.get_config(key="len_tol_pos", section="filter") if length_filter else 0
    len_tol_neg = config.get_config(key="len_tol_neg", section="filter") if length_filter else 0
    # dist.barrier()
    # Run database search for each dict item
    unfiltered_time = 0

    print("Running filtered {} database search.".format("target" if rank == 0 else "decoy"))
    for key in spec_filt_dict:
        print("Searching for key {}.".format(key))
        spec_inds = []
        pep_inds = []
        psm_vals = []
        pep_info = PepInfo([], [], [])
        cum = 0
        for tol in range(len_tol_neg, len_tol_pos + 1):
            key_len, key_clv, key_mod = (
                int(key.split("-")[0]),
                int(key.split("-")[1]),
                int(key.split("-")[2]),
            )
            if key_len + tol < min_pep_len or key_len + tol > max_pep_len:
                continue
            file_name = "{}-{}-{}".format(key_len + tol, key_clv, key_mod)
            pep_classes_path = join(index_path, "peptide_classes")
            pep_file_path = join(pep_classes_path, file_name)
            if not os.path.exists(pep_file_path):
                print("Key {} not found in pep_dataset".format(pep_file_path))
                continue
            print("Processing file: {}".format(file_name))
            # process peptides
            pep_dataset = pepdataset.PeptideDataset(pep_dir, pep_file_path, decoy=rank == 1)
            # pep_loader = torch.utils.data.DataLoader(
            #     dataset=pep_dataset, batch_size=pep_batch_size,
            #     collate_fn=dbsearch.pep_collate)
            pep_info.pep_list += pep_dataset.pep_list
            pep_info.prot_list += pep_dataset.prot_list
            pep_info.pep_mass_list += pep_dataset.pep_mass_list

            # load embeddings
            pep_embeddings_path = join(index_path, "peptide_embeddings" if rank == 0 else "decoy_embeddings")
            embedding_file_path = join(pep_embeddings_path, file_name)
            e_peps = torch.load(embedding_file_path)
            # pep_data = [[idx + class_offsets[file_name], e_pep, mass]
            #             for idx, (e_pep, mass) in enumerate(zip(e_peps, pep_dataset.pep_mass_list))]
            pep_data = [
                [idx + cum, e_pep, mass] for idx, (e_pep, mass) in enumerate(zip(e_peps, pep_dataset.pep_mass_list))
            ]
            cum += len(pep_data)

            print("Searching against key {} with {} peptides.".format(file_name, len(pep_dataset.pep_mass_list)))
            spec_subset = spec_filt_dict[key]
            search_loader = torch.utils.data.DataLoader(
                dataset=spec_subset,
                num_workers=0,
                batch_size=search_spec_batch_size,
                shuffle=False,
            )
            pep_dataset = None
            unfiltered_start_time = time.time()
            l_spec_inds, l_pep_inds, l_psm_vals = dbsearch.filtered_parallel_search(search_loader, pep_data, rank)
            unfiltered_time += time.time() - unfiltered_start_time

            if not l_spec_inds:
                continue
            spec_inds.extend(l_spec_inds)
            pep_inds.append(l_pep_inds)
            psm_vals.append(l_psm_vals)

        if not spec_inds:
            continue
        # spec_inds.extend(l_spec_inds)
        # pep_inds.append(l_pep_inds)
        # psm_vals.append(l_psm_vals)

        pep_inds = torch.cat(pep_inds, 0)
        psm_vals = torch.cat(psm_vals, 0)

        print("{} PSMS: {}".format("Target" if rank == 0 else "Decoy", len(pep_inds)))

        # 4 - Write PSMs to pin file
        postprocess.write_to_pin(rank, pep_inds, psm_vals, spec_inds, pep_info, spec_charges, out_pin_dir)


def run_atles_search(rank, world_size, config_path, args_dict):
    setup(rank, world_size)
    config.init_config(config_path)
    arg_parse.process_args_dict(args_dict)
    pep_dir = config.get_config(key="pep_dir", section="search")
    pep_index_name = PurePath(pep_dir).name
    index_path = join(config.get_config(key="index_path", section="search"), pep_index_name, "filtered")

    length_filter = config.get_config(key="length_filter", section="filter")
    len_tol_pos = config.get_config(key="len_tol_pos", section="filter") if length_filter else 0

    spectra = PurePath(config.get_config(key="prep_path", section="search")).name
    filt = "filt{}".format(len_tol_pos)
    out_pin_dir = join(os.getcwd(), "percolator", pep_index_name + "-" + filt + "-" + spectra)

    print(f"Pep dir name {pep_dir}")
    print(f"length filter {length_filter}")
    print(f"len_tol_pos {len_tol_pos}")

    with torch.no_grad():
        print("Running filtered ooc search on {}.".format(pep_index_name))
        snap_model = get_snap_model(rank)
        if not os.path.exists(index_path):
            if rank == 0:
                classify_peptides(index_path)
            dist.barrier()
            process_peptides(rank, snap_model, index_path)

        t_time = time.time()
        e_specs, lens, cleavs, mods, spec_masses, spec_charges = process_spectra(rank, snap_model)
        spec_filt_dict = create_spectra_dict(lens, cleavs, mods, e_specs, spec_masses)
        l_time = time.time()
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

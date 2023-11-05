import time
import argparse
import os
from collections import defaultdict
from os.path import dirname, join

import pandas as pd
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from tqdm import tqdm

from src.atlesconfig import config
from src.atlespredict import (
    dbsearch,
    pepdataset,
    postprocess,
    specdataset,
    specollate_model,
)
from src.atlestrain import model


def run_atles(rank, spec_loader):
    model_ = model.Net().to(rank)
    model_ = nn.parallel.DistributedDataParallel(model_, device_ids=[rank])
    # model_.load_state_dict(torch.load('atles-out/16403437/models/pt-mass-ch-16403437-1toz70vi-472.pt')['model_state_dict'])
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


def run_specollate_par(rank, world_size):
    setup(rank, world_size)
    # rank = config.get_config(key="rank", section="input")
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
    pep_dir = config.get_config(key="pep_dir", section="search")
    out_pin_dir = config.get_config(key="out_pin_dir", section="search")

    # scratch_loc = "/scratch/mtari008/job_" + os.environ['SLURM_JOB_ID'] + "/"

    # mgf_dir     = scratch_loc + mgf_dir
    # prep_dir    = scratch_loc + prep_dir
    # pep_dir     = scratch_loc + pep_dir
    # out_pin_dir = scratch_loc + out_pin_dir

    if rank == 0:
        tqdm.write("Reading input files...")

    prep_path = config.get_config(section="search", key="prep_path")
    spec_batch_size = config.get_config(key="spec_batch_size", section="search")
    spec_dataset = specdataset.SpectraDataset(join(prep_path, "specs.pkl"))
    spec_loader = torch.utils.data.DataLoader(
        dataset=spec_dataset,
        batch_size=spec_batch_size,
        collate_fn=dbsearch.spec_collate,
    )

    atles_start_time = time.time()
    lens, cleavs, mods = run_atles(rank, spec_loader)
    atles_end_time = time.time()
    atles_time = atles_end_time - atles_start_time

    pep_batch_size = config.get_config(key="pep_batch_size", section="search")

    pep_dataset = pepdataset.PeptideDataset(pep_dir, decoy=rank == 1)
    pep_loader = torch.utils.data.DataLoader(
        dataset=pep_dataset, batch_size=pep_batch_size, collate_fn=dbsearch.pep_collate
    )

    dist.barrier()

    # os.environ['MASTER_ADDR'] = 'localhost'
    # os.environ['MASTER_PORT'] = '12350'
    # dist.init_process_group(backend='nccl', world_size=1, rank=0)
    # model_name = "512-embed-2-lstm-SnapLoss2D-80k-nist-massive-no-mc-semi-randbatch-62.pt" # 28.8k
    model_name = "512-embed-2-lstm-SnapLoss2D-80k-nist-massive-no-mc-semi-r2r-18.pt"  # 28.975k
    model_name = "512-embed-2-lstm-SnapLoss2D-80k-nist-massive-no-mc-semi-r2r2r-22.pt"
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

    print("Processing spectra...")
    e_specs = dbsearch.runSpeCollateModel(spec_loader, snap_model, "specs", rank)
    print("Spectra done!")

    dist.barrier()

    print("Processing {}...".format("Peptides" if rank == 0 else "Decoys"))
    e_peps = dbsearch.runSpeCollateModel(pep_loader, snap_model, "peps", rank)
    print("Peptides done!")

    dist.barrier()

    min_pep_len = config.get_config(key="min_pep_len", section="ml")
    max_pep_len = config.get_config(key="max_pep_len", section="ml")
    max_clvs = config.get_config(key="max_clvs", section="ml")

    length_filter = config.get_config(key="length_filter", section="filter")
    len_tol_pos = config.get_config(key="len_tol_pos", section="filter") if length_filter else 0
    len_tol_neg = config.get_config(key="len_tol_neg", section="filter") if length_filter else 0
    missed_cleavages_filter = config.get_config(key="missed_cleavages_filter", section="filter")
    modification_filter = config.get_config(key="modification_filter", section="filter")

    print("Creating spectra filtered dictionary.")
    spec_dataset.filt_dict = defaultdict(list)
    for idx, (l, clv, mod) in enumerate(zip(lens, cleavs, mods)):
        if min_pep_len <= l <= max_pep_len and 0 <= clv <= max_clvs:
            l = int(l) if length_filter else 0
            clv = int(clv) if missed_cleavages_filter else 0
            mod = int(mod) if modification_filter else 0
            key = "{}-{}-{}".format(l, clv, int(mod))
            spec_dataset.filt_dict[key].append([idx, e_specs[idx], spec_dataset.masses[idx]])

    pep_batch_size = config.get_config(key="pep_batch_size", section="search")
    ####### rank==1 decides whether to search against decoy database #######
    pep_dataset.filt_dict = defaultdict(list)
    print("Creating peptide filtered dictionary.")
    for idx, (pep, clv, mod) in enumerate(
        zip(
            pep_dataset.pep_list,
            pep_dataset.missed_cleavs,
            pep_dataset.pep_modified_list,
        )
    ):
        pep_len = sum(map(str.isupper, pep))
        if min_pep_len <= pep_len <= max_pep_len and 0 <= clv <= max_clvs:
            pep_len = int(pep_len) if length_filter else 0
            clv = int(clv) if missed_cleavages_filter else 0
            mod = int(mod) if modification_filter else 0
            key = "{}-{}-{}".format(pep_len, clv, int(mod))
            pep_dataset.filt_dict[key].append([idx, e_peps[idx], pep_dataset.pep_mass_list[idx]])

    search_spec_batch_size = config.get_config(key="search_spec_batch_size", section="search")
    dist.barrier()
    if rank == 0:
        search_start_time = time.time()
    # Run database search for each dict item
    unfiltered_time = 0
    spec_inds = []
    pep_inds = []
    psm_vals = []
    print("Running filtered {} database search.".format("target" if rank == 0 else "decoy"))
    for key in spec_dataset.filt_dict:
        print("Searching for key {}.".format(key))
        for tol in range(len_tol_neg, len_tol_pos + 1):
            key_len, key_clv, key_mod = (
                int(key.split("-")[0]),
                int(key.split("-")[1]),
                int(key.split("-")[2]),
            )
            pep_key = "{}-{}-{}".format(key_len + tol, key_clv, key_mod)
            if pep_key not in pep_dataset.filt_dict:
                print("Key {} not found in pep_dataset".format(pep_key))
                continue
            print("Searching against key {} with {} peptides.".format(pep_key, len(pep_dataset.filt_dict[pep_key])))
            spec_subset = spec_dataset.filt_dict[key]
            search_loader = torch.utils.data.DataLoader(
                dataset=spec_subset,
                num_workers=0,
                batch_size=search_spec_batch_size,
                shuffle=False,
            )
            unfiltered_start_time = time.time()
            l_spec_inds, l_pep_inds, l_psm_vals = dbsearch.filtered_parallel_search(
                search_loader, pep_dataset.filt_dict[pep_key], rank
            )
            unfiltered_time += time.time() - unfiltered_start_time

            if not l_spec_inds:
                continue
            spec_inds.extend(l_spec_inds)
            pep_inds.append(l_pep_inds)
            psm_vals.append(l_psm_vals)

        # if not l_spec_inds:
        #     continue
        # spec_inds.extend(l_spec_inds)
        # pep_inds.append(l_pep_inds)
        # psm_vals.append(l_psm_vals)

    pep_inds = torch.cat(pep_inds, 0)
    psm_vals = torch.cat(psm_vals, 0)

    print("{} PSMS: {}".format("Target" if rank == 0 else "Decoy", len(pep_inds)))

    dist.barrier()
    print("Unfiltered Time: {}".format(unfiltered_time))
    if rank == 0:
        print("Database Search Time Taken: {}".format(time.time() - search_start_time))
        print("Database Search Time + Atles Time Taken: {}".format((time.time() - search_start_time) + atles_time))

    pin_charge = config.get_config(section="search", key="charge")
    charge_cols = [f"charge-{ch+1}" for ch in range(pin_charge)]
    cols = (
        [
            "SpecId",
            "Label",
            "ScanNr",
            "SNAP",
            "ExpMass",
            "CalcMass",
            "deltCn",
            "deltLCn",
        ]
        + charge_cols
        + ["dM", "absdM", "enzInt", "PepLen", "Peptide", "Proteins"]
    )

    dist.barrier()

    if rank == 0:
        print("Generating percolator pin files...")
    global_out = postprocess.generate_percolator_input(
        pep_inds,
        psm_vals,
        spec_inds,
        pep_dataset,
        spec_dataset.charges,
        "target" if rank == 0 else "decoy",
    )
    df = pd.DataFrame(global_out, columns=cols)
    df.sort_values(by="SNAP", inplace=True, ascending=False)
    os.makedirs(out_pin_dir, exist_ok=True)
    df.to_csv(
        join(out_pin_dir, "target.pin" if rank == 0 else "decoy.pin"),
        sep="\t",
        index=False,
    )

    if rank == 0:
        print("Wrote percolator files: ")
    dist.barrier()
    print("{}".format(join(out_pin_dir, "target.pin") if rank == 0 else join(out_pin_dir, "decoy.pin")))


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(config.get_config(key="master_port", section="input"))
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", world_size=world_size, rank=rank)


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
    num_gpus = 2
    mp.spawn(run_specollate_par, args=(num_gpus,), nprocs=num_gpus, join=True)
    print("Total time: {}".format(time.time() - start_time))

import argparse
import pandas as pd
import os
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.distributed as dist
from os.path import join

from src.snapconfig import config
from src.snaptrain import model
from src.snapsearch import pepdataset, specdataset, dbsearch, postprocess, preprocess


if __name__ == '__main__':
    # Initialize parser 
    parser = argparse.ArgumentParser()
    
    # Adding optional argument 
    parser.add_argument("-c", "--config", help="Path to the config file.")
    parser.add_argument("-p", "--preprocess", help="Which server the code is running on. \
        Options: raptor, comet. Default: comet", default="True")
    
    # Read arguments from command line 
    args = parser.parse_args() 

    if args.config:
        tqdm.write("config: %s" % args.path)
    # config.PARAM_PATH = args.config if args.config else join((os.path.dirname(__file__)), "config.ini")
    
    mgf_dir     = config.get_config(key="mgf_dir", section="search")
    prep_dir    = config.get_config(key="prep_dir", section="search")
    pep_dir     = config.get_config(key="pep_dir", section="search")
    out_pin_dir = config.get_config(key="out_pin_dir", section="search")

    # scratch_loc = "/scratch/mtari008/job_" + os.environ['SLURM_JOB_ID'] + "/"

    # mgf_dir     = scratch_loc + mgf_dir
    # prep_dir    = scratch_loc + prep_dir
    # pep_dir     = scratch_loc + pep_dir
    # out_pin_dir = scratch_loc + out_pin_dir
    
    if args.preprocess and args.preprocess == "True":
        tqdm.write("Preprocessing mgf files...")
        preprocess.preprocess_mgfs(mgf_dir, prep_dir)

    tqdm.write("Reading input files...")
    spec_dataset = specdataset.SpectralDataset(prep_dir)
    pep_dataset  = pepdataset.PeptideDataset(pep_dir)
    dec_dataset  = pepdataset.PeptideDataset(pep_dir, decoy=True)

    spec_batch_size = config.get_config(key="spec_batch_size", section="search")
    pep_batch_size  = config.get_config(key="pep_batch_size", section="search")

    print("Generating data loaders...")
    spec_loader = torch.utils.data.DataLoader(
        dataset=spec_dataset, batch_size=spec_batch_size,
        collate_fn=dbsearch.spec_collate)
    pep_loader = torch.utils.data.DataLoader(
        dataset=pep_dataset, batch_size=pep_batch_size,
        collate_fn=dbsearch.pep_collate)
    dec_loader = torch.utils.data.DataLoader(
        dataset=dec_dataset, batch_size=pep_batch_size,
        collate_fn=dbsearch.pep_collate)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12350'
    dist.init_process_group(backend='nccl', world_size=1, rank=0)
    snap_model = model.Net(vocab_size=30, embedding_dim=512, hidden_lstm_dim=512, lstm_layers=2).to(device)
    # print(snap_model.linear1_1.weight.shape)
    snap_model = nn.parallel.DistributedDataParallel(snap_model, device_ids=[0])
    # snap_model.load_state_dict(torch.load('models/32-embed-2-lstm-SnapLoss2-noch-3k-1k-152.pt')['model_state_dict'])
    # below one has 26975 identified peptides.
    # snap_model.load_state_dict(torch.load('models/512-embed-2-lstm-SnapLoss-noch-80k-nist-massive-52.pt')['model_state_dict'])
    model_name = join("./models", config.get_config(key="model_name", section="search"))
    snap_model.load_state_dict(torch.load(model_name)['model_state_dict'])
    snap_model = snap_model.module
    snap_model.eval()
    snap_model

    print("Processing spectra...")
    e_specs = dbsearch.runModel(spec_loader, snap_model, "specs", device)
    print("Spectra done!")

    print("Processing peptides...")
    e_peps = dbsearch.runModel(pep_loader, snap_model, "peps", device)
    print("Peptides done!")

    print("Processing decoys...")
    e_decs = dbsearch.runModel(dec_loader, snap_model, "peps", device)
    print("Decoys done!")

    search_spec_batch_size = config.get_config(key="search_spec_batch_size", section="search")
    datasets = {"spec_dataset":spec_dataset, "pep_dataset":pep_dataset, "dec_dataset":dec_dataset}
    embeddings = {"e_specs":e_specs, "e_peps":e_peps, "e_decs":e_decs}
    search_loader = torch.utils.data.DataLoader(
        dataset=e_specs, batch_size=search_spec_batch_size, shuffle=False)
    
    pep_inds, pep_vals, dec_inds, dec_vals = dbsearch.search(search_loader, datasets, embeddings, device)

    pin_charge = config.get_config(section="search", key="charge")
    charge_cols = [f"charge-{ch+1}" for ch in range(pin_charge)]
    cols = ["SpecId", "Label", "ScanNr", "SNAP", "ExpMass", "CalcMass", "deltCn", "deltLCn"] + charge_cols + ["dM", "absdM", "enzInt", "PepLen", "Peptide", "Proteins"]
    
    print("Generating percolator pin files...")
    global_out = postprocess.generate_percolator_input(pep_inds, pep_vals, pep_dataset, spec_dataset, "target")
    df = pd.DataFrame(global_out, columns=cols)
    df.sort_values(by="SNAP", inplace=True, ascending=False)
    df.to_csv(join(out_pin_dir, "target.pin"), sep="\t", index=False)

    global_out = postprocess.generate_percolator_input(dec_inds, dec_vals, dec_dataset, spec_dataset, "decoy")
    df = pd.DataFrame(global_out, columns=cols)
    df.sort_values(by="SNAP", inplace=True, ascending=False)
    df.to_csv(join(out_pin_dir, "decoy.pin"), sep="\t", index=False)
    print("Wrote percolator files: \n{}\n{}".format(
        join(out_pin_dir, "target.pin"), join(out_pin_dir, "decoy.pin")))
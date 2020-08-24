import argparse
import os
import shutil
from os.path import join
import re
import pickle
import random as rand
import numpy as np

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
torch.manual_seed(0)

from src.snapconfig import config
from src.snaptrain import dataset, model, trainmodel

# with redirect_output("deepSNAP_redirect.txtS"):
train_loss = []
test_loss = []
train_accuracy = []
test_accuracy = []

def run_par(rank, world_size):
    #rank = dist.get_rank()
    setup(rank, world_size)

    batch_size  = config.get_config(section="ml", key="batch_size")
    charge      = config.get_config(section='input', key='charge')
    use_mods    = config.get_config(section='input', key='use_mods')
    num_mods    = config.get_config(section='input', key='num_mods')
    filt        = {'charge': charge, 'mods': num_mods if use_mods else 0}
    test_size   = config.get_config(section='ml', key='test_size')

    msp_dir     = config.get_config(section='preprocess', key='msp_dir')
    in_tensor_dir = config.get_config(section='preprocess', key='in_tensor_dir')
    # msp_dir = "/DeepSNAP/data/msp-labeled/"
    # in_tensor_dir = "/scratch/train_lstm/"
    print(in_tensor_dir)

    listing_path = join(in_tensor_dir, 'pep_spec.pkl')
    pep_file_names, spec_file_names_lists = load_file_names(filt=filt, listing_path=listing_path)
    means = np.load(join(in_tensor_dir, "means.npy"))
    stds = np.load(join(in_tensor_dir, "stds.npy"))
    
    split_rand_state = rand.randint(0, 1000)
    trains, tests = train_test_split(
        list(zip(pep_file_names, spec_file_names_lists)), test_size=test_size, 
        random_state=split_rand_state, shuffle=True)

    train_peps, train_specs = map(list, zip(*trains))
    test_peps, test_specs = map(list, zip(*tests))
    train_dataset = dataset.LabeledSpectra(in_tensor_dir, train_peps, train_specs, means, stds)
    test_dataset  = dataset.LabeledSpectra(in_tensor_dir, test_peps, test_specs, means, stds)

    vocab_size = train_dataset.vocab_size

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    # test_sampler = torch.utils.data.distributed.DistributedSampler(
    #     test_dataset,
    #     num_replicas=world_size,
    #     rank=rank
    # )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=batch_size,
        drop_last=True, num_workers=8, collate_fn=psm_collate,
        #sampler=train_sampler
        shuffle=True
        )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=batch_size,
        collate_fn=psm_collate, drop_last=True, num_workers=8,
        shuffle=True)

    print("Learning without DeepNovo dataset.")
    lr = 0.0005
    print("Learning Rate: {}".format(lr))
    num_epochs = 500
    weight_decay = 0.0001
    print("Weigh Decay: {}".format(weight_decay))
    margin = 0.2

    triplet_loss = nn.TripletMarginLoss(margin=margin, p=2, reduction='sum')
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
    zero_tensor = torch.tensor(0.).to(rank)

    model_ = model.Net(vocab_size, output_size=512, embedding_dim=256,
                        hidden_lstm_dim=512, # 1024 
                        lstm_layers=1).to(rank)
    model_ = nn.parallel.DistributedDataParallel(model_, device_ids=[rank])
    optimizer = optim.Adam(model_.parameters(), lr=lr, weight_decay=weight_decay)
    #optimizer = optim.SGD(model_.parameters(), lr=lr)

    # Create directories to cache files on the scratch storage.
    # moved out of this function since only once per node is needed.
    # os.makedirs("/scratch/train_lstm/spectra/", 0o755, exist_ok=True)
    # os.makedirs("/scratch/train_lstm/peptides/", 0o755, exist_ok=True)

    for epoch in range(num_epochs):
        l_epoch = (epoch * world_size) + rank
        print("Epoch: {}".format(epoch))
        train_sampler.set_epoch(l_epoch)
        trainmodel.train(model_, rank, train_loader, triplet_loss, optimizer)
        trainmodel.test(model_, rank, test_loader, triplet_loss)

        if epoch % 2 == 0 and rank == 0:
            torch.save(model_, 'models/hcd/model-{}.pt'.format(epoch))
        
        dist.barrier()
    
    cleanup()

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'
    dist.init_process_group(backend='nccl', world_size=world_size, rank=rank)
    # dist.init_process_group(backend='nccl', world_size=world_size, rank=rank)

def cleanup():
    dist.destroy_process_group()


def apply_filter(filt, file_name):
    try:
        file_parts = re.search(r"(\d+)-(\d+)-(\d+.\d+)-(\d+)-(\d+).[pt|npy]", file_name)
        charge = int(file_parts[4])
        mods = int(file_parts[5])
    except:
        print(file_name)
        print(file_parts)
    
    if ((filt["charge"] == 0 or charge <= filt["charge"])
        and (mods <= filt["mods"])):
        return True
    
    return False


def load_file_names(filt, listing_path):
    'Load the peptide and corresponding spectra file names that satisfy the filter'
    with open(listing_path, 'rb') as f:
        dir_listing = pickle.load(f)

    pep_file_names = []
    spec_file_names_lists = []
    for pep, spec_list in dir_listing:
        spec_file_list = []
        for spec in spec_list:
            if apply_filter(filt, spec):
                spec_file_list.append(spec)
        if spec_file_list:
            pep_file_names.append(pep)
            spec_file_names_lists.append(spec_file_list)

    assert len(pep_file_names) == len(spec_file_names_lists)
    return pep_file_names, spec_file_names_lists


def psm_collate(batch):
    specs = torch.cat([item[0] for item in batch], 0)
    peps = torch.stack([item[1] for item in batch], 0)
    dpeps = torch.stack([item[2] for item in batch if len(item[2]) > 0])
    peps_set = set(map(tuple, peps.tolist()))
    dpeps_set = set(map(tuple, dpeps.tolist()))
    dpeps_list = list(dpeps_set - dpeps_set.intersection(peps_set))
    dpeps = torch.tensor(dpeps_list, dtype=torch.long)
    counts = np.array([item[3] for item in batch])
    return [specs, peps, dpeps, counts]

# drop_prob=0.5
# print(vocab_size)


if __name__ == '__main__':

    # Initialize parser 
    parser = argparse.ArgumentParser()
    
    # Adding optional argument 
    parser.add_argument("-j", "--job-id", help="No arguments should be passed. \
        Instead use the shell script provided with the code.") 
    parser.add_argument("-p", "--path", help="Path to the config file.")
    parser.add_argument("-s", "--server-name", help="Which server the code is running on. \
        Options: raptor, comet. Default: comet", default="comet")
    
    # Read arguments from command line 
    args = parser.parse_args() 
    
    if args.job_id: 
        print("job_id: %s" % args.job_id)
        job_id = args.job_id

    if args.path:
        print("job_id: %s" % args.path)
        scratch = args.path

    mp.set_start_method('forkserver')
    config.PARAM_PATH = join((os.path.dirname(__file__)), "config.ini")


    
    do_learn = True
    save_frequency = 2
    
    # torch.manual_seed(0)
    # torch.cuda.manual_seed(0)

    num_gpus = torch.cuda.device_count()
    print("Num GPUs: {}".format(num_gpus))
    mp.spawn(run_par, args=(num_gpus,), nprocs=num_gpus, join=True)

    # model.linear1_1.weight.requires_grad = False
    # model.linear1_1.bias.requires_grad = False
    # model.linear1_2.weight.requires_grad = False
    # model.linear1_2.bias.requires_grad = False

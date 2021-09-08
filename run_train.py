import os
import timeit
from os.path import join
import re
import pickle
import numpy as np

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import BatchSampler
torch.manual_seed(1)

from src.snapconfig import config
from src.snaptrain import dataset, model, trainmodel, sampler
from src.snaputils import simulatespectra as sim

# with redirect_output("deepSNAP_redirect.txtS"):
train_loss     = []
test_loss      = []
train_accuracy = []
test_accuracy  = []

def run_par(rank, world_size):
    model_name = config.get_config(section="ml", key="model_name") #first k is spec size second is batch size
    print("Training {}."
        .format(model_name))
    # model_name = "time-4096"
    # wandb.init(project="SpeCollate", entity="pcds")
    # wandb.run.name = "{}-{}".format(model_name, wandb.run.id)
    # wandb.config.learning_rate = 0.00005
    
    setup(rank, world_size)

    batch_size  = config.get_config(section="ml", key="batch_size")
    in_tensor_dir = config.get_config(section='preprocess', key='in_tensor_dir') # for raptor
    # in_tensor_dir = "/scratch/mtari008/job_" + os.environ['SLURM_JOB_ID'] + "/nist_massiv_80k_ch_graymass27-semi" # for comet
    
    train_peps, train_specs, train_masses, test_peps, test_specs, test_masses = read_split_listings(in_tensor_dir)

    train_sampler = sampler.PSMSampler(train_masses)
    test_sampler  = sampler.PSMSampler(test_masses)

    train_batch_sampler = BatchSampler(sampler=train_sampler, batch_size=batch_size, drop_last=False)
    test_batch_sampler = BatchSampler(sampler=test_sampler, batch_size=batch_size, drop_last=False)

    train_dataset = dataset.LabeledSpectra(in_tensor_dir, train_peps, train_specs)
    test_dataset  = dataset.LabeledSpectra(in_tensor_dir, test_peps, test_specs)

    vocab_size = train_dataset.vocab_size

    # train_sampler = torch.utils.data.distributed.DistributedSampler(
    #     train_dataset,
    #     num_replicas=world_size,
    #     rank=rank,
    #     shuffle=True
    # )
    # test_sampler = torch.utils.data.distributed.DistributedSampler(
    #     test_dataset,
    #     num_replicas=world_size,
    #     rank=rank
    # )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, num_workers=8, collate_fn=psm_collate,
        # batch_sampler=train_batch_sampler
        batch_size=batch_size, shuffle=True
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, num_workers=8, collate_fn=psm_collate,
        # batch_sampler=test_batch_sampler,
        batch_size=batch_size, shuffle=True
    )

    lr = config.get_config(section="ml", key="lr")
    num_epochs = config.get_config(section="ml", key="epochs")
    weight_decay = config.get_config(section="ml", key="weight_decay")
    margin = config.get_config(section="ml", key="margin")
    snp_weight = config.get_config(section="ml", key="snp_weight")
    ce_weight = config.get_config(section="ml", key="ce_weight")
    mse_weight = config.get_config(section="ml", key="mse_weight")

    if rank == 0:
        print("same as 0 with filter enabled.")
        print("Learning Rate: {}".format(lr))
        print("Weigh Decay: {}".format(weight_decay))
        print("snp weight: {}".format(snp_weight))
        print("ce weight: {}".format(ce_weight))
        print("mse weight: {}".format(mse_weight))
        print("margin: {}".format(margin))

    triplet_loss = nn.TripletMarginLoss(margin=margin, p=2, reduction="sum", swap=True)
    cross_entropy_loss = nn.CrossEntropyLoss(reduction="sum")
    mse_loss = nn.MSELoss(reduction="mean")
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)

    model_ = model.Net(vocab_size, output_size=512, embedding_dim=512, hidden_lstm_dim=512, lstm_layers=2).to(rank)
    optimizer = optim.Adam(model_.parameters(), lr=lr, weight_decay=weight_decay)

    #model_ = torch.load("./models/hcd/single-mod-10-0.3-0.0005.pt")
    model_ = nn.parallel.DistributedDataParallel(model_, device_ids=[rank])
    # model_, optimizer = apex.amp.initialize(model_, optimizer, opt_level="O1")
    # model_ = apex.parallel.DistributedDataParallel(model_)
    # model_.load_state_dict(torch.load("models/hcd/512-embed-2-lstm-SnapLoss2D-80k-nist-massive-gmc-semi-r-10.pt")["model_state_dict"])
    #optimizer = optim.SGD(model_.parameters(), lr=lr)

    # Create directories to cache files on the scratch storage.
    # moved out of this function since only once per node is needed.
    # os.makedirs("/scratch/train_lstm/spectra/", 0o755, exist_ok=True)
    # os.makedirs("/scratch/train_lstm/peptides/", 0o755, exist_ok=True)

    # wandb.watch(model_)
    for epoch in range(num_epochs):
        l_epoch = (epoch * world_size) + rank
        print("Epoch: {}".format(l_epoch))
        # train_sampler.set_epoch(l_epoch)
        start_time = timeit.default_timer()
        loss = trainmodel.train(model_, rank, train_loader, triplet_loss, cross_entropy_loss, mse_loss, optimizer, l_epoch)
        trainmodel.test(model_, rank, test_loader, triplet_loss, cross_entropy_loss, mse_loss, l_epoch)
        elapsed = timeit.default_timer() - start_time
        print("time takes: {} secs.".format(elapsed))

        dist.barrier()

        if l_epoch % 1 == 0 and rank == 0:
            torch.save({
            'epoch': l_epoch,
            'model_state_dict': model_.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, "./models/{}-{}.pt".format(model_name, l_epoch))
            # model_name = "single_mod-{}-{}.pt".format(epoch, lr)
            # print(wandb.run.dir)
            # torch.save(model_.state_dict(), join("./models/hcd/", model_name))
            # wandb.save("{}-{}.pt".format(model_name, l_epoch))
    
    cleanup()

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(config.get_config(section="input", key="master_port"))
    torch.cuda.set_device(rank)
    dist.init_process_group(backend='nccl', world_size=world_size, rank=rank)
    # dist.init_process_group(backend='nccl', world_size=world_size, rank=rank)

def cleanup():
    dist.destroy_process_group()


def apply_filter(filt, file_name):
    file_parts = []
    charge = 0
    mods = 0
    try:
        file_parts = re.search(r"(\d+)-(\d+)-(\d+.\d+)-(\d+)-(\d+).[pt|npy]", file_name)
        charge = int(file_parts[4])
        mods = int(file_parts[5])
    except:
        print(file_name)
        print(file_parts)
    
    if ((filt["charge"] == 0 or charge <= filt["charge"]) # change this back to <=
        and (mods <= filt["mods"])):
        return True
    
    return False


def psm_collate(batch):
    specs = torch.cat([item[0] for item in batch], 0)
    peps = torch.stack([item[1] for item in batch], 0)
    dpeps = torch.stack([item[2] for item in batch if item[2].nelement() > 0], 0)
    peps_set = set(map(tuple, peps.tolist()))
    dpeps_set = set(map(tuple, dpeps.tolist()))
    dpeps_list = list(dpeps_set - dpeps_set.intersection(peps_set))
    dpeps = torch.tensor(dpeps_list, dtype=torch.long)
    charges = torch.cat([item[3] for item in batch], 0)
    # spec_masses = torch.cat([item[4] for item in batch], 0).view(-1, 1)
    # pep_masses = torch.FloatTensor([item[5] for item in batch]).view(-1, 1)
    
    # aas = ['_PAD'] + list(config.AAMass.keys())
    # idx2aa = {i:a for i, a in enumerate(aas)}
    # dpep_masses = []
    # for dpep in dpeps:
    #     dpep = "".join([idx2aa[i.item()] for i in dpep if i > 0])
    #     dpep_masses.append(sim.get_pep_mass(dpep))
    # dpep_masses = torch.FloatTensor(dpep_masses).view(-1, 1)
    counts = np.array([item[4] for item in batch])
    # return [specs, peps, dpeps, charges, spec_masses, pep_masses, dpep_masses, counts]
    return [specs, peps, dpeps, charges, counts]

# drop_prob=0.5
# print(vocab_size)

def read_split_listings(l_in_tensor_dir):
    print(l_in_tensor_dir)

    print("Reading train test split listings from pickles.")
    with open(join(l_in_tensor_dir, "train_peps.pkl"), "rb") as trp:
        train_peps = pickle.load(trp)
    with open(join(l_in_tensor_dir, "train_specs.pkl"), "rb") as trs:
        train_specs = pickle.load(trs)
    with open(join(l_in_tensor_dir, "test_peps.pkl"), "rb") as tep:
        test_peps = pickle.load(tep)
    with open(join(l_in_tensor_dir, "test_specs.pkl"), "rb") as tes:
        test_specs = pickle.load(tes)

    train_masses = []
    for train_pep in train_peps:
        train_mass = float(re.search(r"(\d+)-(\d+.\d+).pep", train_pep)[2])
        train_masses.append(train_mass)

    test_masses = []
    for test_pep in test_peps:
        test_mass = float(re.search(r"(\d+)-(\d+.\d+).pep", test_pep)[2])
        test_masses.append(test_mass)

    train_peps, train_specs, train_masses = zip(*sorted(zip(train_peps, train_specs, train_masses), key=lambda x: x[2]))
    train_peps, train_specs, train_masses = list(train_peps), list(train_specs), list(train_masses)

    test_peps, test_specs, test_masses = zip(*sorted(zip(test_peps, test_specs, test_masses), key=lambda x: x[2]))
    test_peps, test_specs, test_masses = list(test_peps), list(test_specs), list(test_masses)

    return train_peps, train_specs, train_masses, test_peps, test_specs, test_masses


if __name__ == '__main__':

    # Initialize parser 
    # parser = argparse.ArgumentParser()
    
    # # Adding optional argument 
    # parser.add_argument("-j", "--job-id", help="No arguments should be passed. \
    #     Instead use the shell script provided with the code.") 
    # parser.add_argument("-p", "--path", help="Path to the config file.")
    # parser.add_argument("-s", "--server-name", help="Which server the code is running on. \
    #     Options: raptor, comet. Default: comet", default="comet")
    
    # # Read arguments from command line 
    # args = parser.parse_args() 
    
    # if args.job_id: 
    #     print("job_id: %s" % args.job_id)
    #     job_id = args.job_id

    # if args.path:
    #     print("job_id: %s" % args.path)
    #     scratch = args.path

    

    mp.freeze_support() # needed to package multiprocessing libraries
    mp.set_start_method('forkserver')
    config.PARAM_PATH = join((os.path.dirname(__file__)), "config.ini")
    
    do_learn = True
    save_frequency = 2
    
    # torch.manual_seed(0)
    # torch.cuda.manual_seed(0)

    num_gpus = torch.cuda.device_count()
    print("Num GPUs: {}".format(num_gpus))
    # mp.spawn(run_par, args=(num_gpus,), nprocs=num_gpus, join=True)
    run_par(0, 1)

    # model.linear1_1.weight.requires_grad = False
    # model.linear1_1.bias.requires_grad = False
    # model.linear1_2.weight.requires_grad = False
    # model.linear1_2.bias.requires_grad = False
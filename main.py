import argparse
import os
import shutil
from os.path import join

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
torch.manual_seed(0)

from src.snapconfig import config
from src.snapprocess import dataset, model, trainmodel

# with redirect_output("deepSNAP_redirect.txtS"):
train_loss = []
test_loss = []
train_accuracy = []
test_accuracy = []

def run_par(rank, world_size):
    #rank = dist.get_rank()
    setup(rank, world_size)

    batch_size = config.get_config(section="ml", key="batch_size")
    charge = config.get_config(section='input', key='charge')
    use_mods = config.get_config(section='input', key='use_mods')
    filt = {'charge': charge, 'modified': use_mods}

    msp_dir = config.get_config(section='preprocess', key='msp_dir')
    in_tensor_dir = config.get_config(section='preprocess', key='in_tensor_dir')
    # msp_dir = "/DeepSNAP/data/msp-labeled/"
    # in_tensor_dir = "/scratch/train_lstm/"
    print(in_tensor_dir)
    train_dataset = dataset.LabeledSpectra(in_tensor_dir, filt, test=False)
    test_dataset = dataset.LabeledSpectra(in_tensor_dir, filt, test=True)

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
        drop_last=True, num_workers=24,
        #sampler=train_sampler
        shuffle=True
        )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=False,
        drop_last=True, num_workers=24)

    lr = 0.00001
    num_epochs = 500
    weight_decay = 0.00001
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
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(backend='nccl', world_size=world_size, rank=rank)
    # dist.init_process_group(backend='nccl', world_size=world_size, rank=rank)

def cleanup():
    dist.destroy_process_group()

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

    #num_gpus = torch.cuda.device_count()
    num_gpus = 1
    print("Num GPUs: {}".format(num_gpus))
    mp.spawn(run_par, args=(num_gpus,), nprocs=num_gpus, join=True)

    # model.linear1_1.weight.requires_grad = False
    # model.linear1_1.bias.requires_grad = False
    # model.linear1_2.weight.requires_grad = False
    # model.linear1_2.bias.requires_grad = False

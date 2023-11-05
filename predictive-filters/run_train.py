import argparse
import os
import pickle
import re
import timeit
from os.path import join

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.sampler import WeightedRandomSampler

import wandb
from src.atlesconfig import config, wandbsetup
from src.atlestrain import dataset, model, trainmodel

torch.manual_seed(1)

# with redirect_output("deepSNAP_redirect.txtS"):
train_loss = []
test_loss = []
train_accuracy = []
test_accuracy = []


def run_par(rank, world_size):
    model_name = "nist-massive-deepnovo-mass-ch"  # first k is spec size second is batch size
    print("Training {}.".format(model_name))
    wandb.init(project="deepatles", entity="pcds")
    wandb.run.name = "{}-{}-{}".format(model_name, os.environ["SLURM_JOB_ID"], wandb.run.id)
    wandbsetup.set_wandb_config(wandb)

    setup(rank, world_size)

    batch_size = config.get_config(section="ml", key="batch_size")
    prep_dir = config.get_config(section="input", key="prep_dir")

    train_dataset = dataset.SpectraDataset(join(prep_dir, "train_specs.pkl"))
    val_dataset = dataset.SpectraDataset(join(prep_dir, "val_specs.pkl"))

    weights_all = train_dataset.class_weights_all
    weighted_sampler = WeightedRandomSampler(weights=weights_all, num_samples=len(weights_all), replacement=True)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        num_workers=0,
        collate_fn=psm_collate,
        batch_size=batch_size,
        sampler=weighted_sampler,
    )

    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset, num_workers=0, collate_fn=psm_collate, batch_size=batch_size, shuffle=False
    )

    lr = config.get_config(section="ml", key="lr")
    num_epochs = config.get_config(section="ml", key="epochs")
    weight_decay = config.get_config(section="ml", key="weight_decay")
    embedding_dim = config.get_config(section="ml", key="embedding_dim")
    encoder_layers = config.get_config(section="ml", key="encoder_layers")
    num_heads = config.get_config(section="ml", key="num_heads")
    dropout = config.get_config(section="ml", key="dropout")

    if rank == 0:
        print("Batch Size: {}".format(batch_size))
        print("Learning Rate: {}".format(lr))
        print("Weigh Decay: {}".format(weight_decay))
        print("Embedding Dim: {}".format(embedding_dim))
        print("Encoder Layers: {}".format(encoder_layers))
        print("Heads: {}".format(num_heads))
        print("Dropout: {}".format(dropout))

    ce_loss = nn.CrossEntropyLoss(reduction="mean")
    mse_loss = nn.MSELoss(reduction="mean")
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)

    model_ = model.Net().to(rank)
    for p in model_.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    optimizer = optim.Adam(model_.parameters(), lr=lr, weight_decay=weight_decay)
    # model_, optimizer = apex.amp.initialize(model_, optimizer, opt_level="O2")
    model_ = nn.parallel.DistributedDataParallel(model_)
    # model_.load_state_dict(torch.load("./models/attn-2-199.pt")["model_state_dict"])

    scaler = torch.cuda.amp.GradScaler()

    wandb.watch(model_)
    for epoch in range(num_epochs):
        l_epoch = (epoch * world_size) + rank
        print("Epoch: {}".format(l_epoch))
        # train_sampler.set_epoch(l_epoch)
        start_time = timeit.default_timer()
        loss = trainmodel.train(model_, rank, train_loader, mse_loss, ce_loss, optimizer, scaler, l_epoch)
        trainmodel.test(model_, rank, val_loader, mse_loss, ce_loss, l_epoch)
        elapsed = timeit.default_timer() - start_time
        print("time takes: {} secs.".format(elapsed))

        dist.barrier()

        if l_epoch % 1 == 0 and rank == 0:
            torch.save(
                {
                    "epoch": l_epoch,
                    "model_state_dict": model_.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": loss,
                },
                "atles-out/" + os.environ["SLURM_JOB_ID"] + "/models/{}-{}.pt".format(wandb.run.name, l_epoch),
            )
            # remove the model two steps before.
            if os.path.exists(
                "atles-out/" + os.environ["SLURM_JOB_ID"] + "/models/{}-{}.pt".format(wandb.run.name, l_epoch - 2)
            ):
                os.remove(
                    "atles-out/" + os.environ["SLURM_JOB_ID"] + "/models/{}-{}.pt".format(wandb.run.name, l_epoch - 2)
                )
            # model_name = "single_mod-{}-{}.pt".format(epoch, lr)
            # print(wandb.run.dir)
            # torch.save(model_.state_dict(), join("./models/hcd/", model_name))
            wandb.save("{}-{}.pt".format(wandb.run.name, l_epoch))

    cleanup()


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(config.get_config(section="input", key="master_port"))
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", world_size=world_size, rank=rank)
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
    except Exception as e:
        print("Exception thrown: {}".format(str(e)))
        print(file_name)
        print(file_parts)

    if (filt["charge"] == 0 or charge <= filt["charge"]) and (mods <= filt["mods"]):  # change this back to <=
        return True

    return False


def psm_collate(batch):
    specs = torch.cat([item[0] for item in batch], 0)
    # mzs = torch.LongTensor([item[0] for item in batch])
    # ints = torch.LongTensor([item[1] for item in batch])
    char_mass = torch.FloatTensor([item[1] for item in batch])
    lens = torch.FloatTensor([item[2] for item in batch])
    mods = torch.LongTensor([item[3] for item in batch])
    miss_cleavs = torch.LongTensor([item[4] for item in batch])
    return [specs, char_mass, lens, mods, miss_cleavs]


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

    out_train_masses = []
    out_train_peps = []
    out_train_specs = []
    for train_pep, pep_train_specs in zip(train_peps, train_specs):
        for train_spec in pep_train_specs:
            out_train_peps.append(train_pep)
            out_train_specs.append(train_spec)
            spec_mass = float(re.search(r"(\d+)-(\d+)-(\d+.\d+)-(\d+)-(\d+).[pt|npy]", train_spec)[3])
            out_train_masses.append(spec_mass)

    out_test_masses = []
    out_test_peps = []
    out_test_specs = []
    for test_pep, pep_test_specs in zip(test_peps, test_specs):
        for test_spec in pep_test_specs:
            out_test_peps.append(test_pep)
            out_test_specs.append(test_spec)
            spec_mass = float(re.search(r"(\d+)-(\d+)-(\d+.\d+)-(\d+)-(\d+).[pt|npy]", test_spec)[3])
            out_test_masses.append(spec_mass)

    # train_peps, train_specs, train_masses = zip(*sorted(zip(
    # train_peps, train_specs, train_masses), key=lambda x: x[2]))
    # train_peps, train_specs, train_masses = list(train_peps), list(train_specs), list(train_masses)

    # test_peps, test_specs, test_masses = zip(*sorted(zip(test_peps, test_specs, test_masses), key=lambda x: x[2]))
    # test_peps, test_specs, test_masses = list(test_peps), list(test_specs), list(test_masses)

    return out_train_peps, out_train_specs, out_train_masses, out_test_peps, out_test_specs, out_test_masses


if __name__ == "__main__":

    # Initialize parser
    parser = argparse.ArgumentParser()

    # Adding optional argument
    parser.add_argument(
        "-j",
        "--job-id",
        help="No arguments should be passed. \
        Instead use the shell script provided with the code.",
    )
    parser.add_argument("-p", "--path", help="Path to the config file.")
    parser.add_argument(
        "-s",
        "--server-name",
        help="Which server the code is running on. \
        Options: raptor, comet. Default: comet",
        default="comet",
    )

    # Read arguments from command line
    args = parser.parse_args()

    if args.job_id:
        print("job_id: %s" % args.job_id)
        job_id = args.job_id

    if args.path:
        print("job_id: %s" % args.path)
        scratch = args.path

    mp.set_start_method("forkserver")
    config.param_path = join((os.path.dirname(__file__)), "config.ini")

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

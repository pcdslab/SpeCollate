import pickle
from os.path import join

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import sys
sys.path.append('../DeepSNAP')
import run_train as main

sys.path.append('../DeepSNAP/src')
from src.snapconfig import config
from src.snaptrain import dataset
from src.snaputils import simulatespectra as sim


print(config.get_config(section="search", key="charge"))
print(config.get_config(section="input", key="use_mods"))
print(config.get_config(section="ml", key="batch_size"))
print(config.get_config(section="input", key="num_species"))
print(config.get_config(section="search", key="num_mods"))
print(config.get_config(section="input", key="spec_size"))


def ppm(val, ppm_val):
    return (ppm_val / 1000000) * val


# adding useless comment
class Net(nn.Module):
    def __init__(
        self,
        vocab_size,
        output_size=512,
        embedding_dim=512,
        hidden_lstm_dim=1024,
        lstm_layers=2,
    ):
        super(Net, self).__init__()
        self.spec_size = config.get_config(section="input", key="spec_size")
        self.spec_size = 80000
        self.seq_len = config.get_config(section="ml", key="pep_seq_len")
        self.output_size = output_size
        self.lstm_layers = lstm_layers
        self.hidden_lstm_dim = hidden_lstm_dim
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size

        ################### Spectra branch ###################
        self.linear1_1 = nn.Linear(self.spec_size, 512)
        self.linear1_2 = nn.Linear(512, 256)

        ################### Peptide branch ###################
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim,
            self.hidden_lstm_dim,
            self.lstm_layers,
            # dropout=0.5,
            batch_first=True,
            bidirectional=True,
        )

        self.linear2_1 = nn.Linear(self.hidden_lstm_dim * 2, 512)  # 2048, 1024
        self.linear2_2 = nn.Linear(512, 256)
        do = config.get_config(section="ml", key="dropout")
        self.dropout1_1 = nn.Dropout(do)
        self.dropout2_1 = nn.Dropout(do)
        self.dropout2_2 = nn.Dropout(do)
        print("dropout: {}".format(do))

    def forward(self, data, data_type=None):
        assert not data_type or data_type == "specs" or data_type == "peps"
        res = []
        if not data_type or data_type == "specs":
            specs = data[0].squeeze()

            out = F.relu((self.linear1_1(specs.view(-1, self.spec_size))))
            out = self.dropout1_1(out)

            out_spec = F.relu(self.linear1_2(out))
            out_spec = F.normalize(out_spec)
            res.append(out_spec)

        if not data_type or data_type == "peps":
            for peps in data[1:3]:
                peps = peps.squeeze()
                embeds = self.embedding(peps)
                hidden = self.init_hidden(len(peps))
                hidden = tuple([e.data for e in hidden])
                lstm_out, _ = self.lstm(embeds, hidden)
                lstm_out = lstm_out[:, -1, :]
                out = lstm_out.contiguous().view(-1, self.hidden_lstm_dim * 2)
                out = self.dropout2_1(out)

                out = F.relu((self.linear2_1(out)))
                out = self.dropout2_2(out)

                out_pep = F.relu(self.linear2_2(out))
                out_pep = F.normalize(out_pep)
                res.append(out_pep)
        return res

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (
            weight.new(self.lstm_layers * 2, batch_size, self.hidden_lstm_dim).zero_(),
            weight.new(self.lstm_layers * 2, batch_size, self.hidden_lstm_dim).zero_(),
        )
        return hidden

    def one_hot_tensor(self, peps):
        batch_size = len(peps)
        src = torch.zeros((batch_size, self.seq_len), dtype=torch.float16, device="cuda")
        src[peps > 0] = 1.0
        one_hots = torch.zeros(
            (batch_size, self.seq_len, self.vocab_size),
            dtype=torch.float16,
            device="cuda",
        )
        one_hots.scatter_(
            2,
            peps.view(batch_size, self.seq_len, 1),
            src.view(batch_size, self.seq_len, 1),
        )
        one_hots.requires_grad = True
        return one_hots

    def name(self):
        return "Net"


rank = 0
world_size = 1
main.setup(rank, world_size)


model_name = "512-embed-2-lstm-SnapLoss2D-80k-nist-massive-no-mc-semi-r2r2r-22.pt"
snap_model = Net(vocab_size=30, embedding_dim=512, hidden_lstm_dim=512, lstm_layers=2).to(rank)
snap_model = nn.parallel.DistributedDataParallel(snap_model, device_ids=[rank])
snap_model.load_state_dict(torch.load("./models/{}".format(model_name))["model_state_dict"])
snap_model = snap_model.module
snap_model.eval()
print(snap_model)


# Training Data Embeddings
batch_size = config.get_config(section="ml", key="batch_size")
# in_tensor_dir = config.get_config(section='preprocess', key='in_tensor_dir') # for raptor
in_tensor_dir = "/lclhome/mtari008/data/deepsnap/nist_massiv_80k_no_ch_graymass-semi"  # for comet

(
    train_peps,
    train_specs,
    train_masses,
    test_peps,
    test_specs,
    test_masses,
) = main.read_split_listings(in_tensor_dir)

np_specs = []
spec_path = join(in_tensor_dir, "spectra")

# Wrap the outer loop with tqdm
for spec_file_list in tqdm(train_specs, desc="Outer Loop"):
    for spec_file in spec_file_list:
        np_spec = np.load(join(spec_path, spec_file))
        np_specs.append(np_spec)


train_dataset = dataset.LabeledSpectra(in_tensor_dir, train_peps, train_specs)
test_dataset = dataset.LabeledSpectra(in_tensor_dir, test_peps, test_specs)

train_peps_strings, train_dpeps_strings = [], []
train_peps_masses, train_dpeps_masses = [], []
for train_pep in train_peps:
    pep_path = join(in_tensor_dir, "peptides", train_pep)
    with open(pep_path, "r") as f:
        pep = f.readlines()[0].strip()
        train_peps_masses.append(sim.get_pep_mass(pep))
        train_peps_strings.append(pep)
        dpep = train_dataset.get_decoy(pep)
        if dpep:
            train_dpeps_strings.append(dpep)
            train_dpeps_masses.append(sim.get_pep_mass(dpep))

vocab_size = train_dataset.vocab_size

train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    num_workers=8,
    collate_fn=main.psm_collate,
    # batch_sampler=train_batch_sampler
    batch_size=batch_size,
    shuffle=False,
)

test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset,
    num_workers=8,
    collate_fn=main.psm_collate,
    # batch_sampler=test_batch_sampler,
    batch_size=batch_size,
    shuffle=False,
)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.cuda.set_device(rank)

q, p, d, train_spec_labels, train_spec_charges, train_spec_masses = [], [], [], [], [], []
print("Num batches: {}".format(len(train_loader)))
for idx, data in enumerate(train_loader):
    if idx % 1000 == 0:
        print("Batch: {}".format(idx))
    q_len = len(data[0])
    p_len = len(data[1])
    d_len = len(data[2])
    if p_len > d_len:
        seq_len = config.get_config(section="ml", key="pep_seq_len")  # + charge
        zero_pad = torch.zeros(p_len - d_len, seq_len, dtype=torch.long)
        data[2] = torch.cat((data[2], zero_pad))
    data[0] = data[0].to(rank)  # spectra
    data[1] = data[1].to(rank)  # peptides
    data[2] = data[2].to(rank)  # decoys
    Q, P, D = snap_model(data[:-1])
    Q = Q.detach().cpu().numpy()
    P = P.detach().cpu().numpy()
    D = D.detach().cpu().numpy()
    q.append(Q)
    p.append(P)
    d.append(D)
    l_spec_labels = np.repeat(np.arange(data[-1].size), data[-1])
    train_spec_labels.append(l_spec_labels)
    train_spec_charges.append(data[3])
    train_spec_masses.append(data[4])

q = np.concatenate(q)
p = np.concatenate(p)
d = np.concatenate(d)
train_spec_labels = np.concatenate(train_spec_labels)
train_spec_charges = np.concatenate(train_spec_charges)
train_spec_masses = np.concatenate(train_spec_masses)


# Sort q, train_dataset.np_specs, train_spec_charges by train_spec_masses
print("Sorting data by mass...")
zipped = zip(q, np_specs, train_spec_charges, train_spec_masses)
print(len(q), len(np_specs), len(train_spec_charges), len(train_spec_masses))
sorted_zipped = sorted(zipped, key=lambda x: x[-1])
q, np_specs, train_spec_charges, train_spec_masses = zip(*sorted_zipped)

print("Saving training data embeddings...")
np.save("uncertainty_analysis/training_data/q.npy", q)
np.save("uncertainty_analysis/training_data/p.npy", p)
np.save("uncertainty_analysis/training_data/d.npy", d)
np.save("uncertainty_analysis/training_data/spec_labels.npy", train_spec_labels)
np.save("uncertainty_analysis/training_data/charges.npy", train_spec_charges)
np.save("uncertainty_analysis/training_data/masses.npy", train_spec_masses)
pickle.dump(np_specs, open("uncertainty_analysis/training_data/np_specs.pkl", "wb"))
pickle.dump(train_peps_strings, open("uncertainty_analysis/training_data/peps.pkl", "wb"))
pickle.dump(train_dpeps_strings, open("uncertainty_analysis/training_data/dpeps.pkl", "wb"))
pickle.dump(train_peps_masses, open("uncertainty_analysis/training_data/pep_masses.pkl", "wb"))
pickle.dump(train_dpeps_masses, open("uncertainty_analysis/training_data/dpep_masses.pkl", "wb"))

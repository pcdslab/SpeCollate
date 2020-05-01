import torch
import torch.nn as nn
import torch.optim as optim

from src.snapconfig import config
from src.snapprocess import model, dataset, trainmodel

# with redirect_output("deepSNAP_redirect.txtS"):
train_loss = []
test_loss = []
train_accuracy = []
test_accuracy = []
# drop_prob=0.5
# print(vocab_size)

batch_size = config.get_config(section="ml", key="batch_size")
charge = config.get_config(section='input', key='charge')
use_mods = config.get_config(section='input', key='use_mods')
filt = {'charge': charge, 'modified': use_mods}

msp_dir = "/disk/raptor/lclhome/mtari008/Proteogenomics/DeepSNAP/data/msp-labeled/"
in_tensor_dir = "/disk/raptor/lclhome/mtari008/Proteogenomics/DeepSNAP/data/train_lstm/"

train_dataset = dataset.LabeledSpectra(in_tensor_dir, filt, test=False)
test_dataset = dataset.LabeledSpectra(in_tensor_dir, filt, test=True)

train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=8)

test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=8)

do_learn = True
save_frequency = 2
lr = 0.0001
num_epochs = 200
weight_decay = 0.0001
margin = 0.2
vocab_size = train_dataset.vocab_size
# torch.manual_seed(0)
# torch.cuda.manual_seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

triplet_loss = nn.TripletMarginLoss(margin=margin, p=2, reduction='sum')
zero_tensor = torch.tensor(0.).to(device)

model_ = model.Net(device, vocab_size, output_size=512, embedding_dim=256,
                   hidden_lstm_dim=1024, lstm_layers=1).to(device)
# model.linear1_1.weight.requires_grad = False
# model.linear1_1.bias.requires_grad = False
# model.linear1_2.weight.requires_grad = False
# model.linear1_2.bias.requires_grad = False

if do_learn:  # training mode

    optimizer = optim.Adam(model_.parameters(), lr=lr,
                           weight_decay=weight_decay)
    #optimizer = optim.SGD(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        print("Epoch: {}".format(epoch))
        trainmodel.train(model_, device, train_loader, triplet_loss, optimizer)
        trainmodel.test(model_, device, test_loader, triplet_loss)

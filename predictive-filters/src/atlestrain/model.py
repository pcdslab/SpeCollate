import os
from os.path import join
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.atlesconfig import config


# adding useless comment
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.spec_size = config.get_config(section='input', key='spec_size')
        self.charge = config.get_config(section='input', key='charge')
        self.gray_len = 24
        self.max_spec_len = config.get_config(section='ml', key='max_spec_len')
        self.max_pep_len = config.get_config(section='ml', key='max_pep_len')
        self.min_pep_len = config.get_config(section='ml', key='min_pep_len')
        self.embedding_dim = config.get_config(section='ml', key='embedding_dim')
        self.num_encoder_layers = config.get_config(section='ml', key='encoder_layers')
        self.num_heads = config.get_config(section='ml', key='num_heads')
        do = config.get_config(section="ml", key="dropout")

        ################### Spectra branch ###################
        # self.spec_embedder = nn.Embedding(self.spec_size, self.embedding_dim, padding_idx=0)
        # self.int_embedder = nn.Embedding(101, self.embedding_dim, padding_idx=0)
        # self.spec_pos_encoder = PositionalEncoding(self.embedding_dim, dropout=do, max_len=self.max_spec_len)
        # encoder_layers = nn.TransformerEncoderLayer(self.embedding_dim, nhead=self.num_heads, dropout=do,
        # batch_first=True)
        # self.encoder = nn.TransformerEncoder(encoder_layers, num_layers=self.num_encoder_layers)
        # self.bn1 = nn.BatchNorm1d(num_features=self.embedding_dim * self.max_spec_len)

        self.linear1_1 = nn.Linear(self.spec_size, 1024)
        # self.linear1_1 = nn.Linear(self.spec_size, 1024)
        self.bn1 = nn.BatchNorm1d(num_features=1024)

        self.linear1_2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(num_features=512)

        self.linear1_3 = nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm1d(num_features=256)

        # Incoming branch for charge
        self.linear_ch_1 = nn.Linear(self.charge + self.gray_len, 128)
        self.bn_ch_1 = nn.BatchNorm1d(num_features=128)
        self.linear_ch_2 = nn.Linear(128, 256)
        self.bn_ch_2 = nn.BatchNorm1d(num_features=256)
        self.linear_ch_3 = nn.Linear(256, 512)

        ################### Missed cleavage branch ########
        # self.linear_miss_clv_1 = nn.Linear(512, 256)
        # self.bn_miss_clv_1 = nn.BatchNorm1d(num_features=256)

        self.linear_miss_clv_2 = nn.Linear(256, 128)
        self.bn_miss_clv_2 = nn.BatchNorm1d(num_features=128)

        self.linear_miss_clv_3 = nn.Linear(128, 3)

        ################### Modification branch ###########
        self.linear_mod_1 = nn.Linear(256, 128)
        self.bn_mod_1 = nn.BatchNorm1d(num_features=128)

        self.linear_mod_2 = nn.Linear(128, 2)

        # Spectra branch continues

        self.linear1_4 = nn.Linear(256, 128)
        self.bn4 = nn.BatchNorm1d(num_features=128)

        # self.linear_out = nn.Linear(256, self.max_pep_len - self.min_pep_len)
        self.linear_out = nn.Linear(128, 1)

        self.dropout = nn.Dropout(do)

    def forward(self, spec, chars, mask):

        # mzs = self.spec_embedder(mzs) * math.sqrt(self.embedding_dim)
        # ints = self.int_embedder(ints)
        # data = mzs + ints
        # data = self.spec_pos_encoder(data)
        # ints = self.int_embedder(ints)
        # print(mzs.shape)
        # print(ints.shape)
        # data = ints + mzs
        # print(data.shape)

        # out = self.encoder(data, src_key_padding_mask=mask)
        # out = torch.mean(out, dim=1)
        # out = out[:, -1, :]

        out = F.relu(self.bn1(self.linear1_1(spec.view(-1, self.spec_size))))
        # out = F.relu((self.linear1_1(data.view(-1, self.spec_size))))
        out = self.dropout(out)

        # out = F.relu(self.bn2(self.linear1_2(out)))
        out = self.linear1_2(out)

        ch_out = F.relu(self.bn_ch_1(self.linear_ch_1(chars.view(-1, self.charge + self.gray_len))))
        ch_out = self.dropout(ch_out)
        ch_out = F.relu(self.bn_ch_2(self.linear_ch_2(ch_out)))
        ch_out = self.dropout(ch_out)
        ch_out = self.linear_ch_3(ch_out)

        out = F.relu(self.bn2(out + ch_out))
        out = self.dropout(out)

        out = F.relu(self.bn3(self.linear1_3(out)))
        out = self.dropout(out)

        # Missed cleavage branch
        # out_clv = F.relu(self.linear_miss_clv_1(out))
        # out_clv = self.dropout(out_clv)

        out_clv = F.relu(self.linear_miss_clv_2(out))
        out_clv = self.dropout(out_clv)

        out_clv = self.linear_miss_clv_3(out_clv)

        # Mod branch
        out_mod = F.relu(self.linear_mod_1(out))
        out_mod = self.dropout(out_mod)

        out_mod = self.linear_mod_2(out_mod)

        # Spectra branch continues
        out = F.relu(self.bn4(self.linear1_4(out)))
        out = self.dropout(out)

        out = self.linear_out(out)

        return out, out_clv, out_mod

    def name(self):
        return "Net"


# taken from https://pytorch.org/tutorials/beginner/transformer_tutorial.html 6/25/2021
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # .transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

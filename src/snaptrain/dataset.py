from os.path import join
from pathlib import Path
import re

import numpy as np
import torch
from torch.utils import data
from sklearn import preprocessing

from src.snapconfig import config


class LabeledSpectra(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, dir_path, pep_file_name, spec_file_names_lists):
        'Initialization'
        
        in_path = Path(dir_path)
        assert in_path.exists()
        assert in_path.is_dir()
        self.aas            = ['_PAD'] + list(config.AAMass.keys())# + list(config.ModCHAR.values())
        self.aa2idx         = {a:i for i, a in enumerate(self.aas)}
        self.idx2aa         = {i:a for i, a in enumerate(self.aas)}
        
        self.spec_path      = join(dir_path, 'spectra')
        self.pep_path       = join(dir_path, 'peptides')
        self.num_species    = config.get_config(section='input', key='num_species')
        self.vocab_size     = len(self.aa2idx) # + self.charge + self.num_species + 1
        print("Vocabulary Size: {}".format(self.vocab_size))
        # self.vocab_size   = round(max(config.AAMass.values())) + 1
        self.spec_size      = config.get_config(section='input', key='spec_size')
        self.seq_len        = config.get_config(section='ml', key='pep_seq_len')
        
        self.pep_file_names = pep_file_name
        self.spec_file_names_lists = spec_file_names_lists # a list of lists containing spectra for each peptide
        means = np.load(join(dir_path, "means.npy"))
        stds = np.load(join(dir_path, "stds.npy"))
        self.means = torch.from_numpy(means).float()
        self.stds = torch.from_numpy(stds).float()

        
        print('dataset size: {}'.format(len(self.pep_file_names)))
        

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.pep_file_names)


    def __getitem__(self, index):
        'Generates one sample of data'
        pep_file_name = ''
        spec_file_list = []
        # Select sample
        pep_file_name = self.pep_file_names[index]
        spec_file_list = self.spec_file_names_lists[index]

        # Load spectra
        torch_spec_list = []
        for spec_file in spec_file_list:
            np_spec = np.load(join(self.spec_path, spec_file))
            ind = torch.LongTensor([[0]*np_spec.shape[1], np_spec[0]])
            val = torch.FloatTensor(np_spec[1])
            torch_spec = torch.sparse_coo_tensor(
                ind, val, torch.Size([1, self.spec_size])).to_dense()
            torch_spec = (torch_spec - self.means) / self.stds
            torch_spec_list.append(torch_spec)

        torch_spec = torch.cat(torch_spec_list, dim=0)

        # Load peptide
        pep_file_name = join(self.pep_path, pep_file_name)
        f = open(pep_file_name, "r")
        pep = f.readlines()[0].strip()
        f.close()
        
        pepl = [self.aa2idx[aa] for aa in pep]
        pepl = self.pad_left(pepl, self.seq_len)
        torch_pep = torch.tensor(pepl, dtype=torch.long)

        dpep = self.get_decoy(pep)
        if dpep:
            dpepl = [self.aa2idx[aa] for aa in dpep]
            dpepl = self.pad_left(dpepl, self.seq_len)
            torch_pep_decoy = torch.tensor(dpepl, dtype=torch.long)
        else:
            torch_pep_decoy = torch.empty(0, 0, dtype=torch.long)
        
        return torch_spec, torch_pep, torch_pep_decoy, len(torch_spec_list)
        

    def pad_left(self, arr, size):
        out = np.zeros(size)
        out[-len(arr):] = arr
        return out

    def get_decoy(self, pep):
        pep_parts = re.findall(r"([A-Z][a-z]?)", pep)
        decoy_pep = pep_parts[0] + "".join(pep_parts[-2:0:-1]) + pep_parts[-1]
        if decoy_pep != pep:
            return decoy_pep
        else:
            return []

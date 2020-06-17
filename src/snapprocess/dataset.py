import random as rand
import re
from os import listdir
from os.path import join
from pathlib import Path
import glob

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils import data

from src.snapconfig import config


class LabeledSpectra(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, dir_path, filt, test=False):
        'Initialization'
        
        in_path = Path(dir_path)
        assert in_path.exists()
        assert in_path.is_dir()

        self.aas         = ['_PAD'] + list(config.AAMass.keys())
        self.aa2idx      = {a:i for i, a in enumerate(self.aas)}
        self.idx2aa      = {i:a for i, a in enumerate(self.aas)}
        
        self.spec_path   = join(dir_path, 'spectra')
        self.pep_path    = join(dir_path, 'peptides')
        self.charge      = filt['charge'] if 'charge' in filt else config.get_config(section='input', key='charge')
        self.num_species = config.get_config(section='input', key='num_species')
        # self.vocab_size  = len(self.aa2idx) + self.charge + self.num_species + 1
        self.vocab_size  = round(max(config.AAMass.values())) + 1
        self.spec_size   = config.get_config(section='input', key='spec_size')
        self.seq_len     = config.get_config(section='ml', key='pep_seq_len')
        self.modified    = filt['modified'] if 'modified' in filt else False
        self.test_size   = config.get_config(section='ml', key='test_size')
        self.test        = test
        
        self.pep_file_names  = []
        self.spec_file_names_lists = [] # a list of lists containing spectra for each peptide
        self.load_file_names()

        split_rand_state = rand.randint(0, 1000)
        self.train_peps, self.test_peps = train_test_split(
            self.pep_file_names, test_size = self.test_size, random_state = split_rand_state, shuffle=True)
        self.train_specs, self.test_specs = train_test_split(
            self.spec_file_names_lists, test_size = self.test_size, random_state = split_rand_state, shuffle=True)
        print('dataset size: {}'.format(len(self.pep_file_names)))
        if self.test:
            print('test size: {}'.format(len(self.test_peps)))
        else:
            print('train size: {}'.format(len(self.train_peps)))
        

    def __len__(self):
        'Denotes the total number of samples'
        if self.test:
            return len(self.test_peps)
        else:
            return len(self.train_peps)


    def __getitem__(self, index):
        'Generates one sample of data'
        pep_file_name = ''
        spec_file_list = []
        # Select sample
        if self.test:
            pep_file_name = self.test_peps[index]
            spec_file_list = self.test_specs[index]
        else:
            pep_file_name = self.train_peps[index]
            spec_file_list = self.train_specs[index]

        'Load spectra'
        torch_spec_list = []
        for spec_file in spec_file_list:
            np_spec = np.load(join(self.spec_path, spec_file))
            ind = torch.LongTensor([[0]*np_spec.shape[1], np_spec[0]])
            val = torch.FloatTensor(spec[1])
            torch_spec = torch.sparse_coo_tensor(ind, val, torch.Size([1, self.spec_size]))
            torch_spec = (torch_spec.to_dense().squeeze() - 12.311) / 325.394
            torch_spec_list.append(torch_spec)

        'Load peptide'
        pep_file_name = join(self.pep_path, pep_file_name)
        f = open(pep_file_name, "r")
        pep = f.readlines()[0].strip()
        f.close()
        
        pepl = [self.aa2idx[aa] for aa in pep]
        pepl = self.pad_left(pepl, self.seq_len)
        torch_pep = torch.tensor(pepl, dtype=torch.long)
        
        return torch_spec_list, torch_pep, len(torch_spec_list)
        
    
    def apply_filter(self, file_name):
        try:
            file_parts = re.search(r"(\d+)-(\d+)-(\d+.\d+)-(\d+)-(0|1).[pt|npy]", file_name)
            charge = int(file_parts[4])
            modified = bool(int(file_parts[5]))
        except:
            print(file_name)
            print(file_parts)
        
        if ((self.charge == 0 or charge <= self.charge)
            and (self.modified or self.modified == modified)):
            return True
        
        return False
    

    def pad_left(self, arr, size):
        out = np.zeros(size)
        out[-len(arr):] = arr
        return out


    def verfiy_files(self, specs):
        'Make sure at least one spectrum file satisifies the filter.'
        for spec in specs:
            if self.apply_filter(spec):
                return True
        return False

    
    def load_file_names(self):
        'Load the peptide and corresponding spectra file names that satisfy the filter'
        for pep_file in listdir(self.pep_path):
            spec_file_pattern = join(self.spec_path, pep_file.replace('.pep', '*.npy'))
            spec_file_names = glob.glob(spec_file_pattern)
            spec_file_list = []
            for spec in spec_file_names:
                l_spec = spec.split('/')[-1]
                if self.apply_filter(l_spec):
                    spec_file_list.append(l_spec)
            if spec_file_list:
                self.pep_file_names.append(pep_file)
                self.spec_file_names_lists.append(spec_file_list)
import re
import random as rand
from os import listdir
from os import path
from os.path import join
from shutil import copyfile

import numpy as np
import torch
from torch.utils import data
from sklearn.model_selection import train_test_split

from src.snapconfig import config

class LabeledSpectra(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, dir_path, filt, test=False):
        'Initialization'
        
        self.aas         = ['_PAD'] + list(config.AAMass.keys())
        self.aa2idx      = {a:i for i, a in enumerate(self.aas)}
        self.idx2aa      = {i:a for i, a in enumerate(self.aas)}
        
        self.spec_path   = join(dir_path, 'spectra')
        self.pep_path    = join(dir_path, 'peptides')
        self.charge      = filt['charge'] if 'charge' in filt else config.get_config(section='input', key='charge')
        self.num_species = config.get_config(section='input', key='num_species')
        # self.vocab_size  = len(self.aa2idx) + self.charge + self.num_species + 1
        self.vocab_size  = round(max(config.AAMass.values())) + 1
        self.seq_len     = config.get_config(section='ml', key='pep_seq_len')
        self.modified    = filt['modified'] if 'modified' in filt else False
        self.test_size   = config.get_config(section='ml', key='test_size')
        self.test        = test
        
        self.file_names  = []
        for file in listdir(self.spec_path):
            if self.apply_filter(file):
                self.file_names.append(file)
        
        print('dataset size: {}'.format(len(self.file_names)))        
        
        self.train_files, self.test_files = train_test_split(
            self.file_names, test_size = self.test_size, random_state = rand.randint(0, 1000), shuffle = True)
        
        if self.test:
            print('test size: {}'.format(len(self.test_files)))
        else:
            print('train size: {}'.format(len(self.train_files)))
        
    def __len__(self):
        'Denotes the total number of samples'
        if self.test:
            return len(self.test_files)
        else:
            return len(self.train_files)

    def __getitem__(self, index):
        'Generates one sample of data'
        file_name = ''
        # Select sample
        if self.test:
            file_name = self.test_files[index]
        else:
            file_name = self.train_files[index]

        spec_file_name = join("/scratch/train_lstm/spectra",  file_name)
        pep_file_name  = join("/scratch/train_lstm/peptides", file_name.replace('.pt', '.pep'))

        # if not path.isfile(spec_file_name):
        #     src = join(self.spec_path, file_name)
        #     dst = spec_file_name
        #     copyfile(src, dst)
        
        # if not path.isfile(pep_file_name):
        #     src = join(self.pep_path, file_name.replace('.pt', '.pep'))
        #     dst = pep_file_name
        #     copyfile(src, dst)

        # spec_file_name = join(self.spec_path, spec_file_name)
        # pep_file_name  = join(self.pep_path, pep_file_name)
        
        # Load data and get label
        spec_torch = torch.load(spec_file_name)
        
        # Load peptide and convert to idx array
        f = open(pep_file_name, "r")
        pep = f.readlines()[0].strip()
        f.close()
        
        pepl = np.zeros(len(pep))
        file_parts = re.search(r"(\d+)-(\d+)-(\d+.\d+)-(\d)-(0|1).pt", file_name)
        pepl[0] = int(file_parts[4]) + len(self.aas)  # coded value of charge
        pepl[1] = int(file_parts[2]) + self.charge + 1 + len(self.aas) # coded value of specie id
        
        # for i in range(2, len(pep)):
        #     pepl[i] = self.aa2idx[pep[i]]
        for i, aa in enumerate(pep[2:]):
            pepl[i + 2] = self.aa2idx[aa]
            # pepl[i + 2] = round(config.AAMass[aa])
        
        pepl = self.pad_left(pepl, self.seq_len)
        pep_torch = torch.tensor(pepl, dtype=torch.long)
        
        return [spec_torch, pep_torch]
    
    def apply_filter(self, file_name):
        file_parts = re.search(r"(\d+)-(\d+)-(\d+.\d+)-(\d)-(0|1).pt", file_name)
        charge = int(file_parts[4])
        modified = bool(int(file_parts[5]))
        
        if ((self.charge == 0 or charge <= self.charge)
            and (self.modified or self.modified == modified)):
            return True
        
        return False
    
    def pad_left(self, arr, size):
        out = np.zeros(size)
        out[-len(arr):] = arr
        return out
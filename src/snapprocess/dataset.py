import random as rand
import re
from os import listdir
from os.path import join
from pathlib import Path

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
        
        self.file_names  = []
        for file in listdir(self.spec_path):
            #print(file)
            if self.apply_filter(file):
                self.file_names.append(file)
        
        print('dataset size: {}'.format(len(self.file_names)))        
        
        self.train_files, self.test_files = train_test_split(
            self.file_names, test_size = self.test_size, random_state = rand.randint(0, 1000), shuffle=True)
        
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

        ext = '.' + file_name.split('.')[-1]

        spec_file_name = join(self.spec_path, file_name)
        pep_file_name = join(self.pep_path,
                             file_name.replace(ext, '.pep'))
        
        # Load data and get label
        if ext == '.pt':
            spec_torch = torch.load(spec_file_name)
        elif ext == '.npy':
            spec = np.load(spec_file_name)
            ind = torch.LongTensor([[0]*spec.shape[1], spec[0]])
            val = torch.FloatTensor((spec[1]))
            spec_torch = torch.sparse_coo_tensor(ind, val, torch.Size([1, self.spec_size]))
        #if "torch.sparse" in spec_torch.type():
            # dense tensor. subtract mean and divide by std
            #spec_torch = (spec_torch.to_dense().squeeze() - 13.007) / 339.345
            spec_torch = (spec_torch.to_dense().squeeze() - 12.311) / 325.394

        # Load peptide and convert to idx array
        f = open(pep_file_name, "r")
        pep = f.readlines()[0].strip()
        f.close()
        
        pepl = np.zeros(len(pep) + 0)
        file_parts = re.search(r"(\d+)-(\d+)-(\d+.\d+)-(\d)-(0|1).[pt|npy]", file_name)
        #pepl[0] = int(file_parts[4]) + len(self.aas)  # coded value of charge
        #pepl[1] = int(file_parts[2]) + self.charge + 1 + len(self.aas) # coded value of specie id
        
        # for i in range(2, len(pep)):
        #     pepl[i] = self.aa2idx[pep[i]]
        for i, aa in enumerate(pep):
            pepl[i+0] = self.aa2idx[aa]
            # pepl[i + 2] = round(config.AAMass[aa])
        
        pepl = self.pad_left(pepl, self.seq_len)
        pep_torch = torch.tensor(pepl, dtype=torch.long)
        
        return [spec_torch, pep_torch]
    
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

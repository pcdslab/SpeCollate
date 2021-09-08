import sys
from os.path import join
from pathlib import Path
import re

import numpy as np
from torch.utils import data
import torch
from tqdm import tqdm

from src.snapconfig import config
from src.snaputils import preprocess as prep


class SpectralDataset(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, dir_path):
        'Initialization'
        print(dir_path)
        in_path = Path(dir_path)
        assert in_path.exists()
        assert in_path.is_dir()
        
        self.spec_path = join(dir_path, "spectra")
        self.spec_size = config.get_config(section='input', key='spec_size')
        
        self.means = torch.from_numpy(np.load(join(dir_path, "means.npy"))).float()
        self.stds  = torch.from_numpy(np.load(join(dir_path, "stds.npy"))).float()
        
        spec_ids, spec_lst, spec_mass_lst, spec_charge_lst = load_specs(self.spec_path)
        all_sorts = list(zip(*sorted(zip(spec_ids, spec_lst, spec_mass_lst, spec_charge_lst), key=lambda x: x[2])))
        self.spec_ids         = all_sorts[0]
        self.spec_list        = all_sorts[1]
        self.spec_mass_list   = all_sorts[2]
        self.spec_charge_list = all_sorts[3]
        print('Spectral dataset size: {}'.format(len(self.spec_list)))
        

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.spec_list)


    def __getitem__(self, index):
        'Generates one sample of data'

        # Load spectra
        np_spec = self.spec_list[index]
        ind = torch.LongTensor([[0]*np_spec.shape[1], np_spec[0]])
        val = torch.FloatTensor(np_spec[1])
        torch_spec = torch.sparse_coo_tensor(
            ind, val, torch.Size([1, self.spec_size])).to_dense()
        self.means[:32] = 0.0
        self.stds[:32] = 1.0
        torch_spec = (torch_spec - self.means) / self.stds

        return torch_spec


def load_specs(spec_dir):
    spec_size = config.get_config(key="spec_size", section="input")
    charge = config.get_config(key="charge", section="search")
    spec_files = prep.verify_in_dir(spec_dir, "npy")
    spec_ids = []
    spec_list = []
    masses = []
    charges = []
    count = 0

    pbar = tqdm(spec_files, file=sys.stdout)
    pbar.set_description('Loading Spectra...')
    # with progressbar.ProgressBar(max_value=len(spec_files)) as bar:
    for spec_file in pbar:
        file_name = spec_file.split('/')[-1]
        file_parts = re.search(r"(\d+)-(\d+.\d+)-(\d+).[pt|npy]", file_name)
        spec_id = int(file_parts[1])
        mass = round(float(file_parts[2]), 2)
        l_charge = int(file_parts[3])
        if l_charge > charge:
            continue
        spec_ids.append(spec_id)
        np_spec = np.load(spec_file)
        spec_list.append(np_spec)
        masses.append(mass)
        charges.append(l_charge)

        count += 1
        # bar.update(i)
    # print("count: {}".format(count))
    return spec_ids, spec_list, masses, charges
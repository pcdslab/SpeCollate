from os.path import join
from pathlib import Path
import re

import numpy as np
import torch
from torch.utils import data

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
        self.charge         = config.get_config(section='input', key='charge')
        self.charge2idx     = {i+1:i for i in range(self.charge)}
        self.idx2charge     = {i:i+1 for i in range(self.charge)}
        
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

        # aux_means = np.load(join(dir_path, "aux_means.npy"))
        # aux_stds = np.load(join(dir_path, "aux_stds.npy"))
        # self.aux_means = torch.from_numpy(aux_means).float()
        # self.aux_stds = torch.from_numpy(aux_stds).float()

        # db_peps_path = config.get_config(section="input", key="db_peps_path")
        # with open(db_peps_path, 'rb') as fp:
        #     self.db_peps = pickle.load(fp)
        
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
        torch_spec_charge_list = []
        torch_spec_mass_list = []
        for spec_file in spec_file_list:
            file_parts = re.search(r"(\d+)-(\d+)-(\d+.\d+)-(\d+)-(\d+).[pt|npy]", spec_file)
            spec_charge = int(file_parts[4])
            spec_mass = float(file_parts[3])
            np_spec = np.load(join(self.spec_path, spec_file))
            ind = torch.LongTensor([[0]*np_spec.shape[1], np_spec[0]])
            val = torch.FloatTensor(np_spec[1])

            # coin_toss = random.randrange(0,2,1)
            # if coin_toss == 1:
            #     zero_inds = random.sample(range(len(val)), int(len(val)*0.1))
            #     val[zero_inds] = 0.0
            
            torch_spec = torch.sparse_coo_tensor(
                ind, val, torch.Size([1, self.spec_size])).to_dense()
            # self.means[:32] = 0.0
            # self.stds[:32] = 1.0
            torch_spec = (torch_spec - self.means) / self.stds
            # torch_spec[1:10] = 0.
            torch_spec_list.append(torch_spec)
            torch_spec_charge_list.append(spec_charge)
            torch_spec_mass_list.append(spec_mass)
        
        torch_spec_charge_list = [self.charge2idx[charge] for charge in torch_spec_charge_list]
        torch_spec = torch.cat(torch_spec_list, dim=0)
        torch_spec_charge = torch.LongTensor(torch_spec_charge_list)
        torch_spec_mass = torch.FloatTensor(torch_spec_mass_list)

        # Load peptide
        pep_file_name = join(self.pep_path, pep_file_name)
        f = open(pep_file_name, "r")
        pep = f.readlines()[0].strip()
        f.close()
        
        pepl = [self.aa2idx[aa] for aa in pep]
        # torch_pep = self.one_hot_tensor(pepl) # no need to call pad after this.
        pepl = self.pad_left(pepl, self.seq_len)

        # pep_mass = sim.get_pep_mass(pep)
        # gray_mass = sim.gray_code(round(pep_mass * 1000))
        # mass_arr = sim.decimal_to_binary_array(gray_mass, 27)
        # pepl = np.concatenate((mass_arr, pepl))

        # pep_len = np.zeros(64)
        # missed_cleavs_vec = np.zeros(17)
        # aas = np.zeros(30)
        # pep_len[len(pep)] = 1
        # missed_cleavs = (pep.count("K") + pep.count("R")) - (pep.count("KP") + pep.count("RP")) - 1 # might not be correct for non-tryp peps
        # missed_cleavs_vec[missed_cleavs] = 1
        # for a in pep:
        #     aa_id = self.aa2idx[a]
        #     aas[aa_id] += 1
        # concat_vec = np.concatenate((pep_len, missed_cleavs_vec, aas))
        # concat_vec = (concat_vec - self.aux_means) / self.aux_stds
        # pepl = np.concatenate((concat_vec, pepl))
        
        torch_pep = torch.tensor(pepl, dtype=torch.long)
        dpep = self.get_decoy(pep)
        # dpep = random.choice(self.db_peps[round(pep_mass*10)]) if self.db_peps[round(pep_mass*10)] else self.get_decoy(pep)
        if dpep:
            dpepl = [self.aa2idx[aa] for aa in dpep]
            # torch_pep_decoy = self.one_hot_tensor(dpepl) # no need to call pad after this.
            dpepl = self.pad_left(dpepl, self.seq_len)
            # dpepl = np.concatenate((mass_arr, dpepl))
            # dpepl = np.concatenate((concat_vec, dpepl))
            torch_pep_decoy = torch.tensor(dpepl, dtype=torch.long)
        else:
            torch_pep_decoy = torch.empty(0, 0, dtype=torch.long)
        # return (torch_spec, torch_pep, torch_pep_decoy, torch_spec_charge,
        #     torch_spec_mass, pep_mass, len(torch_spec_list))
        return (torch_spec, torch_pep, torch_pep_decoy, torch_spec_charge, len(torch_spec))

    def pad_left(self, arr, size):
        out = np.zeros(size)
        out[-len(arr):] = arr
        return out


    # TODO: Needs fixing for multiple n-term mods
    def get_decoy(self, pep):
        first = ""
        if pep[0].islower():
            first = pep[0]
            pep = pep[1:]
        
        pep_parts = re.findall(r"([A-Z][a-z]?)", pep)
        decoy_pep = pep_parts[0] + "".join(pep_parts[-2:0:-1]) + pep_parts[-1] # peptide reverse
        # middle = pep_parts[1:-1]
        # random.shuffle(middle)
        # decoy_pep = pep_parts[0] + "".join(middle) + pep_parts[-1] # peptide shuffle
        decoy_pep = first + decoy_pep
        if decoy_pep == pep:
            decoy_pep = []

        return decoy_pep


    # def one_hot_tensor(self, pep):
    #     one_hots = torch.zeros((len(pep), self.vocab_size), dtype=torch.long)
    #     zeros = torch.zeros(((self.seq_len - len(pep)), self.vocab_size), dtype=torch.long)
    #     index = torch.tensor(pep)
    #     one_hots.scatter_(1, index.view(-1, 1), 1)
    #     return torch.cat((zeros, one_hots), dim=0)


import sys
import re
from pathlib import Path

import numpy as np
import progressbar
import torch
from torch.utils import data
from tqdm import tqdm
from os.path import join

from src.atlesconfig import config
from src.atlespredict import preprocess
from src.atlesutils import simulatespectra as sim


class PeptideDataset(data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, dir_path, fine_name=None, decoy=False):
        'Initialization'

        in_path = Path(dir_path)
        assert in_path.exists()
        assert in_path.is_dir()

        self.aas = ['_PAD'] + list(config.AAMass.keys())  # + list(config.ModCHAR.values())
        self.aa2idx = {a: i for i, a in enumerate(self.aas)}
        self.idx2aa = {i: a for i, a in enumerate(self.aas)}

        self.pep_path = dir_path
        self.vocab_size = len(self.aa2idx)  # + self.charge + self.num_species + 1
        print("Vocabulary size: {}".format(self.vocab_size))
        self.seq_len = config.get_config(section='ml', key='pep_seq_len')

        print("Loading peptides...")
        # pep_lst, prot_list, pep_mass_lst, pep_modified_lst = load_peps(self.pep_path)
        out = load_peps(self.pep_path, fine_name, decoy=decoy)

        print("peptide list len: {}".format(len(out)))
        # print("peptide set len: {}".format(len(self.pep_lst_set)))
#         out_dir = "/disk/raptor-2/mtari008/data/deepsnap/preprocessed-human-hcd-tryp-best/pts/"
#         with open(join(out_dir, 'pep_pickle.pkl'), 'rb') as f:
#             search_peps = pickle.load(f)
#         added_counter = 0
#         for s_pep in search_peps:
#             if s_pep not in self.pep_lst_set:
#                 self.pep_lst_set.add(s_pep)
#                 added_counter += 1
#                 pep_lst.append(s_pep)
#                 prot_list.append("unknown")
#                 pep_mass_lst.append(sim.get_pep_mass(s_pep))
#                 pep_modified_lst.append(any(aa.islower() for aa in s_pep))
#         print("New peptides added: {}".format(added_counter))

        print("Sorting peptides...")
        # all_sorts = list(zip(*sorted(zip(pep_lst, prot_list, pep_mass_lst, pep_modified_lst), key=lambda x: x[2])))
        out.sort(key=lambda x: x[2])
        self.pep_list, self.prot_list, self.pep_mass_list, self.pep_modified_list = zip(*out)
        self.pep_lst_set = set(self.pep_list)
        # self.pep_list = all_sorts[0]
        # self.prot_list = all_sorts[1]
        # self.pep_mass_list = all_sorts[2]
        # self.pep_modified_list = all_sorts[3]
        print("Peptides sorted.")
        print("Getting missed cleavages...")
        self.missed_cleavs = []
        for pep in self.pep_list:
            miss_clvs = (pep.count("K") + pep.count("R")) - (pep.count("KP") + pep.count("RP"))
            if pep[-1] == 'K' or pep[-1] == 'R':
                miss_clvs -= 1
            self.missed_cleavs.append(miss_clvs)
        if decoy:
            print("Generating decoy database...")
            self.pep_list, self.prot_list, self.pep_mass_list, self.pep_modified_list, self.missed_cleavs = \
                self.get_docoys()

        print('{} Peptide Dataset Size: {}'.format("Decoy" if decoy else "Target", len(self.pep_list)))

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.pep_list)

    def __getitem__(self, index):
        'Generates one sample of data'
        pep = self.pep_list[index].strip()
        pepl = [self.aa2idx[aa] for aa in pep]
        pepl = self.pad_left(pepl)

        # pep_mass = sim.get_pep_mass(pep)
        # gray_mass = sim.gray_code(round(pep_mass * 100))
        # mass_arr = sim.decimal_to_binary_array(gray_mass, 24)
        # pepl = np.concatenate((mass_arr, pepl))

        torch_pep = torch.tensor(pepl, dtype=torch.long)
        return torch_pep

    def pad_left(self, arr):
        out = np.zeros(self.seq_len)
        out[-len(arr):] = arr
        return out

    def get_docoys(self):
        decoy_list = []
        decoy_prot_list = []
        decoy_mass_list = []
        decoy_modified_list = []
        decoy_miss_clvs = []
        for pep, prot, mass, modified, miss_clv in zip(
                self.pep_list, self.prot_list, self.pep_mass_list, self.pep_modified_list, self.missed_cleavs):

            pep_parts = re.findall(r"([A-Z][a-z]?)", pep)
            decoy_pep = pep_parts[0] + "".join(pep_parts[-2:0:-1]) + pep_parts[-1]
            if decoy_pep not in self.pep_lst_set:
                decoy_list.append(decoy_pep)
                decoy_prot_list.append(prot)
                decoy_mass_list.append(mass)
                decoy_modified_list.append(modified)
                decoy_miss_clvs.append(miss_clv)
        return decoy_list, decoy_prot_list, decoy_mass_list, decoy_modified_list, decoy_miss_clvs


### These are not class functions. Use them normally. ###

def find_occurrences(s, ch):
    return [i for i, letter in enumerate(s) if letter == ch]


def apply_mod(peps, mod):
    result_peps = set()
    pep_seq_len = config.get_config(section="ml", key="pep_seq_len")
    for pep in peps:
        if len(pep) >= pep_seq_len:
            continue
        for mod_aa in mod["aas"]:
            if mod_aa == "nt" and not pep[1].islower():
                result_peps.update([pep[0] + mod["mod_char"] + pep[1:]])
            elif mod_aa == "ct" and not pep[-1].islower():
                result_peps.update([pep + mod["mod_char"]])
            else:
                aa_indices = find_occurrences(pep, mod_aa)
                for index in aa_indices:
                    if index == len(pep) - 1 or not pep[index + 1].islower():
                        result_peps.update([pep[: index + 1] + mod["mod_char"] +
                                            (pep[index + 1:] if index < len(pep) else "")])

    return result_peps


def add_mods(pep, mods, num_mods):
    mod_peps = set([pep])
    result_peps = set([pep])
    for i in range(num_mods):
        temp_mod_peps = set()
        for mod in mods:
            temp_mod_peps.update(apply_mod(mod_peps, mod))
        mod_peps.update(set(temp_mod_peps))
        result_peps.update(mod_peps)

    return result_peps


def load_peps(pep_dir, file_name=None, decoy=False):
    if file_name:
        fasta_files = [join(pep_dir, file_name)]
    else:
        fasta_files = preprocess.verify_in_dir(pep_dir, "fasta")

    use_mods = config.get_config(key="use_mods", section="input")
    mods_list = config.Mods
    num_mods = config.get_config(key="num_mods", section="search")
    pep_seq_len = config.get_config(key="pep_seq_len", section="ml")

    pep_set = set()
    pep_list = []
    masses = []
    modifieds = []
    prot_list = []

    tot_pep_count = 0
    for fasta_file in fasta_files:
        tqdm.write('Reading: {}'.format(fasta_file))

        f = open(fasta_file, "r")
        lines = f.readlines()
        f.close()

        # print("File length: {}".format(len(lines)))
        peps = []
        temp_prot = ""
        pbar = tqdm(lines, file=sys.stdout)
        pbar.set_description('Loading {}'.format('Decoys...' if decoy else 'Peptides...'))
        out = []
        # with progressbar.ProgressBar(max_value=len(lines)) as bar:
        for line in pbar:
            line = line.strip().replace("C", "Cc")
            if line.startswith(">"):
                temp_prot = line[1:].strip()
                continue
            elif any(x in line for x in config.Ignore):
                continue
            peps = add_mods(line, mods_list, num_mods) if use_mods else [line]
            for pep in peps:
                pep = pep.strip()
                mass = sim.get_pep_mass(pep)
                modified = any(aa.islower() for aa in pep if aa != "c")
                if pep not in pep_set and len(pep) <= pep_seq_len:
                    out.append((pep, temp_prot, mass, modified))
                    pep_set.add(pep)
                    # pep_list.append(pep)
                    # masses.append(mass)
                    # modifieds.append(modified)
                    # prot_list.append(temp_prot)
                    tot_pep_count += 1
            # bar.update(i)
        tqdm.write("Peptides written: {}".format(tot_pep_count))

        return out

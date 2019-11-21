import random as rand
import numpy as np
from heapq import merge

import snapconfig.config as config


def get_rand_mod(seq, num_mods=1):
    """
    Get num_mods number of random modifications added to peptide seq.
    :param seq: str
    :param num_mods: int
    :return: str
    """
    aas = list(config.AAMass.keys())
    res = temp = seq
    for i in range(num_mods):
        while res == temp:
            rand_indx = rand.randint(0, len(seq) - 1)
            rand_mod = aas[rand.randint(0, len(aas)) - 1]
            temp = temp[:rand_indx] + rand_mod + temp[rand_indx + 1:]
        res = temp
    return res


def get_aa_mass(aa):
    """
    Get amino acid mass from the given aa character.
    :param aa: char
    :return: float
    """
    return config.AAMass[aa] + 57.021464 if aa == 'C' else config.AAMass[aa]


def get_spectrum(seq):
    """
    Get theoretical spectrum from a peptide string seq.
    :param seq: str
    :return: int[]
    """

    spec_size = config.get_config(section='input', key='spec_size')

    if len(seq) == 0:
        print('Error: seq length is zero.')
        return

    b_spectrum = []
    y_spectrum = []

    b_spectrum.append(get_aa_mass(seq[0]) + config.PROTON)
    y_spectrum.append(get_aa_mass(seq[-1]) + config.H2O + config.PROTON)

    for i, (faa, baa) in enumerate(zip((seq[1:]), seq[-2::-1])):
        b_spectrum.append(b_spectrum[i] + get_aa_mass(faa))
        y_spectrum.append(y_spectrum[i] + get_aa_mass(baa))

    merged_out = list(merge(b_spectrum, y_spectrum))
    if merged_out[-1] > spec_size:
        print('Error: peptide mass {} is larger than {}'.format(merged_out[-1], spec_size))
        print(seq)
    t_spec = np.zeros(spec_size)
    t_spec[np.rint(merged_out).astype(int)] = 1
    return t_spec

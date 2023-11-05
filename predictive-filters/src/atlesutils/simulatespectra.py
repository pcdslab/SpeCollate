import re
import random as rand
from heapq import merge
import numpy as np

from sklearn import preprocessing

from src.atlesconfig import config


def get_rand_mod(seq, num_mods=1):
    """
    Get num_mods number of random modifications added to peptide seq.
    :param seq: str
    :param num_mods: int
    :return: str
    """
    aas = list(config.AAMass.keys())
    res = temp = seq
    for _ in range(num_mods):
        while res == temp:
            rand_indx = rand.randint(0, len(seq) - 1)
            rand_mod = aas[rand.randint(0, len(aas)) - 1]
            temp = temp[:rand_indx] + rand_mod + temp[rand_indx + 1:]
        res = temp
    return res


def pad_right(lst, max_len):
    lst_len = len(lst)
    zeros = [0] * (max_len - lst_len)
    return list(lst) + zeros


def gray_code(num):
    return num ^ (num >> 1)


def decimal_to_binary_array(num, arr_len):
    bin_arr = [float(i) for i in list('{0:0b}'.format(num))]
    assert len(bin_arr) <= arr_len
    res = [0.] * (arr_len - len(bin_arr)) + bin_arr
    # greater than zero. 0.1 for the floating pointing errors.
    # inds = [int(i) for i, _ in enumerate(res) if res[i] > 0.1]
    return res


def get_aa_mass(aa):
    """
    Get amino acid mass from the given aa character.
    :param aa: char
    :return: float
    """
    return config.AAMass[aa] + 57.021464 if aa == 'C' else config.AAMass[aa]

def get_mod_aa_mass(aa):
    """
    Get amino acid mass from the given (modified/unmodified) aa.
    :param aa: char/s
    :return: float
    """
    return sum(config.AAMass[sub_aa] for sub_aa in aa)


def get_pep_mass(pep):
    """
    Get peptide mass from the given pep string.
    :param pep: str
    :return: float
    """
    return sum(config.AAMass[aa] for aa in pep) + config.H2O


def get_spectrum(seq):
    """
    Get theoretical spectrum from a peptide string seq.
    :param seq: str
    :return: int[]
    """

    spec_size = config.get_config(section='input', key='spec_size')
    # charge = config.get_config(section='input', key='charge')

    if len(seq) == 0:
        print('Error: seq length is zero.')
        return []

    first = ""
    if seq[0].islower():
        first = seq[0]
        seq = seq[1:]

    pep_parts = re.findall(r"([A-Z][a-z]?)", seq)
    pep_parts[0] = first + pep_parts[0]

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


def get_mod_spectrum(seq):
    """
    Get theoretical spectrum from a peptide string seq.
    :param seq: str
    :return: int[]
    """

    spec_size = config.get_config(section='input', key='spec_size')
    # charge = config.get_config(section='input', key='charge')

    if len(seq) == 0:
        print('Error: seq length is zero.')
        return []

    first = ""
    if seq[0].islower():
        first = seq[0]
        seq = seq[1:]

    pep_parts = re.findall(r"([A-Z][a-z]?)", seq)
    pep_parts[0] = first + pep_parts[0]

    b_spectrum = []
    y_spectrum = []

    b_spectrum.append(get_mod_aa_mass(pep_parts[0]) + config.PROTON)
    y_spectrum.append(get_mod_aa_mass(pep_parts[-1]) + config.H2O + config.PROTON)

    for i, (faa, baa) in enumerate(zip((seq[1:]), pep_parts[-2::-1])):
        b_spectrum.append(b_spectrum[i] + get_mod_aa_mass(faa))
        y_spectrum.append(y_spectrum[i] + get_mod_aa_mass(baa))

    merged_out = list(merge(b_spectrum, y_spectrum))
    if merged_out[-1] > spec_size:
        print('Error: peptide mass {} is larger than {}'.format(merged_out[-1], spec_size))
        print(seq)

    return merged_out


def get_mod_spectrum_hyperscore(seq):
    """
    Get theoretical spectrum from a peptide string seq.
    :param seq: str
    :return: int[]
    """

    if len(seq) == 0:
        print('Error: seq length is zero.')
        return []

    first = ""
    if seq[0].islower():
        first = seq[0]
        seq = seq[1:]

    pep_parts = re.findall(r"([A-Z][a-z]?)", seq)
    pep_parts[0] = first + pep_parts[0]

    b_spectrum = []
    y_spectrum = []

    b_spectrum.append(get_mod_aa_mass(pep_parts[0]) + config.PROTON)
    y_spectrum.append(get_mod_aa_mass(pep_parts[-1]) + config.H2O + config.PROTON)

    for i, (faa, baa) in enumerate(zip((seq[1:]), pep_parts[-2::-1])):
        b_spectrum.append(b_spectrum[i] + get_mod_aa_mass(faa))
        y_spectrum.append(y_spectrum[i] + get_mod_aa_mass(baa))

    return b_spectrum, y_spectrum


def fasta_to_spectra(lines, start, count, dh=None):
    t_spectra = []
    masses = []
    peps = []

    prev = 0
    end = min(start + count, len(lines))
    for i, line in enumerate(lines[start:end]):
        splits = line.split('\t')

        pep = splits[0]
        # print(pep)
        peps.append(pep)
        spec = get_spectrum(pep)
        t_spectra.append(preprocessing.scale(spec))
        masses.append(float(splits[1]))

        # print(splits[1])
        # Progress Monitor
        new = int(((i + start) / len(lines)) * 100)
        if dh and new > prev:
            dh.update(str(new) + '%')
            prev = new

    return t_spectra, masses, peps

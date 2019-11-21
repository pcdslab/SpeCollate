import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile
import numpy as np
import random as rand
import queue
import csv
from collections import OrderedDict
from IPython.display import clear_output
import csv
from heapq import merge
from sklearn import preprocessing
import gc

from src.snapconfig import config
from src.snapprocess import simulatespectra as sim


def read_msp_with_decoy(mspfile, charge, use_mods):
    """Read annotated spectra from msp file and return
    data structure along with decoy peptides.
    :param mspfile: str
    :param charge: int
    :param use_mods: bool
    :returns list
    """

    f = open(mspfile, "r")
    lines = f.readlines()
    f.close()

    dataset = []
    label = []
    spec_size = config.get_config(section='input', key='spec_size')
    print('len of file: ' + str(len(lines)))
    count = 0
    limit = 200000
    pep = 0
    spec = []
    is_name = is_mw = is_num_peaks = False
    prev = 0
    max_peaks = max_moz = 0
    i = 0
    while i < len(lines) and limit > 0:
        line = lines[i]
        i += 1
        splits = line.split(':')
        if (splits[0] == 'Name') and '_' in line:
            split1 = splits[1]
            l_charge = int(split1[split1.find('_') - 1])
            if l_charge != charge:  # l_charge == l_charge always true.
                continue
            if use_mods:
                pep = split1.split('/')[0].lstrip(' ')
                is_name = True
            elif '(' not in splits[1] and ')' not in splits[1]:
                pep = split1.split('/')[0].lstrip(' ')
                is_name = True

        if is_name and splits[0] == 'MW':
            mass = float(splits[1])
            if round(mass) < spec_size:
                is_mw = True
                # limit = limit - 1
            else:
                is_name = is_mw = is_num_peaks = False
                continue

        if is_name and is_mw and splits[0] == 'Num peaks':
            num_peaks = int(splits[1])
            if num_peaks > max_peaks:
                max_peaks = num_peaks

            spec = np.zeros(spec_size)
            while lines[i] != '\n':
                mz_line = lines[i]
                i += 1
                mz_splits = mz_line.split('\t')
                moz, intensity = float(mz_splits[0]), float(mz_splits[1])
                if moz > max_moz:
                    max_moz = moz
                spec[round(moz)] += round(intensity)

            spec = np.clip(spec, None, 1000.0)
            spec = preprocessing.scale(spec)

            is_num_peaks = True

        if is_name and is_mw and is_num_peaks:
            is_name = is_mw = is_num_peaks = False
            # revPep = pep[0] + pep[1:-1][::-1] + pep[-1]
            revPep = sim.get_rand_mod(pep)
            if pep == revPep:
                print('decoy is the same. shuffling')
                # revPep = ''.join(rand.sample(revPep,len(revPep)))
                revPep = sim.get_rand_mod(pep, len(pep))
                print(pep)
                print(revPep)
            t_spec = preprocessing.scale(sim.get_spectrum(pep))
            rt_spec = preprocessing.scale(sim.get_spectrum(revPep))

            dataset.append([spec, t_spec, rt_spec])
            label.append([1, -1])

            count = count + 1
            pep = 0
            spec = []
            new = int((i / len(lines)) * 100)
            if new > prev:
                # clear_output(wait=True)
                print(str(new) + '%')
                prev = new

    print('max peaks: ' + str(max_peaks))
    print('count: ' + str(count))
    print('max moz: ' + str(max_moz))
    return dataset, label

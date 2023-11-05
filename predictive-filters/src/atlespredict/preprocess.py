import math
from os import listdir
from os.path import isfile, join
from pathlib import Path
import shutil
import re

import numpy as np
import torch

from ..atlesconfig import config


def create_out_dir(dir_path, exist_ok=True):
    out_path = Path(dir_path)
    if out_path.exists() and out_path.is_dir():
        if not exist_ok:
            shutil.rmtree(out_path)
            out_path.mkdir()
    else:
        out_path.mkdir()

    Path(join(out_path, 'spectra')).mkdir()
    Path(join(out_path, 'peptides')).mkdir()


def verify_in_dir(dir_path, ext, ignore_list=[]):
    in_path = Path(dir_path)
    assert in_path.exists() and in_path.is_dir()

    files = [join(dir_path, f) for f in listdir(dir_path) if isfile(join(dir_path, f))
             and not f.startswith('.') and f.split('.')[-1] == ext and f not in ignore_list]

    assert len(files) > 0
    return files


def isfloat(str_float):
    try:
        float(str_float)
        return True
    except ValueError:
        return False


def mod_repl(match):
    lookup = str(round(float(match.group(0)), 2))
    return config.ModCHAR[lookup] if lookup in config.ModCHAR else ""


def mod_repl_2(match):
    return '[' + str(round(float(match.group(0)), 2)) + ']'


def preprocess_mgfs(mgf_dir, out_dir):

    mgf_files = verify_in_dir(mgf_dir, "mgf")
    create_out_dir(out_dir, exist_ok=False)

    print('reading {} files'.format(len(mgf_files)))

    spec_size = config.get_config(section='input', key='spec_size')
    charge = config.get_config(section='input', key='charge')
    use_mods = config.get_config(section='input', key='use_mods')
    num_species = config.get_config(section='input', key='num_species')
    seq_len = config.get_config(section='ml', key='pep_seq_len')

    ch = np.zeros(20)
    modified = 0
    unmodified = 0
    unique_pep_set = set()

    pep_dict = {}
    idx_spec_map = []
    pep_spec = []
    pep_idx = 0

    summ = np.zeros(spec_size)
    sq_sum = np.zeros(spec_size)
    N = 0

    tot_count = 0
    max_peaks = max_moz = 0
    for species_id, mgf_file in enumerate(mgf_files):
        print('Reading: {}'.format(mgf_file))

        f = open(mgf_file, "r")
        lines = f.readlines()
        f.close()

        count = lcount = 0

        pep_list = []
        dataset = []
        label = []

        mass_ign = 0
        pep_len_ign = 0
        dup_ign = 0

        print('len of file: ' + str(len(lines)))
        limit = 200000
        pep = []
        spec = []
        pep_set = set()
        is_name = is_mw = is_charge = False
        prev = 0
        i = 0
        while i < len(lines) and limit > 0:
            line = lines[i]
            i += 1

            if line.startswith('PEPMASS'):
                count += 1
                mass = float(re.findall(r"PEPMASS=([-+]?[0-9]*\.?[0-9]*)", line)[0])
                is_mw = True

            if is_mw and line.startswith('CHARGE'):
                l_charge = int(re.findall(r"CHARGE=([-+]?[0-9]*\.?[0-9]*)", line)[0])
                is_charge = True
                mass = (mass - config.PROTON) * l_charge

            if is_mw and is_charge:

                ind = [0]  # setting the precision to one decimal point.
                val = [0]
                for ch_val in range(l_charge):
                    ind.append(ch_val + 1)
                    val.append(0)

                while not isfloat(re.split(' |\t|=', lines[i])[0]):
                    i += 1
                num_peaks = 0
                while 'END IONS' not in lines[i].upper():
                    if lines[i] == '\n':
                        i += 1
                        continue
                    mz_line = lines[i]
                    i += 1
                    num_peaks += 1
                    mz_splits = re.split(' |\t', mz_line)
                    moz, intensity = float(mz_splits[0]), math.sqrt(
                        float(mz_splits[1]) + 1.0)  # adding 1 to avoid sqrt of zero
                    if moz > max_moz:
                        max_moz = moz
                    if 0 < round(moz) < spec_size:  # moz*10 for 0.1 precision
                        # spec[round(moz*10)] += round(intensity)
                        if ind[-1] == round(moz):  # moz*10 for 0.1 precision
                            val[-1] = intensity if val[-1] < intensity else val[-1]
                        else:
                            ind.append(round(moz))  # moz*10 for 0.1 precision
                            val.append(intensity)
                if num_peaks < 15:
                    is_name = is_mw = is_charge = False
                    continue

                ind = np.array(ind)
                val = np.array(val)
                val = (val - np.amin(val)) / (np.amax(val) - np.amin(val))
                val[0] = mass
                for ch_val in range(l_charge):
                    val[ch_val + 1] = 1
                assert len(ind) == len(val)
                spec = np.array([ind, val])

                summ[ind] += val
                sq_sum[ind] += val**2
                N += 1

                is_name = True

            if is_name and is_mw and is_charge:
                is_name = is_mw = is_charge = False

                """output the data to """
                spec_file_name = '{}-{}-{}.npy'.format(lcount, mass, l_charge)
                np.save(join(out_dir, 'spectra', spec_file_name), spec)

                lcount += 1
                tot_count += 1

                pep = 0
                spec = []
                new = int((i / len(lines)) * 100)
                if new >= prev + 10:
                    # clear_output(wait=True)
                    print('count: ' + str(lcount))
                    print(str(new) + '%')
                    prev = new

        # print('max peaks: ' + str(max_peaks))
        print('In current file, read {} out of {}'.format(lcount, count))
        print("Ignored: large mass: {}, pep len: {}, dup: {}".format(mass_ign, pep_len_ign, dup_ign))
        print('overall running count: ' + str(tot_count))
        print('max moz: ' + str(max_moz))
#         return pep_list, dataset, label
#         tmp_pep_list, tmp_dataset, tmp_labels = read_msp(msp_file, species_id, decoy)
#         pep_list.extend(tmp_dataset)
#         dataset.extend(tmp_dataset)
#         label.extend(tmp_labels)

    # save the map. this will be used to generate masks for hard positive/negative mining during training.
    # np.save(join(out_dir, "idx_spec_map.npy"), idx_spec_map)
    # with open(join(out_dir, 'pep_spec.pkl'), 'wb') as f:
    #     pickle.dump(pep_spec, f)

    print("Statistics:")
    print("Charge distribution:")
    print(ch)
    print("Modified:\t{}".format(modified))
    print("Unmodified:\t{}".format(unmodified))
    print("Unique Peptides:\t{}".format(len(unique_pep_set)))
    print("Sum: {}".format(summ))
    print("Sum-Squared: {}".format(sq_sum))
    print("N: {}".format(N))
    means = summ / N
    print("mean: {}".format(means))
    stds = np.sqrt((sq_sum / N) - means**2)
    stds[stds < 0.0000001] = float("inf")
    print("std: {}".format(stds))
    np.save(join(out_dir, 'means.npy'), means)
    np.save(join(out_dir, 'stds.npy'), stds)

# return spectra, masses, charges

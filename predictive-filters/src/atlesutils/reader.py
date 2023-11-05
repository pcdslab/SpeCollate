import re
from os import listdir
from os.path import isfile, join

import numpy as np
from IPython.display import clear_output
from sklearn import preprocessing

from src.atlesconfig import config
from src.atlesutils import simulatespectra as sim


def read_msps(msp_folder, decoy=False):
    msp_files = [join(msp_folder, f) for f in listdir(msp_folder) if
                 isfile(join(msp_folder, f)) and f.split('.')[-1] == 'msp']
    assert len(msp_files) > 0

    print('reading {} files'.format(len(msp_files)))
    pep_list = []
    dataset = []
    label = []

    for species_id, msp_file in enumerate(msp_files):
        print('Reading: {}'.format(msp_file))
        tmp_pep_list, tmp_dataset, tmp_labels = read_msp(msp_file, species_id, decoy)
        pep_list.extend(tmp_dataset)
        dataset.extend(tmp_dataset)
        label.extend(tmp_labels)

    return pep_list, dataset, label


def read_msp(msp_file, species_id, decoy=False):
    """Read annotated spectra from msp file and return
    peptide list, dataset, and labels.
    :param decoy:
    :param species_id: id of the species
    :param msp_file: str
    :returns list
    """

    f = open(msp_file, "r")
    lines = f.readlines()
    f.close()

    pep_list = []
    dataset = []
    label = []

    # FIXME: config should use only one get_config call.
    spec_size = config.get_config(section='input', key='spec_size')
    charge = config.get_config(section='input', key='charge')
    use_mods = config.get_config(section='input', key='use_mods')
    num_species = config.get_config(section='input', key='num_species')

    print('len of file: ' + str(len(lines)))
    count = 0
    limit = 200000
    pep = []
    spec = []
    pep_set = set()
    is_name = is_mw = is_num_peaks = False
    prev = 0
    max_peaks = max_moz = 0
    i = 0
    while i < len(lines) and limit > 0:
        line = lines[i]
        i += 1
        if line.startswith('Name:'):
            name_groups = re.search(r"Name:\s(?P<pep>[a-zA-Z]+)/(?P<charge>\d+)"
                                    r"(?:_(?P<num_mods>\d+)(?P<mods>.*))?", line)
            if not name_groups:
                continue
            pep = name_groups['pep']
            l_charge = int(name_groups['charge'])
            num_mods = int(name_groups['num_mods'])

            if l_charge > charge:
                continue

            if (use_mods or not num_mods) and pep + str(l_charge) not in pep_set:
                pep_set.add(pep + str(l_charge))
                is_name = True
            else:
                continue

        if is_name and line.startswith('MW:'):
            mass = float(re.findall(r"MW:\s([-+]?[0-9]*\.?[0-9]*)", line)[0])
            if round(mass) < spec_size:
                is_mw = True
                # limit = limit - 1
            else:
                is_name = is_mw = is_num_peaks = False
                continue

        if is_name and is_mw and line.startswith('Num peaks:'):
            num_peaks = int(re.findall(r"Num peaks:\s([0-9]*\.?[0-9]*)", line)[0])
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

            # for k in range(1, charge + 1):
            #     spec[-k] = 0
            # spec[-l_charge] = 1000.0
            spec = np.clip(spec, None, 1000.0)
            # spec = preprocessing.scale(spec)

            is_num_peaks = True

        if is_name and is_mw and is_num_peaks:
            is_name = is_mw = is_num_peaks = False
            # revPep = pep[0] + pep[1:-1][::-1] + pep[-1]
            pep_list.append(pep)
            t_spec = sim.get_spectrum(pep)

            for k in range(0, charge):
                t_spec[k] = 1.0 if k <= l_charge - 1 else 0.0
            for k in range(charge, charge + num_species):
                t_spec[k] = 1.0 if k - charge == species_id else 0.0
            t_spec = preprocessing.scale(t_spec)

            if decoy:
                revPep = sim.get_rand_mod(pep)
                if pep == revPep:
                    print('decoy is the same. shuffling')
                    # revPep = ''.join(rand.sample(revPep,len(revPep)))
                    revPep = sim.get_rand_mod(pep, len(pep))
                    print(pep)
                    print(revPep)
                rt_spec = sim.get_spectrum(revPep)
                #rt_spec = preprocessing.scale(rt_spec)
                dataset.append([spec, t_spec, rt_spec])
                label.append([1, -1])
            else:
                dataset.append([spec, t_spec])
                label.append([1])

            count = count + 1
            pep = 0
            spec = []
            new = int((i / len(lines)) * 100)
            if new > prev:
                clear_output(wait=True)
                print(str(new) + '%')
                prev = new

    print('max peaks: ' + str(max_peaks))
    print('count: ' + str(count))
    print('max moz: ' + str(max_moz))
    return pep_list, dataset, label


# def read_msp_backup(msp_file, decoy=False):
#     """Read annotated spectra from msp file and return
#     peptide list, dataset, and labels.
#     :param msp_file: str
#     :returns list
#     """
#
#     f = open(msp_file, "r")
#     lines = f.readlines()
#     f.close()
#
#     pep_list = []
#     dataset = []
#     label = []
#
#     # FIXME: config should use only one get_config call.
#     spec_size = config.get_config(section='input', key='spec_size')
#     charge = config.get_config(section='input', key='charge')
#     use_mods = config.get_config(section='input', key='use_mods')
#
#     print('len of file: ' + str(len(lines)))
#     count = 0
#     limit = 200000
#     pep = []
#     spec = []
#     pep_set = set()
#     is_name = is_mw = is_num_peaks = False
#     prev = 0
#     max_peaks = max_moz = 0
#     i = 0
#     while i < len(lines) and limit > 0:
#         line = lines[i]
#         i += 1
#         splits = line.split(':')
#         if (splits[0] == 'Name') and '_' in line:
#             split1 = splits[1]
#             l_charge = int(split1[split1.find('_') - 1])
#             if l_charge != charge and charge > 0:  # l_charge == l_charge always true.
#                 continue
#
#             if use_mods or ('(' not in splits[1] and ')' not in splits[1]):
#                 pep = split1.split('/')[0].lstrip(' ')
#
#                 if pep not in pep_set:
#                     pep_set.add(pep)
#                 else:
#                     continue
#
#                 is_name = True
#
#         if is_name and splits[0] == 'MW':
#             mass = float(splits[1])
#             if round(mass) < spec_size:
#                 is_mw = True
#                 # limit = limit - 1
#             else:
#                 is_name = is_mw = is_num_peaks = False
#                 continue
#
#         if is_name and is_mw and splits[0] == 'Num peaks':
#             num_peaks = int(splits[1])
#             if num_peaks > max_peaks:
#                 max_peaks = num_peaks
#
#             spec = np.zeros(spec_size)
#             while lines[i] != '\n':
#                 mz_line = lines[i]
#                 i += 1
#                 mz_splits = mz_line.split('\t')
#                 moz, intensity = float(mz_splits[0]), float(mz_splits[1])
#                 if moz > max_moz:
#                     max_moz = moz
#                 spec[round(moz)] += round(intensity)
#
#             spec = np.clip(spec, None, 1000.0)
#             spec = preprocessing.scale(spec)
#
#             is_num_peaks = True
#
#         if is_name and is_mw and is_num_peaks:
#             is_name = is_mw = is_num_peaks = False
#             # revPep = pep[0] + pep[1:-1][::-1] + pep[-1]
#             pep_list.append(pep)
#             t_spec = preprocessing.scale(sim.get_spectrum(pep))
#             if decoy:
#                 revPep = sim.get_rand_mod(pep)
#                 if pep == revPep:
#                     print('decoy is the same. shuffling')
#                     # revPep = ''.join(rand.sample(revPep,len(revPep)))
#                     revPep = sim.get_rand_mod(pep, len(pep))
#                     print(pep)
#                     print(revPep)
#                 rt_spec = preprocessing.scale(sim.get_spectrum(revPep))
#                 dataset.append([spec, t_spec, rt_spec])
#                 label.append([1, -1])
#             else:
#                 dataset.append([spec, t_spec])
#                 label.append([1])
#
#             count = count + 1
#             pep = 0
#             spec = []
#             new = int((i / len(lines)) * 100)
#             if new > prev:
#                 clear_output(wait=True)
#                 print(str(new) + '%')
#                 prev = new
#
#     print('max peaks: ' + str(max_peaks))
#     print('count: ' + str(count))
#     print('max moz: ' + str(max_moz))
#     return pep_list, dataset, label


# def read_msp_with_decoy(msp_file):
#     """Read annotated spectra from msp file and return
#     data structure along with decoy peptides.
#     :param msp_file: str
#     :returns list
#     """
#
#     f = open(msp_file, "r")
#     lines = f.readlines()
#     f.close()
#
#     dataset = []
#     label = []
#
#     # FIXME: config should use only one get_config call.
#     spec_size = config.get_config(section='input', key='spec_size')
#     charge = config.get_config(section='input', key='charge')
#     use_mods = config.get_config(section='input', key='use_mods')
#
#     print('len of file: ' + str(len(lines)))
#     count = 0
#     limit = 200000
#     pep = 0
#     spec = []
#     is_name = is_mw = is_num_peaks = False
#     prev = 0
#     max_peaks = max_moz = 0
#     i = 0
#     while i < len(lines) and limit > 0:
#         line = lines[i]
#         i += 1
#         splits = line.split(':')
#         if (splits[0] == 'Name') and '_' in line:
#             split1 = splits[1]
#             l_charge = int(split1[split1.find('_') - 1])
#             if l_charge != charge and charge > 0:  # l_charge == l_charge always true.
#                 continue
#             if use_mods:
#                 pep = split1.split('/')[0].lstrip(' ')
#                 is_name = True
#             elif '(' not in splits[1] and ')' not in splits[1]:
#                 pep = split1.split('/')[0].lstrip(' ')
#                 is_name = True
#
#         if is_name and splits[0] == 'MW':
#             mass = float(splits[1])
#             if round(mass) < spec_size:
#                 is_mw = True
#                 # limit = limit - 1
#             else:
#                 is_name = is_mw = is_num_peaks = False
#                 continue
#
#         if is_name and is_mw and splits[0] == 'Num peaks':
#             num_peaks = int(splits[1])
#             if num_peaks > max_peaks:
#                 max_peaks = num_peaks
#
#             spec = np.zeros(spec_size)
#             while lines[i] != '\n':
#                 mz_line = lines[i]
#                 i += 1
#                 mz_splits = mz_line.split('\t')
#                 moz, intensity = float(mz_splits[0]), float(mz_splits[1])
#                 if moz > max_moz:
#                     max_moz = moz
#                 spec[round(moz)] += round(intensity)
#
#             spec = np.clip(spec, None, 1000.0)
#             spec = preprocessing.scale(spec)
#
#             is_num_peaks = True
#
#         if is_name and is_mw and is_num_peaks:
#             is_name = is_mw = is_num_peaks = False
#             # revPep = pep[0] + pep[1:-1][::-1] + pep[-1]
#             revPep = sim.get_rand_mod(pep)
#             if pep == revPep:
#                 print('decoy is the same. shuffling')
#                 # revPep = ''.join(rand.sample(revPep,len(revPep)))
#                 revPep = sim.get_rand_mod(pep, len(pep))
#                 print(pep)
#                 print(revPep)
#             t_spec = preprocessing.scale(sim.get_spectrum(pep))
#             rt_spec = preprocessing.scale(sim.get_spectrum(revPep))
#
#             # TODO: revert this back.
#             # dataset.append([spec, t_spec, rt_spec])
#             dataset.append([pep, spec, t_spec, rt_spec])
#             label.append([1, -1])
#
#             count = count + 1
#             pep = 0
#             spec = []
#             new = int((i / len(lines)) * 100)
#             if new > prev:
#                 clear_output(wait=True)
#                 print(str(new) + '%')
#                 prev = new
#
#     print('max peaks: ' + str(max_peaks))
#     print('count: ' + str(count))
#     print('max moz: ' + str(max_moz))
#     return dataset, label


def read_mgfs(folder_path):
    mgf_files = [f for f in listdir(folder_path) if isfile(join(folder_path, f)) and f.split('.')[-1] == 'mgf']
    assert len(mgf_files) > 0

    spec_size = config.get_config(section='input', key='spec_size')
    charge = config.get_config(section='input', key='charge')

    spectra = []
    masses = []
    charges = []
    for file in mgf_files:
        f = open(join(folder_path, file))
        spec_lines = f.readlines()
        f.close()

        if not spec_lines:
            continue

        spec = np.zeros(spec_size)
        isMass = False
        i = 0
        '''Read Headers'''
        while True:
            line = spec_lines[i]
            i += 1

            splits = line.split('=')
            if splits[0].upper() == 'PEPMASS':
                masses.append(float(splits[1].split(' ')[0]))
                isMass = True

            if isMass and splits[0].upper() == 'CHARGE':
                l_charge = int(splits[1][0])
                if charge and l_charge != charge:
                    del masses[-1]
                    isMass = False
                    isCharge = False
                else:
                    charges.append(l_charge)
                    isCharge = True
                break

        '''Read Spectrum'''
        while isMass and isCharge and i < len(spec_lines):
            line = spec_lines[i]
            i += 1

            if line != '\n' and 'END IONS' not in line.upper():
                splits = line.split(' ')
                moz, intensity = float(splits[0]), float(splits[1])
                spec[round(moz)] += round(intensity)
            elif 'END IONS' in line.upper():
                break

        if isMass and isCharge:
            spec = np.clip(spec, None, 1000.0)
            spec = preprocessing.scale(spec)
            spectra.append(spec)

    return spectra, masses, charges


def read_ms2(file):
    f = open(file)
    lines = f.readlines()
    f.close()

    spec_size = config.get_config(section='input', key='spec_size')
    charge = config.get_config(section='input', key='charge')

    spectra = []
    masses = []
    charges = []
    i = 0
    while i < len(lines):
        line = lines[i][:-1]
        i += 1

        splits = line.split('\t')
        if splits[0] == 'Z' and (charge <= 0 or float(splits[1]) == charge):
            charges.append(float(splits[1]))
            masses.append(float(splits[2]))
            spec = np.zeros(spec_size)
            while i < len(lines):
                line = lines[i][:-1]  # remove the \n character
                i += 1
                splits = line.split(' ')
                if 'S' in splits[0]:
                    break
                if 'Z' in splits[0]:
                    continue
                moz, intensity = float(splits[0]), float(splits[1])
                spec[round(moz)] += round(intensity)

            spec = np.clip(spec, None, 1000.0)
            spec = preprocessing.scale(spec)
            spectra.append(spec)

    return spectra, masses, charges

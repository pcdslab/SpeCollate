from os import listdir
from os.path import isfile, join
from pathlib import Path
import shutil
import re

import numpy as np
import torch

from src.snapconfig import config

def preprocess_msps(msp_dir, out_dir):
    in_path = Path(msp_dir)
    assert in_path.exists() and in_path.is_dir()
    
    msp_files = [join(msp_dir, f) for f in listdir(msp_dir) if
                 isfile(join(msp_dir, f)) and f.split('.')[-1] == 'msp']
    assert len(msp_files) > 0
    
    out_path = Path(out_dir)
    if out_path.exists() and out_path.is_dir():
        shutil.rmtree(out_path)
    out_path.mkdir()
    Path(join(out_path, 'spectra')).mkdir()
    Path(join(out_path, 'peptides')).mkdir()
        
    print('reading {} files'.format(len(msp_files)))
    
    count = 0
    max_peaks = max_moz = 0
    for species_id, msp_file in enumerate(msp_files):
        print('Reading: {}'.format(msp_file))
        
        f = open(msp_file, "r")
        lines = f.readlines()
        f.close()

        # FIXME: config should use only one get_config call.
        spec_size = config.get_config(section='input', key='spec_size')
        seq_len = config.get_config(section='ml', key='pep_seq_len')

        print('len of file: ' + str(len(lines)))
        limit = 200000
        pep = []
        spec = []
        is_name = is_mw = is_num_peaks = False
        prev = 0
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
                if len(pep) + 1 > seq_len:
                    continue
                    
                l_charge = int(name_groups['charge'])
                num_mods = int(name_groups['num_mods'])

                is_name = True

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
                
                #pep = '{}{}{}'.format(charge, species_id, pep)

                """output the data to """
                spec_tensor = torch.tensor((np.asarray(spec) - 3.725) / 51.479, dtype=torch.float)
                
                torch.save(spec_tensor, 
                           join(out_dir, 'spectra', '{}-{}-{}-{}-{}.pt'
                                .format(count, species_id, mass, l_charge, int(num_mods > 0))))
                
                pep_file_name = '{}-{}-{}-{}-{}.pep'.format(count, species_id, mass, l_charge, int(num_mods > 0))
                    
                with open(join(out_path, 'peptides', pep_file_name), 'w+') as f:
                    f.write(pep)

                count = count + 1
                pep = 0
                spec = []
                new = int((i / len(lines)) * 100)
                if new > prev + 10:
                    # clear_output(wait=True)
                    print(str(new) + '%')
                    prev = new

        print('max peaks: ' + str(max_peaks))
        print('count: ' + str(count))
        print('max moz: ' + str(max_moz))
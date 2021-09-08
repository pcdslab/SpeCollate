from os import listdir
from os.path import isfile, join
from pathlib import Path
import shutil
import re
import math

import numpy as np

from src.snapconfig import config


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
    
    files = [join(dir_path, f) for f in listdir(dir_path) if
                 isfile(join(dir_path, f)) and not f.startswith('.') 
                 and f.split('.')[-1] == ext and f not in ignore_list]
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
    
    ch = np.zeros(20)
    modified = 0
    unmodified = 0
    unique_pep_set = set()
    
    summ = np.zeros(spec_size)
    sq_sum = np.zeros(spec_size)
    N = 0
    
    tot_count = 0
    max_moz = 0
    for mgf_file in mgf_files:
        print('Reading: {}'.format(mgf_file))
        
        f = open(mgf_file, "r")
        lines = f.readlines()
        f.close()
        
        count = lcount = 0
        
        mass_ign = 0
        pep_len_ign = 0
        dup_ign = 0

        print('len of file: ' + str(len(lines)))
        limit = 200000
        spec = []
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

                while not isfloat(re.split(' |\t|=', lines[i])[0]):
                    i += 1
                    
                spec_ind = []
                spec_val = []
                num_peaks = 0
                while 'END IONS' not in lines[i].upper():
                    if lines[i] == '\n':
                        i += 1
                        continue
                    mz_line = lines[i]
                    i += 1
                    num_peaks += 1
                    mz_splits = re.split(' |\t', mz_line)
                    moz = round(float(mz_splits[0]) * 10) # + 32 # 32 because charge is len 8 and mass is len 24
                    intensity = math.sqrt(float(mz_splits[1]) + 1.0) # adding 1 to avoid sqrt of zero
#                     intensity = float(mz_splits[1])
                    if moz > max_moz:
                        max_moz = moz
                    if 0 < moz < spec_size:
                        # spec[round(moz*10)] += round(intensity)
                        if spec_ind and spec_ind[-1] == moz:
                            spec_val[-1] = max(intensity, spec_val[-1])
                        else:
                            spec_ind.append(moz)
                            spec_val.append(intensity) # adding one to avoid sqrt of zero
                if num_peaks < 10:
                    is_name = is_mw = is_charge = False
                    continue
                    
                spec_ind = np.array(spec_ind)
                spec_val = np.array(spec_val)
                spec_val = (spec_val - np.amin(spec_val)) / (np.amax(spec_val) - np.amin(spec_val))

                ind = spec_ind
                val = spec_val
            
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
                    #clear_output(wait=True)
                    print('count: ' + str(lcount))
                    print(str(new) + '%')
                    prev = new

        #print('max peaks: ' + str(max_peaks))
        print('In current file, read {} out of {}'.format(lcount, count))
        print("Ignored: large mass: {}, pep len: {}, dup: {}".format(mass_ign, pep_len_ign, dup_ign))
        print('overall running count: ' + str(tot_count))
        print('max moz: ' + str(max_moz))
    
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
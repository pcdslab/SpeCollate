import pickle
import re
import time
import random as rand
from os.path import join

from sklearn.model_selection import train_test_split

from src.snapconfig import config

def apply_filter(l_filt, file_name):
    try:
        file_parts = re.search(r"(\d+)-(\d+)-(\d+.\d+)-(\d+)-(\d+).[pt|npy]", file_name)
        l_charge = int(file_parts[4])
        mods = int(file_parts[5])
    except:
        print(file_name)
        print(file_parts)
    
    if ((l_filt["charge"] == 0 or l_charge <= l_filt["charge"]) # change this back to <=
        and (mods <= l_filt["mods"])):
        return True
    
    return True#False

def load_file_names(l_filt, l_listing_path, count=None):
    'Load the peptide and corresponding spectra file names that satisfy the filter'
    with open(l_listing_path, 'rb') as f:
        dir_listing = pickle.load(f)

    rand.shuffle(dir_listing)
    l_pep_file_names = []
    l_spec_file_names_lists = []
    for pep, spec_list in dir_listing[:count]:
        spec_file_list = []
        for spec in spec_list:
            if apply_filter(l_filt, spec):
                spec_file_list.append(spec)
        if spec_file_list:
            l_pep_file_names.append(pep)
            l_spec_file_names_lists.append(spec_file_list)

    assert len(l_pep_file_names) == len(l_spec_file_names_lists)
    return l_pep_file_names, l_spec_file_names_lists

if __name__ == '__main__':
    charge      = config.get_config(section='input', key='charge')
    use_mods    = config.get_config(section='input', key='use_mods')
    num_mods    = config.get_config(section='input', key='num_mods')
    filt        = {'charge': charge, 'mods': num_mods if use_mods else 0}
    test_size   = config.get_config(section='ml', key='test_size')
    train_count = config.get_config(section="ml", key="train_count")
    batch_size  = config.get_config(section="ml", key="batch_size")
    train_count = None if train_count == 0 else train_count

    in_tensor_dir = config.get_config(section='preprocess', key='in_tensor_dir')
    print(in_tensor_dir)
    listing_path = join(in_tensor_dir, 'pep_spec.pkl')
    pep_file_names, spec_file_names_lists = load_file_names(filt, listing_path, train_count)
    
    split_rand_state = int(time.time())
    train_peps, test_peps, train_specs, test_specs = train_test_split(
        pep_file_names, spec_file_names_lists, test_size=test_size,
        random_state=split_rand_state, shuffle=True)
    
    # test_peps, val_peps, test_specs, val_specs = train_test_split(
    #     test_peps, test_specs, test_size=.1,
    #     random_state=split_rand_state, shuffle=True)
    # get the 100k version
    # train_peps  = train_peps[:80000]
    # train_specs = train_specs[:80000]
    # test_peps   = test_peps[:20000]
    # test_specs  = test_specs[:20000]

    print("Writing train test split listings as pickles.")
    with open(join(in_tensor_dir, "train_peps.pkl"), "wb") as trp:
        pickle.dump(train_peps, trp)
    with open(join(in_tensor_dir, "train_specs.pkl"), "wb") as trs:
        pickle.dump(train_specs, trs)
    with open(join(in_tensor_dir, "test_peps.pkl"), "wb") as tep:
        pickle.dump(test_peps, tep)
    with open(join(in_tensor_dir, "test_specs.pkl"), "wb") as tes:
        pickle.dump(test_specs, tes)
    # with open(join(in_tensor_dir, "val_peps.pkl"), "wb") as vap:
    #     pickle.dump(test_peps, vap)
    # with open(join(in_tensor_dir, "val_specs.pkl"), "wb") as vas:
    #     pickle.dump(test_specs, vas)

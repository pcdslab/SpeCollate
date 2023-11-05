import sys
import gc
import numpy as np
import torch
import progressbar
from tqdm import tqdm

from src.atlesconfig import config
from src.atlestrain import process


def ppm(val, ppm_val):
    return (ppm_val / 1000000.0) * val


def spec_collate(batch):
    specs = torch.stack([item for item in batch], 0)
    dummy_pep = np.zeros(config.get_config(section="ml", key="pep_seq_len"))
    dummy_pep = torch.from_numpy(dummy_pep).long().unsqueeze(0)
    return [specs, dummy_pep]


def pep_collate(batch):
    peps = torch.stack([item for item in batch], 0)
    dummy_spec = np.zeros(config.get_config(section="input", key="spec_size"))
    dummy_spec = torch.from_numpy(dummy_spec).float().unsqueeze(0)
    dummy_pep = np.zeros((2, config.get_config(section="ml", key="pep_seq_len") + 24))
    dummy_pep = torch.from_numpy(dummy_pep).long()#.unsqueeze(0)
    # tqdm.write("{}".format(peps.shape))
    # tqdm.write("{}".format(dummy_pep.shape))
    return [dummy_spec, peps, dummy_pep]


def get_search_mask(spec_masses, pep_masses, tol):
    l_tol = tol
    rows = []
    cols = []
    pep_min = pep_max = 0
    for row_id, spec_mass in enumerate(spec_masses):
        # min_mass = max(spec_mass - l_tol, 0.0)
        # max_mass = spec_mass + l_tol
        min_mass = max(spec_mass - ppm(spec_mass, l_tol), 0.0)
        max_mass = spec_mass + ppm(spec_mass, l_tol)
        while (pep_min < len(pep_masses) and 
               min_mass > pep_masses[pep_min]):
            pep_min += 1
        while (pep_max < len(pep_masses) and 
               max_mass > pep_masses[pep_max]):
            pep_max += 1
        # pep_min = max(pep_min - 1, 0)
        # pep_max = min(pep_max + 1, len(pep_masses) - 1)
            
        # if pep_max == pep_min:
        #     print(row_id, pep_max, pep_min)
        rows.extend([row_id] * (pep_max - pep_min))
        cols.extend(range(pep_min, pep_max))
    
    assert len(rows) == len(cols)
    mask = torch.zeros(len(spec_masses), len(pep_masses))
    mask[rows, cols] = 1
    return mask


def runModel(loader, s_model, device):
    with torch.no_grad():
        lens_out = torch.Tensor().cpu()
        cleavs_out = torch.Tensor().cpu()
        mods_out = torch.Tensor().cpu()
        pbar = tqdm(loader, file=sys.stdout)
        pbar.set_description('Running Model...')
        # with progressbar.ProgressBar(max_value=len(loader)) as bar:
        for batch in pbar:
            batch[0], batch[1] = batch[0].to(device), batch[1].to(device)
            input_mask = batch[0] == 0
            lens, cleavs, mods = s_model(batch[0], batch[1], input_mask)
            lens_out = torch.cat((lens_out, lens.to("cpu")), dim=0)
            cleavs_out = torch.cat((cleavs_out, cleavs.to("cpu")), dim=0)
            mods_out = torch.cat((mods_out, mods.to("cpu")), dim=0)
            # bar.update(batch_idx)
        return lens_out, cleavs_out, mods_out


def search(search_loader, datasets, embeddings, device):
    pep_sort_inds = []
    pep_sort_vals = []
    dec_sort_inds = []
    dec_sort_vals = []

    spec_dataset = datasets["spec_dataset"]
    pep_dataset  = datasets["pep_dataset"]
    dec_dataset  = datasets["dec_dataset"]

    e_peps = embeddings["e_peps"]
    e_decs = embeddings["e_decs"]

    keep_psms = config.get_config(key="keep_psms", section="search")
    precursor_tolerance = config.get_config(key="precursor_tolerance", section="search")

    pbar = tqdm(search_loader, file=sys.stdout)
    pbar.set_description('Running Database Search...')
    # with progressbar.ProgressBar(max_value=len(search_loader)) as bar:
    for idx, spec_batch in enumerate(pbar):
        l_tol = precursor_tolerance
        batch_size = search_loader.batch_size
        st = idx * batch_size
        en = st + batch_size
        spec_masses = spec_dataset.spec_mass_list[st:en]
        min_mass = max(spec_masses[0] - l_tol, 0)
        max_mass = spec_masses[-1] + l_tol
        pep_min = pep_max = 0

        while (pep_min < len(pep_dataset.pep_mass_list) and 
                min_mass - pep_dataset.pep_mass_list[pep_min] > 0.01):
            pep_min += 1
        while (pep_max < len(pep_dataset.pep_mass_list) and
                max_mass - pep_dataset.pep_mass_list[pep_max] >= 0.01):
            pep_max += 1

        dec_min = dec_max = 0
        while (dec_min < len(dec_dataset.pep_mass_list) and
                min_mass - dec_dataset.pep_mass_list[dec_min] > 0.01):
            dec_min += 1
        while (dec_max < len(dec_dataset.pep_mass_list) and 
                max_mass - dec_dataset.pep_mass_list[dec_max] >= 0.01):
            dec_max += 1

        pep_batch = e_peps[pep_min:pep_max]
        pep_masses = pep_dataset.pep_mass_list[pep_min:pep_max]
        dec_batch = e_decs[dec_min:dec_max]
        dec_masses = dec_dataset.pep_mass_list[dec_min:dec_max]

        spec_batch = spec_batch.to(device)
        #print("pep batch len: {}".format(len(pep_batch)))
        l_pep_batch_size = 16384
        # l_pep_batch_size = 32768
        pep_loader = torch.utils.data.DataLoader(
            dataset=pep_batch, batch_size=l_pep_batch_size)
        l_pep_dist = []
        for pep_idx, l_pep_batch in enumerate(pep_loader):
            l_pep_batch = l_pep_batch.to(device)
            l_st = pep_idx * l_pep_batch_size
            l_en = l_st + l_pep_batch_size
            l_pep_masses = pep_masses[l_st:l_en]
            spec_pep_mask = get_search_mask(spec_masses, l_pep_masses, precursor_tolerance).to(device)
            # spec_pep_mask[spec_pep_mask == 0] = float("inf")
            spec_pep_dist = 1.0 / process.pairwise_distances(spec_batch, l_pep_batch)
            l_pep_dist.append((spec_pep_dist * spec_pep_mask).to("cpu"))
        
        l_pep_dist.append(torch.zeros(len(spec_batch), keep_psms + 1))
        pep_sort = torch.cat(l_pep_dist, 1)
        pep_lcn = np.ma.masked_array(pep_sort, mask=pep_sort==0).min(1).data
        pep_sort = pep_sort.sort(descending=True)
        pep_sort_inds.append(pep_sort.indices[:, :keep_psms + 1] + pep_min) # offset for the global array
        pep_sort_vals.append(torch.cat((pep_sort.values[:, :keep_psms + 1], 
                                        torch.from_numpy(pep_lcn).unsqueeze(1)), 1))
        
        dec_loader = torch.utils.data.DataLoader(
            dataset=dec_batch, batch_size=l_pep_batch_size)
        l_dec_dist = []
        for dec_idx, l_dec_batch in enumerate(dec_loader):
            l_dec_batch = l_dec_batch.to(device)
            l_st = dec_idx * l_pep_batch_size
            l_en = l_st + l_pep_batch_size
            l_dec_masses = dec_masses[l_st:l_en]
            spec_dec_mask = get_search_mask(spec_masses, l_dec_masses, precursor_tolerance).to(device)
            # spec_dec_mask[spec_dec_mask == 0] = float("inf")
            spec_dec_dist = 1.0 / process.pairwise_distances(spec_batch, l_dec_batch)
            l_dec_dist.append((spec_dec_dist * spec_dec_mask).to("cpu"))
        
        l_dec_dist.append(torch.zeros(len(spec_batch), keep_psms + 1))
        dec_sort = torch.cat(l_dec_dist, 1)
        dec_lcn = np.ma.masked_array(dec_sort, mask=dec_sort==0).min(1).data
        dec_sort = dec_sort.sort(descending=True)
        dec_sort_inds.append(dec_sort.indices[:, :keep_psms + 1] + dec_min) # offset for the global array
        dec_sort_vals.append(torch.cat((dec_sort.values[:, :keep_psms + 1], 
                                        torch.from_numpy(dec_lcn).unsqueeze(1)), 1))
        # bar.update(idx)
    
    pep_inds = torch.cat(pep_sort_inds, 0)
    pep_vals = torch.cat(pep_sort_vals, 0)
    dec_inds = torch.cat(dec_sort_inds, 0)
    dec_vals = torch.cat(dec_sort_vals, 0)
    return pep_inds, pep_vals, dec_inds, dec_vals


def parallel_search(search_loader, datasets, embeddings, rank):
    sort_inds = []
    sort_vals = []

    spec_dataset = datasets["spec_dataset"]
    
    if rank == 0:
        pep_dataset  = datasets["pep_dataset"]
        e_peps = embeddings["e_peps"]
    if rank == 1:
        dec_dataset  = datasets["dec_dataset"]
        e_decs = embeddings["e_decs"]
    

    keep_psms = config.get_config(key="keep_psms", section="search")
    precursor_tolerance = config.get_config(key="precursor_tolerance", section="search")

    pbar = tqdm(search_loader, file=sys.stdout)
    pbar.set_description('Running Database Search...')
    # with progressbar.ProgressBar(max_value=len(search_loader)) as bar:
    for idx, spec_batch in enumerate(pbar):
        l_tol = precursor_tolerance
        batch_size = search_loader.batch_size
        st = idx * batch_size
        en = st + batch_size
        spec_masses = spec_dataset.spec_mass_list[st:en]
        # min_mass = max(spec_masses[0] - l_tol, 0)
        # max_mass = spec_masses[-1] + l_tol
        min_mass = max(spec_masses[0] - ppm(spec_masses[0], l_tol), 0)
        max_mass = spec_masses[-1] + ppm(spec_masses[-1], l_tol)
        
        if rank == 0:
            pep_min = pep_max = 0
            while (pep_min < len(pep_dataset.pep_mass_list) and 
                    min_mass - pep_dataset.pep_mass_list[pep_min] > 0.001):
                pep_min += 1
            while (pep_max < len(pep_dataset.pep_mass_list) and
                    max_mass - pep_dataset.pep_mass_list[pep_max] >= 0.001):
                pep_max += 1

            pep_batch = e_peps[pep_min:pep_max]
            pep_masses = pep_dataset.pep_mass_list[pep_min:pep_max]

        if rank == 1:
            dec_min = dec_max = 0
            while (dec_min < len(dec_dataset.pep_mass_list) and
                    min_mass - dec_dataset.pep_mass_list[dec_min] > 0.001):
                dec_min += 1
            while (dec_max < len(dec_dataset.pep_mass_list) and 
                    max_mass - dec_dataset.pep_mass_list[dec_max] >= 0.001):
                dec_max += 1

            dec_batch = e_decs[dec_min:dec_max]
            dec_masses = dec_dataset.pep_mass_list[dec_min:dec_max]

        spec_batch = spec_batch.to(rank)
        #print("pep batch len: {}".format(len(pep_batch)))
        l_pep_batch_size = 16384
        # l_pep_batch_size = 32768
        if rank == 0:
            pep_loader = torch.utils.data.DataLoader(
                dataset=pep_batch, batch_size=l_pep_batch_size)
            l_pep_dist = []
            for pep_idx, l_pep_batch in enumerate(pep_loader):
                l_pep_batch = l_pep_batch.to(rank)
                l_st = pep_idx * l_pep_batch_size
                l_en = l_st + l_pep_batch_size
                l_pep_masses = pep_masses[l_st:l_en]
                # spec_pep_mask = get_search_mask(spec_masses, l_pep_masses, precursor_tolerance).to(rank)
                # spec_pep_mask[spec_pep_mask == 0] = float("inf")
                spec_pep_dist = 1.0 / process.pairwise_distances(spec_batch, l_pep_batch).to("cpu")
                l_pep_dist.append(spec_pep_dist)
        
            pep_sort = torch.cat(l_pep_dist, 1)
            spec_pep_mask = get_search_mask(spec_masses, pep_masses, precursor_tolerance)
            pep_sort = (pep_sort * spec_pep_mask)
            pep_sort = torch.cat((pep_sort, torch.zeros(len(spec_batch), keep_psms + 1)), axis=1)
            pep_lcn = np.ma.masked_array(pep_sort, mask=pep_sort==0).min(1).data
            pep_sort = pep_sort.sort(descending=True)
            sort_inds.append(pep_sort.indices[:, :keep_psms + 1] + pep_min) # offset for the global array
            sort_vals.append(torch.cat((pep_sort.values[:, :keep_psms + 1], 
                                            torch.from_numpy(pep_lcn).unsqueeze(1)), 1))
        
        if rank == 1:
            dec_loader = torch.utils.data.DataLoader(
                dataset=dec_batch, batch_size=l_pep_batch_size)
            l_dec_dist = []
            for dec_idx, l_dec_batch in enumerate(dec_loader):
                l_dec_batch = l_dec_batch.to(rank)
                l_st = dec_idx * l_pep_batch_size
                l_en = l_st + l_pep_batch_size
                l_dec_masses = dec_masses[l_st:l_en]
                # spec_dec_mask = get_search_mask(spec_masses, l_dec_masses, precursor_tolerance).to(rank)
                # spec_dec_mask[spec_dec_mask == 0] = float("inf")
                spec_dec_dist = 1.0 / process.pairwise_distances(spec_batch, l_dec_batch).to("cpu")
                l_dec_dist.append(spec_dec_dist)
            
            dec_sort = torch.cat(l_dec_dist, 1)
            spec_dec_mask = get_search_mask(spec_masses, dec_masses, precursor_tolerance)
            dec_sort = (dec_sort * spec_dec_mask)
            dec_sort = torch.cat((dec_sort, torch.zeros(len(spec_batch), keep_psms + 1)), axis=1)
            dec_lcn = np.ma.masked_array(dec_sort, mask=dec_sort==0).min(1).data
            dec_sort = dec_sort.sort(descending=True)
            sort_inds.append(dec_sort.indices[:, :keep_psms + 1] + dec_min) # offset for the global array
            sort_vals.append(torch.cat((dec_sort.values[:, :keep_psms + 1], 
                                            torch.from_numpy(dec_lcn).unsqueeze(1)), 1))
        # bar.update(idx)

    if rank == 0:
        pep_inds = torch.cat(sort_inds, 0)
        pep_vals = torch.cat(sort_vals, 0)
        return pep_inds, pep_vals
    if rank == 1:
        dec_inds = torch.cat(sort_inds, 0)
        dec_vals = torch.cat(sort_vals, 0)
        return dec_inds, dec_vals

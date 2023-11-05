import os
from os.path import join
import pandas as pd

from src.atlesconfig import config


def post_process_pin_files(rank, pin_dir_path):
    print("Post-processing pin files at {}".format(pin_dir_path))
    file_path = join(pin_dir_path, "target.pin" if rank == 0 else "decoy.pin")
    df = pd.read_csv(file_path, sep='\t')

    # Do anything to pin files here.

    # Keep only the top PSMs
    top_psms_df = keep_top_psms(df)

    # Write back the pin files to disk
    print("Writing pin files to disk")
    # TODO: uncomment two lines below
    os.remove(file_path)
    top_psms_df.to_csv(file_path, index=False, sep='\t')

    # TODO: remove the lines below
    # out_path = join(pin_dir_path, "target1.pin" if rank == 0 else "decoy1.pin")
    # top_psms_df.to_csv(out_path, index=False, sep='\t')


def keep_top_psms(df):
    keep_psms = config.get_config(section="search", key="keep_psms")
    # Sort the dataframe by 'SpecId' and 'SNAP' in descending order
    print('Keeping only the top {} PSMs'.format(keep_psms))

    print("Size before removal: {}".format(len(df)))
    df = df.sort_values(['SpecId', 'SNAP'], ascending=[True, False])

    # Group the dataframe by 'SpecId' and keep the top 'keep_psms' rows
    top_psms_df = df.groupby('SpecId').head(keep_psms)

    # Reset the index of the filtered dataframe
    top_psms_df.reset_index(drop=True, inplace=True)
    print("Size after removal: {}".format(len(top_psms_df)))
    return top_psms_df


def generate_percolator_input(l_pep_inds, l_pep_vals, l_spec_inds, pd_dataset, spec_charges, res_type):
    assert res_type == "target" or res_type == "decoy"
    assert len(l_pep_inds) == len(l_pep_vals) == len(l_spec_inds)
    pin_charge = config.get_config(section="search", key="charge")
    l_global_out = []
    for l_spec_idx, pep_inds_row, pep_vals_row in zip(l_spec_inds, l_pep_inds, l_pep_vals):
        l_spec_idx = l_spec_idx.item()
        # Reminder: pep_inds_row length is one less than pep_vals_row
        for iidx in range(len(pep_inds_row) - 1):
            pep_ind = pep_inds_row[iidx]
            pep_val = pep_vals_row[iidx]
            if pep_val.item() > 0:
                charge = [0] * pin_charge
                ch_idx = min(spec_charges[l_spec_idx], pin_charge)
                charge[ch_idx - 1] = 1
                label = 1 if res_type == "target" else -1
                out_row = [f"{res_type}-{l_spec_idx}", label, l_spec_idx, pep_val.item()]
                spec_mass = spec_charges[l_spec_idx]
                pep_mass = pd_dataset.pep_mass_list[pep_ind.item()]
                out_row.append(spec_mass)
                out_row.append(pep_mass)
#                 out_row.append(((pep_val - pep_vals_row[iidx + 1]).item()) / max_snap)
#                 out_row.append(((pep_val - pep_vals_row[-1]).item()) / max_snap)
                out_row.append(((pep_val - pep_vals_row[-1]).item()) / max(pep_val.item(), 1.0))
                out_row.append(((pep_val - pep_vals_row[iidx + 1]).item()) / max(pep_val.item(), 1.0))
                out_row.extend(charge)
                out_row.append(spec_mass - pep_mass)
                out_row.append(abs(spec_mass - pep_mass))

                out_pep = pd_dataset.pep_list[pep_ind.item()]
                out_pep_array = []
                for aa in out_pep:
                    if aa.islower():
                        out_pep_array.append("[" + str(config.AAMass[aa]) + "]")
                    else:
                        out_pep_array.append(aa)
                out_pep = "".join(out_pep_array)
                out_row.append((out_pep.count("K") + out_pep.count("R")
                                ) - (out_pep.count("KP") + out_pep.count("RP")) - 1)
                out_prot = pd_dataset.prot_list[pep_ind.item()]
                pep_len = sum([a.isupper() for a in out_pep])
                out_row.append(pep_len)
                out_row.append(out_pep)
                out_row.append(out_prot)
                l_global_out.append(out_row)
    return l_global_out


def write_to_pin(rank, pep_inds, psm_vals, spec_inds, l_pep_dataset, spec_charges, out_pin_dir):
    os.makedirs(out_pin_dir, exist_ok=True)
    if rank == 0:
        print("Generating percolator pin files...")
    pin_charge = config.get_config(section="search", key="charge")
    charge_cols = [f"charge-{ch+1}" for ch in range(pin_charge)]
    cols = (
        [
            "SpecId",
            "Label",
            "ScanNr",
            "SNAP",
            "ExpMass",
            "CalcMass",
            "deltCn",
            "deltLCn",
        ]
        + charge_cols
        + ["dM", "absdM", "enzInt", "PepLen", "Peptide", "Proteins"]
    )
    global_out = generate_percolator_input(
        pep_inds,
        psm_vals,
        spec_inds,
        l_pep_dataset,
        spec_charges,
        "target" if rank == 0 else "decoy",
    )
    df = pd.DataFrame(global_out, columns=cols)
    df.sort_values(by="SNAP", inplace=True, ascending=False)
    with open(join(out_pin_dir, "target.pin" if rank == 0 else "decoy.pin"), "a") as f:
        df.to_csv(f, sep="\t", index=False, header=not f.tell())

    if rank == 0:
        print("Wrote percolator files: ")
    # dist.barrier()
    print("{}".format(join(out_pin_dir, "target.pin") if rank == 0 else join(out_pin_dir, "decoy.pin")))
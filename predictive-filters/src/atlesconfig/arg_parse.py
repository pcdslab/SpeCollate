from src.atlesconfig import config


def process_args_dict(args_dict):
    if args_dict["filter"]:
        filter_value = args_dict["filter"]
        print(f"Filter value to use: {filter_value}")
        if filter_value == "all":
            config.set_config(section="filter", key="length_filter", value=True)
            config.set_config(section="filter", key="missed_cleavages_filter", value=True)
            config.set_config(section="filter", key="modification_filter", value=True)
            config.set_config(section="filter", key="len_tol_neg", value=0)
            config.set_config(section="filter", key="len_tol_pos", value=0)
        elif filter_value == "none":
            config.set_config(section="filter", key="length_filter", value=False)
            config.set_config(section="filter", key="missed_cleavages_filter", value=False)
            config.set_config(section="filter", key="modification_filter", value=False)
        elif filter_value == "len-1":
            config.set_config(section="filter", key="length_filter", value=True)
            config.set_config(section="filter", key="missed_cleavages_filter", value=True)
            config.set_config(section="filter", key="modification_filter", value=True)
            config.set_config(section="filter", key="len_tol_neg", value=-1)
            config.set_config(section="filter", key="len_tol_pos", value=1)
        elif filter_value == "no-len":
            config.set_config(section="filter", key="length_filter", value=True)
            config.set_config(section="filter", key="missed_cleavages_filter", value=True)
            config.set_config(section="filter", key="modification_filter", value=True)
            config.set_config(section="filter", key="len_tol_neg", value=-30)
            config.set_config(section="filter", key="len_tol_pos", value=30)
    
    if args_dict["pep_dir"]:
        peptide_dir = args_dict["pep_dir"]
        print(f"Path to the peptide directory: {peptide_dir}")
        config.set_config(section="search", key="pep_dir", value=peptide_dir)
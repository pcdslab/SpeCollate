from configparser import ConfigParser
import os
import ast

"""Define constants"""
AAMass = {'A': 71.037114, 'C': 103.009185, 'D': 115.026943, 'E': 129.042593, 'F': 147.068414, 'G': 57.021464,
          'H': 137.058912, 'I': 113.084064, 'K': 128.094963, 'L': 113.084064, 'M': 131.040485, 'N': 114.042927,
          'P': 97.052764, 'Q': 128.058578, 'R': 156.101111, 'S': 87.032028, 'T': 101.047679, 'V': 99.068414,
          'W': 186.079313, 'Y': 163.0633}

H2O = 18.015
NH3 = 17.031
PROTON = 1.00727647
DEFAULT_PARAM_PATH = os.path.join(os.getcwd(), 'config.ini')
PARAM_PATH = None

config = None


def get_config(section='input', key=None):
    """Read the configuration parameters and return a dictionary."""

    global config

    # If file path is given use it otherwise use default.
    file_path = PARAM_PATH if PARAM_PATH else DEFAULT_PARAM_PATH

    # Read config and convert each value to appropriate type.
    # Only for the first time.
    if not config:
        config = dict()
        config_ = ConfigParser()
        assert isinstance(file_path, str)
        config_.read(file_path)
        for section_ in config_.sections():
            config[section_] = dict()
            for key_ in config_[section_]:
                try:
                    config[section_][key_] = ast.literal_eval(config_[section_][key_])
                except (ValueError, SyntaxError):
                    config[section_][key_] = config_[section_][key_]

    if section and section in config:
        if key and key in config[section]:
            return config[section][key]
        return config[section]
    return config

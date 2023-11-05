import ast
import os
from collections import OrderedDict
from configparser import ConfigParser

# Define constants
# AAMass = OrderedDict([('A', 71.037114), ('C', 103.009185), ('D', 115.026943), ('E', 129.042593),
#                       ('F', 147.068414), ('G', 57.021464), ('H', 137.058912), ('I', 113.084064),
#                       ('K', 128.094963), ('L', 113.084064), ('M', 131.040485), ('N', 114.042927),
#                       ('P', 97.052764), ('Q', 128.058578), ('R', 156.101111), ('S', 87.032028),
#                       ('T', 101.047679), ('V', 99.068414), ('W', 186.079313), ('Y', 163.0633),
#                       ('p', 79.97), ('c', 57.02), ('o', 15.99)
#                       #('r', -17.03), ('y', 43.01), ('d', -18.01), ('t', 26.02),
#                       #('h', 0.98), ('a', 42.01)
#                       ])

AAMass = OrderedDict(
    [
        ("A", 71.037114),
        ("C", 103.009185),
        ("D", 115.026943),
        ("E", 129.042593),
        ("F", 147.068414),
        ("G", 57.021464),
        ("H", 137.058912),
        ("I", 113.084064),
        ("K", 128.094963),
        ("L", 113.084064),
        ("M", 131.040485),
        ("N", 114.042927),
        ("P", 97.052764),
        ("Q", 128.058578),
        ("R", 156.101111),
        ("S", 87.032028),
        ("T", 101.047679),
        ("V", 99.068414),
        ("W", 186.079313),
        ("Y", 163.0633),
        #   ('p', 79.9663),
        ("o", 15.994915),
        #   ('h', 0.9840),
        ("c", 57.02146)
        #   , ('a', 42.0106),
        #   ('r', -17.026549), ('y', 43.00581), ('d', -18.010565), ('t', 26.02)
    ]
)

ModMass = {
    "Oxidation": 15.994915,
    "CAM": 57.02146,
    "Carbamidomethyl": 57.02146,
    "ICAT_light": 227.12,
    "ICAT_heavy": 236.12,
    "AB_old_ICATd0": 442.20,
    "AB_old_ICATd8": 450.20,
    "Acetyl": 42.0106,
    "Deamidation": 0.9840,
    "Pyro-cmC": -17.026549,
    "Pyro-glu": -17.026549,
    "Pyro_glu": -18.010565,
    "Amide": -0.984016,
    "Phospho": 79.9663,
    "Methyl": 14.0157,
    "Carbamyl": 43.00581,
}

ModCHAR = OrderedDict(
    [
        ("15.99", "o"),
        ("57.02", "c")
        #    ,("0.98", "h"), ("42.01", "a"), ("-17.03", "r"),
        #    ("79.97", "p"), ("43.01", "y"), ("-18.01", "d"), ("26.02", "t")
    ]
)
# ModCHAR = {"15.99": "o", "0.98": "h", "57.02": "c", "42.01": "a", "-17.03": "r", "79.97": "p"}
Ignore = ["Z", "J", "O", "U", "X", "B"]
Mods = [  # {"mod_char": "p", "aas": ["S", "T", "Y"]}
    {"mod_char": "o", "aas": ["M"]}
    # {"mod_char": "a", "aas": ["nt"]}
]
H2O = 18.010564683
NH3 = 17.031
PROTON = 1.00727647

DEFAULT_PARAM_PATH = "./config.ini"
param_path = None
config_dict = None


def init_config(l_param_path):
    """Initialize the configuration parameters."""
    global param_path
    param_path = l_param_path


def get_config(section=None, key=None):
    """Return the configuration parameters."""
    global config_dict

    if section is None or key is None:
        raise ValueError("Section and key must be provided.")

    if not config_dict:
        config_dict = get_all_config()

    return config_dict[section][key]


def get_all_config():
    """Read the configuration parameters and return a dictionary."""
    global config_dict

    # If file path is given use it otherwise use default.
    print("param_path: ", param_path)
    file_path = param_path if param_path else DEFAULT_PARAM_PATH

    # Read config and convert each value to appropriate type.
    config_dict = {}
    config_ = ConfigParser()
    assert isinstance(file_path, str)
    assert os.path.exists(file_path)
    config_.read(file_path)
    for section_ in config_.sections():
        config_dict[section_] = {}
        for key_ in config_[section_]:
            try:
                config_dict[section_][key_] = ast.literal_eval(config_[section_][key_])
            except (ValueError, SyntaxError):
                config_dict[section_][key_] = config_[section_][key_]

    return config_dict


def set_config(section, key, value):
    """Set the value of a configuration parameter."""
    global config_dict

    if not config_dict:
        config_dict = get_all_config()

    if section not in config_dict:
        config_dict[section] = {}

    config_dict[section][key] = value

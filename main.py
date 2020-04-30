import sys

from src.snapconfig import config
from src.snapprocess import simulatespectra as sim

print(config.DEFAULT_PARAM_PATH)

params = config.get_config(section='input', key='spec_size')

print(params)

print(sim.get_spectrum('AAAA'))

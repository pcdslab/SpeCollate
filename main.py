import sys

from snapconfig import config
from snapprocess import simulatespectra as sim
from notebook import scratch

print(config.DEFAULT_PARAM_PATH)

params = config.get_config(section='input', key='spec_size')

print(params)

print(sim.get_spectrum('AAAA'))
scratch.currentworkingd()
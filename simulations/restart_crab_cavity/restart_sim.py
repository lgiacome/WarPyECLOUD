import sys
import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
from warp import restart, me, picmi

BIN = os.path.expanduser("../../")
if BIN not in sys.path:
    sys.path.append(BIN)

import numpy as np
#from warp_pyecloud_sim import warp_pyecloud_sim
#from chamber import CrabCavityWaveguide
#from plots import plot_field_crab

base_folder = str(Path(os.getcwd()).parent.parent)
cwd = str(Path(os.getcwd()))
folder = base_folder + '/dumps'
dumpfile = folder+ '/cavity.%d.dump' %me 

restart(dumpfile)
sim.reinit(laser_func, plots_crab)
n_steps = 1000
sim.tot_nsteps = 1000
sim.saver.extend_probe_vectors(n_steps)
sim.all_steps_no_ecloud()


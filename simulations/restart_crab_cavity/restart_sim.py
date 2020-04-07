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
sim.t_offs = 5.1355E-08
sim.n_bunches = 1
sim.bunch_macro_particles = 1e4
sim.reinit(laser_func, plots_crab,custom_time_prof = None)
n_steps = 100
sim.tot_nsteps = 100
sim.saver.extend_probe_vectors(n_steps)
sim.all_steps_no_ecloud()


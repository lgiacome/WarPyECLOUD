import sys
import os
from warp import *
from warp import picmi

BIN = os.path.expanduser("../../")
if BIN not in sys.path:
    sys.path.append(BIN)

import numpy as np
from warp_pyecloud_sim import warp_pyecloud_sim
from chamber import CrabCavity
from lattice_elements import CrabFields
import matplotlib.pyplot as plt

enable_trap = True

N_mp_max = 6000000
init_num_elecs = 2*10**7
dh = 3.e-4
width = 2*35e-3
height = 2*18e-3
nx = 200 #int(np.ceil(width/dh))
ny = 200 #int(np.ceil(height/dh))
nz = 100

# Compute sigmas
nemittx = 2.5e-6
nemitty = nemittx
beta_x = 100
beta_y = 100

beam_gamma = 479.
beam_beta = np.sqrt(1-1/(beam_gamma**2))
sigmax = np.sqrt(beta_x*nemittx/(beam_gamma*beam_beta))
sigmay = np.sqrt(beta_y*nemitty/(beam_gamma*beam_beta))
print(sigmax)
sigmat= 1.000000e-09/4.
max_z = 0.3

# Paths for the fields
fields_folder = str(Path(os.getcwd()).parent.parent)
efield_path = fields_folder + '/efield.txt'
hfield_path = fields_folder + '/hfield.txt'

chamber = CrabCavity(-max_z, max_z)
E_field_max = 57e6
lattice_elem = CrabFields(max_z, max_rescale = E_field_max, efield_path = efield_path, 
                          hfield_path = hfield_path)
n_bunches = 50

kwargs = {'enable_trap': enable_trap,
	'z_length': 1.,
	'nx': nx,
	'ny': ny, 
	'nz': nz,
	'n_bunches': n_bunches,
    'b_spac' : 25e-9,
    'beam_gamma': beam_gamma, 
	'sigmax': sigmax,
    'sigmay': sigmay, 
    'sigmat': sigmat,
    'bunch_intensity': 1.1e11, 
    'init_num_elecs': init_num_elecs,
    'init_num_elecs_mp': int(0.7*N_mp_max), 
    'pyecloud_nel_mp_ref': init_num_elecs/(0.7*N_mp_max),
	'dt': 25e-12,
    'pyecloud_fact_clean': 1e-6,
	'pyecloud_fact_split': 1.5,
    'Emax': 332., 
    'del_max': 1.7,
    'R0': 0.7, 
    'E_th': 35, 
    'sigmafit': 1.0828, 
    'mufit': 1.6636,
    'secondary_angle_distribution': 'cosine_3D', 
    'N_mp_max': N_mp_max,
    'N_mp_target': N_mp_max/3,
	'flag_checkpointing': True,
	'checkpoints': np.linspace(1, n_bunches, n_bunches),
    'temps_filename': 'complete_temp.mat',
    'flag_output': True,
    'bunch_macro_particles': 1e5,
    't_offs': 3*sigmat+1e-10,
    'output_filename': 'complete_out.mat',
    'images_dir': 'images',
    'flag_relativ_tracking': True,
    'lattice_elem': lattice_elem,
    'chamber': chamber,
}

sim = warp_pyecloud_sim(**kwargs)

sim.all_steps()

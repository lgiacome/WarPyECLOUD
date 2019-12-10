import sys
import os
from warp import *
from warp import picmi

BIN = os.path.expanduser("../../")
if BIN not in sys.path:
    sys.path.append(BIN)

import numpy as np
from warp_pyecloud_sim_new import warp_pyecloud_sim
from chamber import CrabCavity
from lattice_elements import CrabFields
import matplotlib.pyplot as plt

enable_trap = True

nz = 10
N_mp_max_slice = 1
init_num_elecs_slice = 1
dh = 3.e-4
width = 2*35e-3
height = 2*18e-3
nx = 10 #int(np.ceil(width/dh))
ny = 10 #int(np.ceil(height/dh))

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

chamber = CrabCavity(-1, 1)
lattice_elem = CrabFields(max_rescale = 57e6)

def plot_kick(self, l_force=0):
    mass = 1.6726e-27
    E_beam = np.sqrt((mass*self.beam.wspecies.getuz()*picmi.clight)**2 + (mass*picmi.clight**2)**2)
    print(E_beam)
    xp = np.divide(self.beam.wspecies.getuy(), self.beam.wspecies.getuz())
    Vt = E_beam*np.tan(xp)/picmi.echarge
    plt.plot(self.beam.wspecies.getz()-np.mean(self.beam.wspecies.getz()), Vt, 'x')
    plt.xlabel('s  [m]')
    plt.ylabel('Deflecting Voltage  [V]')
    plt.savefig('imgs_kick/kick_%d.png' %top.it)
    plt.clf()

kwargs = {'enable_trap': enable_trap,
	'z_length': 1.,
	'nx': nx,
	'ny': ny, 
	'nz': nz,
	'n_bunches': 50,
    'b_spac' : 25e-9,
    'beam_gamma': beam_gamma, 
	'sigmax': sigmax,
    'sigmay': sigmay, 
    'sigmat': sigmat,
    'bunch_intensity': 1.1e11, 
    'init_num_elecs': init_num_elecs_slice,
    'init_num_elecs_mp': N_mp_max_slice, 
	'By': 0.53549999999999998,
    'pyecloud_nel_mp_ref': 0,
	'dt': 25e-12,
    'pyecloud_fact_clean': 1e-6,
	'pyecloud_fact_split': 1.5,
    'chamber_type': 'rect', 
    'flag_save_video': True,
    'Emax': 332., 
    'del_max': 1.7,
    'R0': 0.7, 
    'E_th': 35, 
    'sigmafit': 1.0828, 
    'mufit': 1.6636,
    'secondary_angle_distribution': 'cosine_3D', 
    'N_mp_max': N_mp_max_slice,
    'N_mp_target': N_mp_max_slice,
	'flag_checkpointing': True,
	'checkpoints': np.linspace(1, 30, 30),
    'flag_output': True,
    'bunch_macro_particles': 1e3,
    't_offs': 3*sigmat+1e-10,
    'width' : width,
    'height' : height,
    'output_filename': 'warp_out.mat',
    'flag_relativ_tracking': True,
    'lattice_elem': lattice_elem,
    'chamber': chamber,
    'custom_plot': plot_kick
}

sim = warp_pyecloud_sim(**kwargs)

sim.all_steps()

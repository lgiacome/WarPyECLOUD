import sys
import os
from pathlib import Path
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

enable_trap = False

#Number of mesh nodes per direction
nx = 5
ny = 5
nz = 20

# Compute sigmas
nemittx = 2.5e-6
nemitty = nemittx
beta_x = 50
beta_y = 50
beam_gamma = 479.
beam_beta = np.sqrt(1-1/(beam_gamma**2))
sigmax = np.sqrt(beta_x*nemittx/(beam_gamma*beam_beta))
sigmay = np.sqrt(beta_y*nemitty/(beam_gamma*beam_beta))
sigmat= 1.000000e-09/4.

# Bounds of the cavity in z direction
min_z = -0.3
max_z = 1

# Paths for the fields
fields_folder = str(Path(os.getcwd()).parent.parent)
efield_path = fields_folder + '/efield_squared.txt'
hfield_path = fields_folder + '/hfield_squared.txt'

# Set up the chamber
chamber = CrabCavity(min_z, max_z)
E_field_max = 20e6

# Set up the RF field
t_offs = 3*sigmat+1e-10
#lattice_elem = CrabFields(min_z, max_rescale = E_field_max, efield_path = efield_path,
#                          hfield_path = hfield_path, chamber = chamber, t_offs = t_offs)
lattice_elem = None
n_bunches = 1

# Def plots
'''
def plot_kick(self, l_force=0):

    fontsz = 16
    plt.rcParams['axes.labelsize'] = fontsz
    plt.rcParams['axes.titlesize'] = fontsz
    plt.rcParams['xtick.labelsize'] = fontsz
    plt.rcParams['ytick.labelsize'] = fontsz
    plt.rcParams['legend.fontsize'] = fontsz
    plt.rcParams['legend.title_fontsize'] = fontsz

    mass = 1.6726e-27
    pz = mass*self.beam.wspecies.getuz()
    E_beam = np.sqrt((pz*picmi.clight)**2 + (mass*picmi.clight**2)**2)
    yp = self.beam.wspecies.getyp() #np.divide(self.beam.wspecies.getuy(), self.beam.wspecies.getuz())
    Vt = E_beam*np.tan(yp)/picmi.echarge
    plt.figure(figsize=(8,6))
    plt.plot(self.beam.wspecies.getz()-np.mean(self.beam.wspecies.getz()), Vt, 'x')
    plt.plot(np.zeros(100), np.linspace(-1e6, 1e6,100),'r--')
    plt.plot(np.linspace(-0.05,0.05,100), np.zeros(100), 'r--')
    plt.plot(np.linspace(-0.2,0.2,100), 3.96e6*np.ones(100), 'g--')
    plt.plot(np.linspace(-0.2,0.2,100), -3.96e6*np.ones(100), 'g--')
    plt.plot(np.linspace(-0.2,0.2,100), 2.36e6*np.ones(100), 'm--')
    plt.plot(np.linspace(-0.2,0.2,100), -2.36e6*np.ones(100), 'm--')
    plt.xlabel('s  [m]')
    plt.ticklabel_format(style = 'sci', axis = 'y', scilimits=(0,0))
    plt.ylabel('Deflecting Voltage  [V]')
    filename = self.images_dir + '/kick_' + repr(int(self.n_step)).zfill(4) + '.png'
    plt.savefig(filename)
    plt.close()
'''
kwargs = {'enable_trap': enable_trap,
    'solver_type': 'ES',
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
    'init_num_elecs': 0,
    'init_num_elecs_mp': 0, 
    'pyecloud_nel_mp_ref': 0,
	'dt': 25e-10,
    'pyecloud_fact_clean': 1e-6,
	'pyecloud_fact_split': 1.5,
    'Emax': 332., 
    'del_max': 1.7,
    'R0': 0.7, 
    'E_th': 35, 
    'sigmafit': 1.0828, 
    'mufit': 1.6636,
    'secondary_angle_distribution': 'cosine_3D', 
    'N_mp_max': 0,
    'N_mp_target': 0,
	'flag_checkpointing': False,
    'flag_output': True,
    'bunch_macro_particles': 1e3,
    't_offs': 3*sigmat+1e-10,
    'output_filename': 'measure_kick_out.h5',
    'images_dir': 'images_kick',
    'flag_relativ_tracking': True,
    'lattice_elem': lattice_elem,
    'chamber': chamber,
#    'custom_plot': plot_kick,
    'stride_imgs': 10
}

sim = warp_pyecloud_sim(**kwargs)

sim.all_steps()
sim.text_trap = None
sim.original = None

dump('try.dump')

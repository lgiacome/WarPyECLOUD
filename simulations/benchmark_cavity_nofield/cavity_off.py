from pathlib import Path
from warp import *
from warp import picmi, dump,  controllerfunctioncontainer
c_light = picmi.clight
import sys
import os
#BIN = os.path.expanduser("../../")
#if BIN not in sys.path:
#    sys.path.append(BIN)

import numpy as np
from warp_pyecloud_sim import warp_pyecloud_sim
from chamber import CrabCavityWaveguide
import saver as saver
from plots import plot_field_crab
import matplotlib.pyplot as plt
import mpi4py 

enable_trap = False

N_mp_max = 60000
init_num_elecs = 2*10**10

nx = 101
ny = 101
nz = 201

# Compute sigmas
nemittx = 2.5e-6
nemitty = nemittx
beta_x = 3500
beta_y = 3500

beam_gamma = 7000.
beam_beta = np.sqrt(1-1/(beam_gamma**2))
sigmax = np.sqrt(beta_x*nemittx/(beam_gamma*beam_beta))
sigmay = np.sqrt(beta_y*nemitty/(beam_gamma*beam_beta))
sigmat= 1.000000e-09/4.

max_z = 0.3

disp = 2e-3
chamber = CrabCavityWaveguide(-max_z, max_z, disp)

n_bunches = 50

fieldsolver_inputs = {'solver_type': 'EM', 'nx': nx, 'ny': ny, 'nz': nz, 'cfl': 1}

ecloud_inputs = {'init_num_elecs': init_num_elecs,
                 'init_num_elecs_mp': int(0.7*N_mp_max),
                 'pyecloud_nel_mp_ref': init_num_elecs/(0.7*N_mp_max),
                 'pyecloud_fact_clean': 1e-6, 'pyecloud_fact_split': 1.5,
                 'Emax': 332., 'del_max': 1.7, 'R0': 0.7, 'E_th': 35, 
                 'sigmafit': 1.0828, 'mufit': 1.6636,
                 'secondary_angle_distribution': 'cosine_3D', 
                 'N_mp_max': N_mp_max, 'N_mp_target': N_mp_max/3,
                 't_inject_elec': 0}

beam_inputs = {'n_bunches': n_bunches, 'b_spac': 25e-9, 
               'beam_gamma': beam_gamma, 'sigmax': sigmax, 'sigmay': sigmay, 
               'sigmat': sigmat, 'bunch_macro_particles': 1e5,
               't_offs': 2.5e-9, 'bunch_intensity': 1.1e11}


def noplots(self, l_force = 0):
    pass

sc_type = 'EM'

saving_inputs = {'images_dir': 'images_cavity_off',
                 'custom_plot': noplots, 'stride_imgs': 1000,
                 'output_filename': '/eos/user/l/lgiacome/benchmark_cavity_nofield/rect_dipole_'+sc_type+'_out.h5'}

simulation_inputs = {'enable_trap': enable_trap, 'chamber': chamber}

sim = warp_pyecloud_sim(fieldsolver_inputs = fieldsolver_inputs, 
                        beam_inputs = beam_inputs, 
                        ecloud_inputs = ecloud_inputs, 
                        saving_inputs = saving_inputs,
                        simulation_inputs = simulation_inputs)

if sc_type == 'ES':
    sim.add_es_solver()
    sim.distribute_species(primary_species=[sim.beam.wspecies], es_species=[sim.ecloud.wspecies])


sim.all_steps_no_ecloud()


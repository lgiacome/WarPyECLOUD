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

enable_trap = True

N_mp_max = 700000
init_num_elecs = 1*10**9

#dy = 4e-3
nghost_x = 2
nghost_y = 2
nghost_z = 2

nx = 100 + 2*nghost_x
ny = 120 + 2*nghost_y
nz = 200 + 2*nghost_z



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
chamber = CrabCavityWaveguide(-max_z, max_z, disp, ghost_x = nghost_x*4e-3, ghost_y = nghost_y*2e-3, ghost_z = nghost_z*3e-3)
print('#########################')
print(chamber.ymin)
laser_source_z = chamber.zmin + 0.5*(-chamber.l_main_z/2 - chamber.zmin)

n_bunches = 10

fieldsolver_inputs = {'solver_type': 'EM', 'nx': nx, 'ny': ny, 'nz': nz, 'cfl': 1, 'source_smoothing': True}

sc_type = 'EM'
init_ecloud_fields = True
ecloud_inputs = {'init_num_elecs': init_num_elecs,
                 'init_num_elecs_mp': int(0.7*N_mp_max),
                 'pyecloud_nel_mp_ref': init_num_elecs/(0.7*N_mp_max),
                 'pyecloud_fact_clean': 1e-6, 'pyecloud_fact_split': 1.5,
                 'Emax': 332., 'del_max': 1.7, 'R0': 0.7, 'E_th': 35, 
                 'sigmafit': 1.0828, 'mufit': 1.6636,
                 'secondary_angle_distribution': 'cosine_3D', 
                 'N_mp_max': N_mp_max, 'N_mp_target': N_mp_max/3,
                 't_inject_elec': 0, 'init_ecloud_fields': init_ecloud_fields}

beam_inputs = {'n_bunches': n_bunches, 'b_spac': 25e-9, 
               'beam_gamma': beam_gamma, 'sigmax': sigmax, 'sigmay': sigmay, 
               'sigmat': sigmat, 'bunch_macro_particles': 1e5,
               't_offs': 2.5e-9, 'bunch_intensity': 1.1e11}


def plot_fields(self, l_force = 0):
    chamber = self.chamber
    em = self.solver.solver

    if self.laser_source_z is None: here_laser_source_z = chamber.zmin + 0.5*(-chamber.l_main_z/2-chamber.zmin)
    else: here_laser_source_z = self.laser_source_z
    k_antenna = int((here_laser_source_z - chamber.zmin)/em.dz)
    j_mid_waveguide = int((chamber.ycen6 - chamber.ymin)/em.dy)

    flist = ['Ex','Ey','Ez','Bx','By','Bz', 'elecs']
    pw = picmi.warp
    if pw.top.it%self.stride_imgs==0 or l_force:
        #fig = plt.figure( figsize=(7,7))
        for ffstr in flist:
            ff = None
            if ffstr == 'Ex': ff = em.gatherex()
            if ffstr == 'Ey': ff = em.gatherey()
            if ffstr == 'Ez': ff = em.gatherez()
            if ffstr == 'Bx': ff = em.gatherbx()
            if ffstr == 'By': ff = em.gatherby()
            if ffstr == 'Bz': ff = em.gatherbz()
            if ff is not None:
                maxe = np.max(ff[:,:,:])
                mine = np.min(ff[:,:,:])
            if ffstr == 'elecs':
                ff = self.ecloud.wspecies.get_density()
                maxe = np.max(ff)
                mine = np.min(ff)
            if pw.me==0:
                plot_field_crab(ff, ffstr, mine, maxe, k_antenna, j_mid_waveguide, chamber, images_dir = self.images_dir)


def noplots(self, l_force = 0):
    pass


saving_inputs = {'images_dir': '/eos/user/l/lgiacome/benchmark_cavity_nofield/images_cavity_off_'+sc_type+'_adj_newsmooth_relat_fix/', 
                 'custom_plot': plot_fields, 'stride_imgs': 1000, 'stride_output': 1000,
                 'output_filename': '/eos/user/l/lgiacome/benchmark_cavity_nofield/cavity_nofield_'+sc_type+'_adj_newsmooth_relat_fix_out.h5'}

simulation_inputs = {'enable_trap': enable_trap, 'chamber': chamber}

sim = warp_pyecloud_sim(fieldsolver_inputs = fieldsolver_inputs, 
                        beam_inputs = beam_inputs, 
                        ecloud_inputs = ecloud_inputs, 
                        saving_inputs = saving_inputs,
                        simulation_inputs = simulation_inputs)

sim.laser_source_z = laser_source_z

sim.add_es_solver()
sim.distribute_species(primary_species=[sim.ecloud.wspecies], es_species=[sim.beam.wspecies])

#for i in range(10):
#    print(np.sum(sim.ecloud.wspecies.getn()))
#    picmi.warp.step()

sim.all_steps_no_ecloud()

dump_folder = '/eos/user/l/lgiacome/benchmark_cavity_nofield/dumps'

if picmi.warp.me == 0 and not os.path.exists(dump_folder):
    os.mkdir(dump_folder)

sim.dump(dump_folder + '/cavity_nofield_adj_nosmooth_fix.dump')

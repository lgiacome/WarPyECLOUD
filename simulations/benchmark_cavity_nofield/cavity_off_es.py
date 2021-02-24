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
laser_source_z = chamber.zmin + 0.5*(-chamber.l_main_z/2 - chamber.zmin)

n_bunches = 10

fieldsolver_inputs = {'solver_type': 'EM', 'nx': nx, 'ny': ny, 'nz': nz, 'cfl': 1}

sc_type = 'ES'
init_ecloud_fields = False
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
    pw = picmi.warp 

    if self.laser_source_z is None: here_laser_source_z = chamber.zmin + 0.5*(-chamber.l_main_z/2-chamber.zmin)
    else: here_laser_source_z = self.laser_source_z
    k_antenna = int((here_laser_source_z - chamber.zmin)/em.dz)
    j_mid_waveguide = int((chamber.ycen6 - chamber.ymin)/em.dy)
    if pw.top.it%self.stride_imgs==0 or l_force:
        #fig = plt.figure( figsize=(7,7))
        Ex = sim.ES_solver.solver.getex()
        Ey = sim.ES_solver.solver.getey()
        Ez = sim.ES_solver.solver.getez()
        elecs = self.ecloud.wspecies.get_density()
        if picmi.warp.me>0:
            mpisend(Ex, tag=8)
            mpisend(Ey, tag=9)
            mpisend(Ez, tag=10)
        if picmi.warp.me==0:
            #nx_loc, ny_loc, nz_loc = np.shape(Ex)
            #Ex_g = np.zeros((nx_loc, ny_loc, npes*nz_loc))
            Ex_g = Ex
            #Ey_g = np.zeros((nx_loc, ny_loc, npes*nz_loc))
            Ey_g = Ey
            #Ez_g = np.zeros((nx_loc, ny_loc, npes*nz_loc))
            Ez_g = Ez
            #count = nz_loc
            for nproc in range(1, npes):
                Ex_l = mpirecv(source=nproc, tag=8)
                #nx_loc, ny_loc, nz_loc = np.shape(Ex_l)
                Ex_g = np.concatenate((Ex_g, Ex_l), axis = 2)
                Ey_l = mpirecv(source=nproc, tag=9)
                #nx_loc, ny_loc, nz_loc = np.shape(Ey_l)
                Ey_g = np.concatenate((Ey_g, Ey_l), axis = 2)
                Ez_l = mpirecv(source=nproc, tag=10)
                #nx_loc, ny_loc, nz_loc = np.shape(Ez_l)
                Ez_g = np.concatenate((Ez_g, Ez_l), axis = 2)
                #count+=nz_loc
            plot_field_crab(Ex_g, 'Ex', np.min(Ex_g), np.max(Ex_g), k_antenna, j_mid_waveguide, chamber, self.images_dir)
            plot_field_crab(Ey_g, 'Ey', np.min(Ey_g), np.max(Ey_g), k_antenna, j_mid_waveguide, chamber, self.images_dir)
            plot_field_crab(Ez_g, 'Ez', np.min(Ez_g), np.max(Ez_g), k_antenna, j_mid_waveguide, chamber, self.images_dir)
            plot_field_crab(elecs, 'elecs', np.min(elecs), np.max(elecs), k_antenna, j_mid_waveguide, chamber, self.images_dir)

def noplots(self, l_force = 0):
    pass


saving_inputs = {'images_dir': '/eos/user/l/lgiacome/benchmark_cavity_nofield/images_cavity_off_ES_adj_relat_fix/', 
                 'custom_plot': plot_fields, 'stride_imgs': 1000, 'stride_output': 1000,
                 'output_filename': '/eos/user/l/lgiacome/benchmark_cavity_nofield/cavity_nofield_'+sc_type+'_adj_relat_out_fix.h5'}

simulation_inputs = {'enable_trap': enable_trap, 'chamber': chamber}

sim = warp_pyecloud_sim(fieldsolver_inputs = fieldsolver_inputs, 
                        beam_inputs = beam_inputs, 
                        ecloud_inputs = ecloud_inputs, 
                        saving_inputs = saving_inputs,
                        simulation_inputs = simulation_inputs)

sim.laser_source_z = laser_source_z

sim.add_es_solver()
sim.distribute_species(primary_species=[], es_species=[sim.beam.wspecies, sim.ecloud.wspecies])

#for i in range(10):
#    print(np.sum(sim.ecloud.wspecies.getn()))
#    picmi.warp.step()

sim.all_steps_no_ecloud()

dump_folder = '/eos/user/l/lgiacome/benchmark_cavity_nofield/dumps'

if picmi.warp.me == 0 and not os.path.exists(dump_folder):
    os.mkdir(dump_folder)

sim.dump(dump_folder + '/cavity_nofield_fix_es.dump')

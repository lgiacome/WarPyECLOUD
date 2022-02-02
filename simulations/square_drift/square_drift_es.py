import sys
import os
from warp import picmi

BIN = os.path.expanduser("../../")
if BIN not in sys.path:
    sys.path.append(BIN)

import numpy as np
from  warp_pyecloud_sim import warp_pyecloud_sim
from chamber import RectChamber
from lattice_elements import Dipole
import matplotlib.pyplot as plt
from plots import plot_fields
from warp import clight, mpisend, mpirecv, npes

enable_trap = False

nz = 100
nz_sol = 100
N_mp_max_slice = 10000
init_num_elecs_slice = 2*10**6
dh = 6.e-4
dz = 0.01
width = 2*18e-3
height = 2*18e-3
z_length = 1.
z_start = -z_length/2
z_end = z_length/2
By = 0.53549999999999998

chamber = RectChamber(width, height, z_start, z_end, ghost_x = 3*dh, ghost_y = 3*dh, ghost_z = 3*dz)

nx = int(np.ceil((chamber.xmax-chamber.xmin)/dh))
ny = int(np.ceil((chamber.ymax-chamber.ymin)/dh))
nz = int(np.ceil((chamber.zmax-chamber.zmin)/dz))

# Compute sigmas
nemittx = 2.5e-6
nemitty = nemittx
beta_x = 85
beta_y = 85

beam_gamma = 479.
beam_beta = np.sqrt(1-1/(beam_gamma**2))
sigmax = np.sqrt(beta_x*nemittx/(beam_gamma*beam_beta))
sigmay = np.sqrt(beta_y*nemitty/(beam_gamma*beam_beta))
n_bunches = 10
print(sigmax)

def plots_square(self, l_force = 0):
    chamber = self.chamber
    em = self.solver.solver
    
    pw = picmi.warp
    if pw.top.it%self.stride_imgs==0 or l_force:
        #fig = plt.figure( figsize=(7,7))
        Ex = sim.es_solver.solver.getex()
        Ey = sim.es_solver.solver.getey()
        Ez = sim.es_solver.solver.getez()
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
            
            plot_fields(Ex_g, 'Ex', np.min(Ex), np.max(Ex), chamber, self.images_dir)
            plot_fields(Ey_g, 'Ey', np.min(Ey), np.max(Ey), chamber, self.images_dir)
            plot_fields(Ez_g, 'Ez', np.min(Ez), np.max(Ez), chamber, self.images_dir)

def noplots(self, l_force=0):
    pass

sc_type = 'ES'

fieldsolver_inputs = {'solver_type':'EM', 'nx': nx, 'ny': ny, 'nz': nz, 'cfl': 1, 'source_smoothing': True}
t_offs = 2.5e-9
#t_offs = 1
beam_inputs = {'n_bunches': n_bunches, 'b_spac' : 25e-9,
               'beam_gamma': beam_gamma, 'sigmax': sigmax, 'sigmay': sigmay, 
               'sigmat': 1.000000e-09/4., 'bunch_macro_particles': 1e5,
               't_offs': t_offs, 'bunch_intensity': 1.1e11}

ecloud_inputs = {'init_num_elecs': init_num_elecs_slice*nz,
              'init_num_elecs_mp': int(0.7*N_mp_max_slice*nz),    
              'pyecloud_nel_mp_ref': init_num_elecs_slice/(0.7*N_mp_max_slice),
              'pyecloud_fact_clean': 1e-6, 'pyecloud_fact_split': 1.5,
              'Emax': 332., 'del_max': 1.7, 'R0': 0.7, 'E_th': 35,
              'sigmafit': 1.0828, 'mufit': 1.6636,
              'secondary_angle_distribution': 'cosine_3D', 
              'N_mp_max': N_mp_max_slice*nz,'N_mp_target': N_mp_max_slice/3*nz, 'init_ecloud_fields': False}

saving_inputs = {'images_dir': '/eos/user/l/lgiacome/benchmark_warp_pyecloud/square_drift_nobunch_'+sc_type+'_images',
                 'custom_plot': noplots, 'stride_imgs': 100, 'stride_output': 1000,
                 'output_filename': '/eos/user/l/lgiacome/benchmark_warp_pyecloud/square_drift/square_drift_'+sc_type+'_out.h5'}

simulation_inputs = {'enable_trap': enable_trap,
                     'flag_relativ_tracking': True,
                     'chamber': chamber}
#                     'tot_nsteps': int(n_bunches*25e-9/25e-12)}

sim = warp_pyecloud_sim(fieldsolver_inputs = fieldsolver_inputs, 
                        beam_inputs = beam_inputs, 
                        ecloud_inputs = ecloud_inputs,
                        saving_inputs = saving_inputs,
                        simulation_inputs = simulation_inputs)

newsteps = int(np.round(25e-9*n_bunches/picmi.warp.top.dt))
sim.tot_nsteps = newsteps
sim.saver.tot_nsteps = newsteps

sim.add_es_solver()
sim.distribute_species(primary_species=[], es_species=[sim.beam.wspecies, sim.ecloud.wspecies])

sim.all_steps_no_ecloud()
#for i in range(3):
#    picmi.warp.step()
    #print(sim.ecloud.wspecies.getn())
    #print(np.sum(sim.ecloud.wspecies.getw()))

fieldsolver_inputs = beam_inputs = ecloud_inputs = antenna_inputs = None
saving_inputs = simulations_inputs = None

dump_folder = '/eos/user/l/lgiacome/benchmark_warp_pyecloud/dumps'

if picmi.warp.me == 0 and not os.path.exists(dump_folder):
    os.mkdir(dump_folder)

sim.dump(dump_folder + '/square_drift_es.dump')


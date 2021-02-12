import sys
import os
from warp import picmi, clight

BIN = os.path.expanduser("../../")
if BIN not in sys.path:
    sys.path.append(BIN)

import numpy as np
from  warp_pyecloud_sim import warp_pyecloud_sim
from chamber import RectChamber
from lattice_elements import Dipole
import matplotlib.pyplot as plt
from plots import plot_fields

enable_trap = False #True

nz = 100
nz_sol = 100
N_mp_max_slice = 10000
init_num_elecs_slice = 2*10**6
dx = dy = dh = 6.e-4
dz = 0.01
width = 2*18e-3
height = 2*18e-3
z_length = 1
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

    flist = ['Ex','Ey','Ez','Bx','By','Bz','elecs']
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
                ff = self.ecloud.wspecies.get_density() + self.beam.wspecies.get_density()
                maxe = 5e9 #np.max(ey[:,:,:])
                mine = 0 #np.min(ey[:,:,:])
                maxe = np.max(self.ecloud.wspecies.get_density())
                mine = np.min(self.ecloud.wspecies.get_density())
            if pw.me==0:
                plot_fields(ff, ffstr, mine, maxe, chamber, self.images_dir)

def noplots(self, l_force=0):
    pass

sc_type = 'EM'

#dt = 1./(clight*np.sqrt(1./dx**2+1./dy**2+1./dz**2))

fieldsolver_inputs = {'solver_type':sc_type, 'nx': nx, 'ny': ny, 'nz': nz, 'cfl': 1, 'source_smoothing': True}

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
              'N_mp_max': N_mp_max_slice*nz,'N_mp_target': N_mp_max_slice/3*nz, 'init_ecloud_fields': True}

saving_inputs = {'images_dir': '/eos/user/l/lgiacome/benchmark_warp_pyecloud/square_drift_'+sc_type+'_images',
                 'custom_plot': plots_square, 'stride_imgs': 1, 'stride_output': 1,
                 'output_filename': '/eos/user/l/lgiacome/benchmark_warp_pyecloud/square_drift/square_drift_'+sc_type+'_newsmooth_out.h5'}

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
sim.distribute_species(primary_species=[sim.ecloud.wspecies], es_species=[sim.beam.wspecies])

#print(sim.ecloud.wspecies.getn())
sim.all_steps_no_ecloud()
#for i in range(3):
#    picmi.warp.step()
    #print(sim.ecloud.wspecies.getn())
    #print(np.sum(sim.ecloud.wspecies.getw()))

dump_folder = '/eos/user/l/lgiacome/benchmark_warp_pyecloud/dumps'

if picmi.warp.me == 0 and not os.path.exists(dump_folder):
    os.mkdir(dump_folder)

sim.dump(dump_folder + '/square_drift_newsmooth.dump')



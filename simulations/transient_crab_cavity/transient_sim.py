from pathlib import Path
from warp import *
from warp import picmi, dump,  controllerfunctioncontainer
c_light = picmi.clight
import sys
import os
BIN = os.path.expanduser("../../")
if BIN not in sys.path:
    sys.path.append(BIN)

import numpy as np
from warp_pyecloud_sim import warp_pyecloud_sim
from chamber import CrabCavityWaveguide
import saver as saver
from plots import plot_field_crab
import matplotlib.pyplot as plt
import mpi4py 

enable_trap = True

N_mp_max = 0
init_num_elecs = 0

nx = 100
ny = 75
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

disp = 5e-3
chamber = CrabCavityWaveguide(-max_z, max_z, disp)

n_bunches = 1 

freq_t = 400e6
phase_delay = np.pi/2
r_time = 10*2.5e-9

# Antenna parameters
laser_source_z = chamber.zmin + 0.5*(-chamber.l_main_z/2 - chamber.zmin) 
laser_polangle = np.pi/2
laser_emax = 10
laser_xmin = chamber.x_min_wg
laser_xmax = chamber.x_max_wg
laser_ymin = chamber.y_min_wg
laser_ymax = chamber.y_max_wg

ramp = lambda tt: 0.5*(1-np.cos(np.pi*tt))

def amplitude(t, Emax, r_time):
    if t < 2*r_time:
        return Emax*ramp(t/r_time)
    else:
        return 0

def laser_func(y, x, t):
    w_t = 2*np.pi*freq_t
    width = laser_xmax - laser_xmin
    w_z = c_light*np.sqrt((w_t/c_light)**2 - (np.pi/width)**2)
    if t<2*r_time:
        return amplitude(t, laser_emax, r_time)*np.sin(-np.pi/width*(x+width/2))*(np.sin(w_t*t-phase_delay)*np.cos(w_z*laser_source_z) - np.cos(w_t*t-phase_delay)*np.sin(w_z*laser_source_z))
    else:
        return 0

def plots_crab(self, l_force = 0):
    fontsz = 16
    plt.rcParams['axes.labelsize'] = fontsz
    plt.rcParams['axes.titlesize'] = fontsz
    plt.rcParams['xtick.labelsize'] = fontsz
    plt.rcParams['ytick.labelsize'] = fontsz
    plt.rcParams['legend.fontsize'] = fontsz
    plt.rcParams['legend.title_fontsize'] = fontsz

    chamber = self.chamber
    if self.laser_source_z is None: 
        self.laser_source_z = chamber.zmin + 0.5*(-chamber.l_main_z/2-chamber.zmin)
    em = self.solver.solver
    k_antenna = int((self.laser_source_z - chamber.zmin)/em.dz)
    j_mid_waveguide = int((chamber.ycen6 - chamber.ymin)/em.dy)

    flist = ['Ex','Ey','Ez','Bx','By','Bz']
    flist = ['Ey']
    pw = picmi.warp
    if pw.top.it%1==0 or l_force:
        #fig = plt.figure( figsize=(7,7))
        for ffstr in flist:
            if ffstr == 'Ex': ff = em.gatherex()
            if ffstr == 'Ey':
                ff = em.gatherey()
                maxe =35*self.em_scale_fac #np.max(ey[:,:,:])
                mine = -35*self.em_scale_fac #np.min(ey[:,:,:])
            if ffstr == 'Ez': ff = em.gatherez()
            if ffstr == 'Bx': ff = em.gatherbx()
            if ffstr == 'By': ff = em.gatherby()
            if ffstr == 'Bz': ff = em.gatherbz()
            if ffstr == 'elecs':
                ff = self.ecloud.wspecies.get_density()
                maxe = 5e9 #np.max(ey[:,:,:])
                mine = 0 #np.min(ey[:,:,:])
            if me==0:
                plot_field_crab(ff, ffstr, mine, maxe, k_antenna, j_mid_waveguide, chamber)


def plots_crab(self, l_force = 0):
    print('plotting something')
    pass

field_probes = [[int(nx/2),int(ny/2),int(nz/2)]]

kwargs = {'enable_trap': enable_trap,
    'ecloud_sim': False,
    'solver_type': 'EM',
	'nx': nx,
	'ny': ny, 
	'nz': nz,
    'Emax': 332., 
    'del_max': 1.7,
    'R0': 0.7, 
    'E_th': 35, 
    'sigmafit': 1.0828, 
    'mufit': 1.6636,
    'secondary_angle_distribution': 'cosine_3D', 
    'pyecloud_nel_mp_ref': 0,
    'pyecloud_fact_clean': 0,
    'pyecloud_fact_split': 0,
    'N_mp_max': N_mp_max,
    'N_mp_target': N_mp_max/3,
	'flag_checkpointing': True,
	'checkpoints': np.linspace(1, n_bunches, n_bunches),
    'temps_filename': 'complete_temp.h5',
    'flag_output': True,
    'bunch_macro_particles': 0,
    'sigmat': sigmat,
    'init_num_elecs': 0,
    'init_num_elecs_mp': 1,
    'bunch_macro_particles': 0,
    'b_spac': 25e-9,
    'beam_gamma': 1,
    't_offs': 3*sigmat+1e-10,
    'images_dir': 'images_transient',
    'chamber': chamber,
    'custom_plot': plots_crab,
    'stride_imgs': 10,
    'EM_method': 'Yee',
    'cfl': 1.,
    'laser_func': laser_func,
    'laser_source_z': laser_source_z,
    'laser_polangle': laser_polangle,
    'laser_emax': laser_emax,
    'laser_xmin': laser_xmin,
    'laser_xmax': laser_xmax,
    'laser_ymin': laser_ymin,
    'laser_ymax': laser_ymax,
    'tot_nsteps': 5000,
    'field_probes': field_probes,
    'field_probes_dump_stride': 100
}

sim = warp_pyecloud_sim(**kwargs)
sim.all_steps_no_ecloud()
kwargs = None
base_folder = str(Path(os.getcwd()).parent.parent)
cwd = str(Path(os.getcwd()))
folder = base_folder + '/dumps'
if picmi.warp.me == 0 and not os.path.exists(folder):
    os.makedirs(folder)
sim.dump(folder+ '/cavity.%d.dump' %picmi.warp.me)


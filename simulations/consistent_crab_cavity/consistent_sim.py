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
from chamber import CrabCavityWaveguide
from plots import plot_field_crab
import matplotlib.pyplot as plt
import matplotlib as mpl

enable_trap = True

N_mp_max = 6000000
init_num_elecs = 2*10**7
dh = 3.e-4

nx = 100
ny = 100
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

bunch_intensity = 0
b_spac = 25e-9
n_bunches = 1 

chamber = CrabCavityWaveguide(-max_z, max_z, disp = 5e-3)
#E_field_max = 5e6 #57

init_em_fields = True
fields_folder = str(Path(os.getcwd()).parent.parent)
file_em_fields = 'em_fields.h5'
folder_em_fields = fields_folder + '/em_fields'
em_scale_fac = 1 #(10/35)*1e6

def plots_crab(self, l_force = 0):
    fontsz = 16
    plt.rcParams['axes.labelsize'] = fontsz
    plt.rcParams['axes.titlesize'] = fontsz
    plt.rcParams['xtick.labelsize'] = fontsz
    plt.rcParams['ytick.labelsize'] = fontsz
    plt.rcParams['legend.fontsize'] = fontsz
    plt.rcParams['legend.title_fontsize'] = fontsz

    chamber = self.chamber
    if self.laser_source_z is None: self.laser_source_z = chamber.zmin + 0.5*(-chamber.l_main_z/2-chamber.zmin)
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

kwargs = {'enable_trap': enable_trap,
    'solver_type': 'EM',
    'nx': nx,
    'ny': ny, 
    'nz': nz,
    'n_bunches': n_bunches,
    'b_spac' : b_spac,
    'beam_gamma': beam_gamma, 
    'sigmax': sigmax,
    'sigmay': sigmay, 
    'sigmat': sigmat,
    'bunch_intensity': bunch_intensity, 
    'init_num_elecs': init_num_elecs,
    'init_num_elecs_mp': int(0.7*N_mp_max), 
    'pyecloud_nel_mp_ref': init_num_elecs/(0.7*N_mp_max),
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
    'flag_checkpointing': False,
    'checkpoints': np.linspace(1, n_bunches, n_bunches),
    'temps_filename': 'complete_temp.h5',
    'flag_output': True,
    'bunch_macro_particles': 1e5,
    't_offs': 3*sigmat+1e-10,
    'output_filename': 'complete_out.h5',
    'images_dir': 'images',
    'flag_relativ_tracking': True,
    'chamber': chamber,
    'custom_plot': plots_crab,
    'stride_imgs': 1,
    'init_em_fields': init_em_fields,
    'file_em_fields': file_em_fields,
    'em_scale_fac': em_scale_fac,
    'folder_em_fields': folder_em_fields,
}

sim = warp_pyecloud_sim(**kwargs)
n_steps = 1000
sim.all_steps_no_ecloud(n_steps)


import sys
import os
from pathlib import Path
from warp import picmi, f3d, clight

BIN = os.path.expanduser("../../")
if BIN not in sys.path:
    sys.path.append(BIN)

import numpy as np
from  warp_pyecloud_sim import warp_pyecloud_sim
from chamber import CrabCavityRound
from plots import plot_fields
import matplotlib.pyplot as plt
from scipy.constants import c

enable_trap = False

N_mp_max = 600000
init_num_elecs = 2*10**7
t_offs = 2.5e-9/4 + 3e-10

# Paths for the fields
fields_folder = str(Path(os.getcwd()).parent.parent)
#efield_path = fields_folder + '/e_simple_dqw.txt'
#hfield_path = fields_folder + '/h_simple_dqw.txt'
nx = 130
ny = 130
nz = 200

filename = 'simple_DQW.msh'
chamber = CrabCavityRound(nz = nz)

# Compute sigmas
nemittx = 2.5e-6
nemitty = nemittx
beta_x = 3500
beta_y = 3500

beam_gamma = 7000.
beam_beta = np.sqrt(1-1/(beam_gamma**2))
sigmax = np.sqrt(beta_x*nemittx/(beam_gamma*beam_beta))
sigmay = np.sqrt(beta_y*nemitty/(beam_gamma*beam_beta))
n_bunches = 72
print(sigmax)

def plot_kick(self, l_force=0):
    fontsz = 16
    plt.rcParams['axes.labelsize'] = fontsz
    plt.rcParams['axes.titlesize'] = fontsz
    plt.rcParams['xtick.labelsize'] = fontsz
    plt.rcParams['ytick.labelsize'] = fontsz
    plt.rcParams['legend.fontsize'] = fontsz
    plt.rcParams['legend.title_fontsize'] = fontsz

    mass = 1.6726e-27
    if self.beam.wspecies.getn()>0:
        pz = mass*self.beam.wspecies.getuz()
        E_beam = np.sqrt((pz*picmi.clight)**2 + (mass*picmi.clight**2)**2)
        yp = self.beam.wspecies.getyp() #np.divide(self.beam.wspecies.getuy(), self.beam.wspecies.getuz())
        Vt = E_beam*np.tan(yp)/picmi.echarge
        plt.figure(figsize=(8,6))
        plt.plot(self.beam.wspecies.getz()-np.mean(self.beam.wspecies.getz()), Vt, 'x')
        plt.plot(np.zeros(100), np.linspace(-1e6, 1e6,100),'r--')
        plt.plot(np.linspace(-0.05,0.05,100), np.zeros(100), 'r--')
    #    plt.plot(np.linspace(-0.2,0.2,100), 3.96e6*np.ones(100), 'g--')
    #    plt.plot(np.linspace(-0.2,0.2,100), -3.96e6*np.ones(100), 'g--')
    #    plt.plot(np.linspace(-0.2,0.2,100), 2.36e6*np.ones(100), 'm--')
    #    plt.plot(np.linspace(-0.2,0.2,100), -2.36e6*np.ones(100), 'm--')
        plt.xlabel('s  [m]')
        plt.ticklabel_format(style = 'sci', axis = 'y', scilimits=(0,0))
        plt.ylabel('Deflecting Voltage  [V]')
        filename = 'images_kick/kick_' + repr(int(self.n_step)).zfill(4) + '.png'
        plt.savefig(filename)
        plt.close('all')

def plot_dens(self, l_force=0):
    ffstr = 'bunch'
    mine = 0
    maxe = 5e9
    chamber = self.chamber
    images_dir = self.images_dir
    if picmi.warp.top.it%self.stride_imgs==0:
        ff = self.ecloud.wspecies.get_density() + self.beam.wspecies.get_density()
        plot_fields(ff, ffstr, mine, maxe, chamber, images_dir)

dz = (chamber.zmax-chamber.zmin)/nz
dt = dz/clight
fieldsolver_inputs = {'solver_type':'ES', 'nx': nx, 'ny': ny, 'nz': nz,
                      'dt': dt}

beam_inputs = {'n_bunches': n_bunches, 'b_spac' : 25e-9,
               'beam_gamma': beam_gamma, 'sigmax': sigmax, 'sigmay': sigmay, 
               'sigmat': 1.000000e-09/4., 'bunch_macro_particles': 1e6,
               't_offs': t_offs, 'bunch_intensity': 1.1e11}

ecloud_inputs = {'init_num_elecs': init_num_elecs,
              'init_num_elecs_mp': int(0.7*N_mp_max),    
              'pyecloud_nel_mp_ref': init_num_elecs/(0.7*N_mp_max),
              'pyecloud_fact_clean': 1e-6, 'pyecloud_fact_split': 1.5,
              'Emax': 332., 'del_max': 1.7, 'R0': 0.7, 'E_th': 35,
              'sigmafit': 1.0828, 'mufit': 1.6636,
              'secondary_angle_distribution': 'cosine_3D', 
              'N_mp_max': N_mp_max,'N_mp_target': N_mp_max/3, 't_inject_elec': 0} 

def noplots(self, l_force=1):
    pass

saving_inputs = {'images_dir': 'cavity_triang_imgs',
#                 'images_dir': '/eos/user/l/lgiacome/cavity_triang/cavity_triang_imgs',
                 'custom_plot': plot_dens, 'stride_imgs': 1,
                 'output_filename': 'cavity_triang_out.h5', 
#                 'output_filename': '/eos/user/l/lgiacome/cavity_triang/cavity_triang_out.h5', 
                 'flag_checkpointing': False,
                 'checkpoints': np.linspace(1,n_bunches,n_bunches),
                 'stride_output': 1, 
#                 'temps_filename': '/eos/user/l/lgiacome/cavity_triang/cavity_triang_temp.h5', 'flag_save_ek_impacts': True}
                 'temps_filename': 'cavity_triang_temp.h5', 'flag_save_ek_impacts': True}

simulation_inputs = {'enable_trap': enable_trap, 'chamber': chamber,
                     't_end': 25e-9*n_bunches}

f3d.mgmaxiters = 300
sim = warp_pyecloud_sim(fieldsolver_inputs = fieldsolver_inputs, 
                        beam_inputs = beam_inputs, 
                        ecloud_inputs = ecloud_inputs,
                        saving_inputs = saving_inputs,
                        simulation_inputs = simulation_inputs)
#breakpoint()
sim.all_steps_no_ecloud()

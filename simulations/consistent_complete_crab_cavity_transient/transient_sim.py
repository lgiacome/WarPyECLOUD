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

N_mp_max = 6000
init_num_elecs = 2*10**7

nx = 50
ny = 50
nz = 50

# Compute sigmas
nemittx = 2.5e-6
nemitty = nemittx
beta_x = 100
beta_y = 100

beam_gamma = 479.
beam_beta = np.sqrt(1-1/(beam_gamma**2))
sigmax = np.sqrt(beta_x*nemittx/(beam_gamma*beam_beta))
sigmay = np.sqrt(beta_y*nemitty/(beam_gamma*beam_beta))
sigmat= 1.000000e-09/4.

max_z = 0.3

disp = 2e-3
chamber = CrabCavityWaveguide(-max_z, max_z, disp)

n_bunches = 1 

# Antenna parameters
laser_source_z = chamber.zmin + 0.5*(-chamber.l_main_z/2 - chamber.zmin) 
laser_polangle = np.pi/2
laser_emax = 10e6
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
    freq_t = 400e6
    r_time = 10*2.5e-9
    phase_delay = 0
    w_t = 2*np.pi*freq_t
    width = laser_xmax - laser_xmin
    r_time = 10*2.5e-9
    w_z = c_light*np.sqrt((w_t/c_light)**2 - (np.pi/width)**2)
    source_z = laser_source_z
    if t<2*r_time:
        return amplitude(t, laser_emax, r_time)*np.sin(-np.pi/width*(x+width/2))*(np.sin(w_t*t-phase_delay)*np.cos(w_z*source_z) - np.cos(w_t*t-phase_delay)*np.sin(w_z*source_z))
    else:
        return 0


def plot_kick(self, l_force=0):
    if self.n_step > 3500:
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
            filename = self.images_dir + '/kick_' + repr(int(self.n_step)).zfill(4) + '.png'
            plt.savefig(filename)
            plt.close('all')

field_probes = [[int(nx/2),int(ny/2),int(nz/2)]]
t_offs = 5.3e-8-2.5e-9/4+2.6685127615852166e-10 - 3.335640951981521e-11

fieldsolver_inputs = {'nx': nx, 'ny': ny, 'nz': nz, 'solver_type': 'EM',
                      'EM_method': 'Yee', 'cfl': 1.}

ecloud_inputs = {'init_num_elecs': init_num_elecs,
                 'init_num_elecs_mp': int(0.7*N_mp_max),
                 'pyecloud_nel_mp_ref': init_num_elecs/(0.7*N_mp_max),
                 'pyecloud_fact_clean': 1e-6, 'pyecloud_fact_split': 1.5,
                 'Emax': 332., 'del_max': 1.7, 'R0': 0.7, 'E_th': 35, 
                 'sigmafit': 1.0828, 'mufit': 1.6636,
                 'secondary_angle_distribution': 'cosine_3D', 
                 'N_mp_max': N_mp_max, 'N_mp_target': N_mp_max/3,
                 't_inject_elec': 5.1e-8}

beam_inputs = {'n_bunches': n_bunches, 'b_spac': 25e-9, 'sigmax': sigmax,
               'sigmay': sigmay, 'sigmat': sigmat, 'beam_gamma': beam_gamma,
               'bunch_intensity': 1.1e11, 'bunch_macro_particles': 10**5,
               't_offs': t_offs}

saving_inputs = {'flag_checkpointing': True,
                 'checkpoints': np.linspace(1, n_bunches, n_bunches),
                 'temps_filename': 'complete_temp.h5', 'flag_output': True,
                 'images_dir': 'images_kick', 'custom_plot': plot_kick,
                 'stride_imgs': 10, 'field_probes': field_probes,
                 'field_probes_dump_stride': 100}

antenna_inputs = {'laser_func': laser_func, 'laser_source_z': laser_source_z,
                  'laser_polangle': laser_polangle, 'laser_emax': laser_emax,
                  'laser_xmin': laser_xmin, 'laser_xmax': laser_xmax,
                  'laser_ymin': laser_ymin, 'laser_ymax': laser_ymax}

simulation_inputs = {'enable_trap': enable_trap, 'chamber': chamber,
                     'tot_nsteps': 3500}


sim = warp_pyecloud_sim(fieldsolver_inputs = fieldsolver_inputs, 
                        beam_inputs = beam_inputs, 
                        ecloud_inputs = ecloud_inputs, 
                        antenna_inputs = antenna_inputs, 
                        saving_inputs = saving_inputs,
                        simulation_inputs = simulation_inputs)

sim.all_steps_no_ecloud()

fieldsolver_inputs = beam_inputs = ecloud_inputs = antenna_inputs = None
saving_inputs = simulations_inputs = None

sim.dump('cavity.%d.dump' %picmi.warp.me)


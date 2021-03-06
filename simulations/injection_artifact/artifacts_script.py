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

enable_trap = True

nz = 100
N_mp_max_slice = 60000
init_num_elecs_slice = 0 #2*10**5
dh = 3.e-4
width = 2*23e-3
height = 2*18e-3
z_length = 1.
z_start = -z_length/2
z_end = z_length/2
By = 0.53549999999999998

chamber = RectChamber(width, height, z_start, z_end)
lattice_elem = Dipole(z_start, z_end, By)

nx = int(np.ceil(width/dh))
ny = int(np.ceil(height/dh))

# Compute sigmas
nemittx = 2.5e-6
nemitty = nemittx
beta_x = 85
beta_y = 90

beam_gamma = 479.
beam_beta = np.sqrt(1-1/(beam_gamma**2))
sigmax = np.sqrt(beta_x*nemittx/(beam_gamma*beam_beta))
sigmay = np.sqrt(beta_y*nemitty/(beam_gamma*beam_beta))
n_bunches = 50
print(sigmax)

def dipole_plots(self, l_force=0):
    fontsz = 16
    plt.rcParams['axes.labelsize'] = fontsz
    plt.rcParams['axes.titlesize'] = fontsz
    plt.rcParams['xtick.labelsize'] = fontsz
    plt.rcParams['ytick.labelsize'] = fontsz
    plt.rcParams['legend.fontsize'] = fontsz
    plt.rcParams['legend.title_fontsize'] = fontsz

    chamber = self.chamber
    if l_force or self.n_step%self.stride_imgs == 0:
        plt.close()
        (Nx, Ny, Nz) = np.shape(self.ecloud.wspecies.get_density())
        fig, axs = plt.subplots(1, 2, figsize = (13.5, 5))
        fig.subplots_adjust(left = 0.1, bottom = 0.07, right = 0.99,
                            top = 0.87)
        d = (self.ecloud.wspecies.get_density()
           + self.beam.wspecies.get_density())
        d2  = (self.ecloud.wspecies.get_density())
        im1 = axs[0].imshow(d[:, :, int(Nz/2)] .T, cmap = 'jet',
              origin = 'lower', vmin = 0,
              vmax = 1e13,
              extent = [chamber.xmin, chamber.xmax ,
                        chamber.ymin, chamber.ymax])
        axs[0].set_xlabel('x [m]')
        axs[0].set_ylabel('y [m]')
        axs[0].set_title('rho')
        fig.colorbar(im1, ax = axs[0])
        im2 = axs[1].imshow(d[int(Nx/2), :, :], cmap = 'jet',
                            origin = 'lower',
                            vmin = 0,
                            vmax = 1e13,
                            extent=[chamber.zmin, chamber.zmax,
                                    chamber.ymin, chamber.ymax])

        axs[1].set_aspect((chamber.zmax-chamber.zmin)/(chamber.xmax-chamber.xmin))
        axs[1].set_xlabel('z [m]')
        axs[1].set_ylabel('y [m]')
        axs[1].set_title('rho')
        fig.suptitle('t = %1.6e' %picmi.warp.top.time, fontsize=fontsz)
        fig.colorbar(im2, ax = axs[1])

        figname = self.images_dir + '/' + repr(int(self.n_step)).zfill(4) + '.png'
        plt.savefig(figname)

def plot_field(self, l_force=0):
    chamber = self.chamber
    em = self.solver.solver
    fields = em.fields
    if picmi.warp.it%10 == 0:
        Ex = ff.gatherex()
        Ey = ff.gatherey()
        Ez = ff.gatherez()
        ff = np.sqrt(np.square(Ex) + np.square(Ey) + np.square(Ez))
        ffstr = 'E_norm'
        mine = np.min(ff)
        maxe = np.max(ff)
        images_dir = 'imgs'
        plot_fields(ff, ffstr, mine, maxe, chamber, 'imgs')


fieldsolver_inputs = {'solver_type':'EM', 'nx': nx, 'ny': ny, 'nz': nz,
                      'cfl': 1.}

beam_inputs = {'n_bunches': n_bunches, 'b_spac' : 25e-9,
               'beam_gamma': beam_gamma, 'sigmax': sigmax, 'sigmay': sigmay, 
               'sigmat': 1.000000e-09/4., 'bunch_macro_particles': 1e7,
               't_offs': 2.5e-9, 'bunch_intensity': 1.1e11}

ecloud_inputs = {'init_num_elecs': init_num_elecs_slice*nz,
              'init_num_elecs_mp': int(0.7*N_mp_max_slice*nz),    
              'pyecloud_nel_mp_ref': init_num_elecs_slice/(0.7*N_mp_max_slice),
              'pyecloud_fact_clean': 1e-6, 'pyecloud_fact_split': 1.5,
              'Emax': 332., 'del_max': 1.7, 'R0': 0.7, 'E_th': 35,
              'sigmafit': 1.0828, 'mufit': 1.6636,
              'secondary_angle_distribution': 'cosine_3D', 
              'N_mp_max': N_mp_max_slice*nz,'N_mp_target': N_mp_max_slice/3*nz}

saving_inputs = {'images_dir': 'images_rect_dipole',
                 'custom_plot': plot_fields, 'stride_imgs': 10000,
                 'output_filename': 'rect_dipole_out.h5'}

simulation_inputs = {'enable_trap': enable_trap,
                     'flag_relativ_tracking': True,
                     'lattice_elem': lattice_elem, 'chamber': chamber}

sim = warp_pyecloud_sim(fieldsolver_inputs = fieldsolver_inputs, 
                        beam_inputs = beam_inputs, 
                        ecloud_inputs = ecloud_inputs,
                        saving_inputs = saving_inputs,
                        simulation_inputs = simulation_inputs)

sim.all_steps()

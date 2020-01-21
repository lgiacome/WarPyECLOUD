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
from chamber import CrabCavity
from lattice_elements import CrabFields
import matplotlib.pyplot as plt

enable_trap = True

N_mp_max = 6000000
init_num_elecs = 2*10**7
dh = 3.e-4

nx = 200/4 
ny = 200/4 
nz = 100/4

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

# Paths for the fields
fields_folder = str(Path(os.getcwd()).parent.parent)
efield_path = fields_folder + '/efield.txt'
hfield_path = fields_folder + '/hfield.txt'

chamber = CrabCavity(-max_z, max_z)
E_field_max = 5e6 #57
#lattice_elem = CrabFields(max_z, max_rescale = E_field_max, efield_path = efield_path, 
#                          hfield_path = hfield_path)
lattice_elem = None
n_bunches = 1 

def plots_crab(self, l_force = 0):
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
        (Nx, Ny, Nz) = np.shape(self.secelec.wspecies.get_density())
        fig, axs = plt.subplots(1, 2, figsize = (12, 5))
        fig.subplots_adjust(left = 0.05, bottom = 0.1, right = 0.97, 
                            top = 0.94, wspace = 0.15)
        d = (self.secelec.wspecies.get_density()
           + self.elecb.wspecies.get_density()
           + self.beam.wspecies.get_density())
        d2  = (self.secelec.wspecies.get_density() 
            + self.elecb.wspecies.get_density())
        im1 = axs[0].imshow(d[:, :, int(Nz/2)] .T, cmap = 'jet', 
              origin = 'lower', vmin = 0.2*np.min(d2[:, :, int(Nz/2)]), 
              vmax = 0.8*np.max(d2[:, :, int(Nz/2)]), 
              extent = [chamber.xmin, chamber.xmax , 
                        chamber.ymin, chamber.ymax])

        lw = 2
        axc = axs[0]
        axc.vlines(x = -chamber.l_main_x/2, ymin = -chamber.l_main_y/2,
                             ymax = chamber.l_main_y/2, color = 'red', lw = lw)
        axc.vlines(x = chamber.l_main_x/2, ymin = -chamber.l_main_y/2,
                             ymax = chamber.l_main_y/2, color = 'red', lw = lw)
        axc.vlines(x = chamber.l_main_int_x, ymax = chamber.l_main_y/2,
                                   ymin = chamber.ycen_up-chamber.l_main_int_y, 
                                                        color = 'red', lw = lw)
        axc.vlines(x = chamber.l_main_int_x, ymin = -chamber.l_main_y/2,
                                 ymax = chamber.ycen_down+chamber.l_main_int_y, 
                                                        color = 'red', lw = lw)
        axc.vlines(x = -chamber.l_main_int_x, ymax = chamber.l_main_y/2,
                                   ymin = chamber.ycen_up-chamber.l_main_int_y, 
                                                        color = 'red', lw = lw)
        axc.vlines(x = -chamber.l_main_int_x, ymin = -chamber.l_main_y/2,
                                 ymax = chamber.ycen_down+chamber.l_main_int_y, 
                                                        color = 'red', lw = lw)
        axc.hlines(y = chamber.l_main_y/2, xmin = -chamber.l_main_x/2,
                          xmax = -chamber.l_main_int_x, color = 'red', lw = lw)
        axc.hlines(y = chamber.l_main_y/2, xmax = chamber.l_main_x/2,
                           xmin = chamber.l_main_int_x, color = 'red', lw = lw)
        axc.hlines(y = -chamber.l_main_y/2, xmin = -chamber.l_main_x/2,
                          xmax = -chamber.l_main_int_x, color = 'red', lw = lw)
        axc.hlines(y = -chamber.l_main_y/2, xmax = chamber.l_main_x/2,
                           xmin = chamber.l_main_int_x, color = 'red', lw = lw)
        axc.hlines(y = chamber.ycen_up - chamber.l_main_int_y, 
                   xmin = -chamber.l_main_int_x, xmax = chamber.l_main_int_x, 
                                                        color = 'red', lw = lw)
        axc.hlines(y = chamber.ycen_down + chamber.l_main_int_y, 
                   xmin = -chamber.l_main_int_x, xmax = chamber.l_main_int_x, 
                                                        color = 'red', lw = lw)
 

        axs[0].set_xlabel('x [m]')
        axs[0].set_ylabel('y [m]')
        axs[0].set_title('e- density')
        fig.colorbar(im1, ax = axs[0])
        im2 = axs[1].imshow(d[int(Nx/2), :, :], cmap = 'jet', 
                            origin = 'lower', 
                            vmin = 0.2*np.min(d2[int(Nx/2), :, :]), 
                            vmax = 0.8*np.max(d2[int(Nx/2), :, :]),
                            extent=[chamber.zmin, chamber.zmax, 
                                    chamber.ymin, chamber.ymax], 
                            aspect = 'equal')
        lw = 2
        axc = axs[1]
        axc.hlines(y = chamber.l_beam_pipe/2, xmin = chamber.zmin, 
                            xmax = -chamber.l_main_z/2, color = 'red', lw = lw)
        axc.hlines(y = -chamber.l_beam_pipe/2, xmin = chamber.zmin, 
                            xmax = -chamber.l_main_z/2, color = 'red', lw = lw)
        axc.hlines(y = chamber.l_beam_pipe/2, xmax = chamber.zmax, 
                             xmin = chamber.l_main_z/2, color = 'red', lw = lw)
        axc.hlines(y = -chamber.l_beam_pipe/2, xmax = chamber.zmax, 
                             xmin = chamber.l_main_z/2, color = 'red', lw = lw)
        axc.hlines(y = chamber.ycen_up-chamber.l_main_int_y, 
                   xmin = -chamber.l_main_int_z, xmax = chamber.l_main_int_z,
                   color = 'red', lw = lw)
        axc.hlines(y = chamber.ycen_down+chamber.l_main_int_y, 
                   xmin = -chamber.l_main_int_z, xmax = chamber.l_main_int_z, 
                   color = 'red', lw = lw)
        axc.hlines(y = chamber.l_main_y/2, xmin = -chamber.l_main_z/2, 
                          xmax = -chamber.l_main_int_z, color = 'red', lw = lw)
        axc.hlines(y = -chamber.l_main_y/2, xmin = -chamber.l_main_z/2, 
                          xmax = -chamber.l_main_int_z, color = 'red', lw = lw)
        axc.hlines(y = chamber.l_main_y/2, xmax = chamber.l_main_z/2, 
                           xmin = chamber.l_main_int_z, color = 'red', lw = lw)
        axc.hlines(y = -chamber.l_main_y/2, xmax = chamber.l_main_int_z, 
                           xmin = chamber.l_main_z/2, color = 'red', lw = lw)
      
        axc.vlines(x = -chamber.l_main_z/2, ymin = chamber.l_beam_pipe/2,
                             ymax = chamber.l_main_y/2, color = 'red', lw = lw)
        axc.vlines(x = -chamber.l_main_z/2, ymax = -chamber.l_beam_pipe/2,
                            ymin = -chamber.l_main_y/2, color = 'red', lw = lw)
        axc.vlines(x = -chamber.l_main_int_z, ymin = chamber.l_beam_pipe/2,
                             ymax = chamber.l_main_y/2, color = 'red', lw = lw)
        axc.vlines(x = -chamber.l_main_int_z, ymax = -chamber.l_beam_pipe/2,
                            ymin = -chamber.l_main_y/2, color = 'red', lw = lw)
        axc.vlines(x = chamber.l_main_int_z, ymax = chamber.l_main_y/2,
                   ymin = chamber.ycen_up - chamber.l_main_int_y, 
                                                        color = 'red', lw = lw)
        axc.vlines(x = chamber.l_main_int_z, ymin = -chamber.l_main_y/2,
                   ymax = chamber.ycen_down + chamber.l_main_int_y, 
                                                        color = 'red', lw = lw)
        axc.vlines(x = chamber.l_main_z/2, ymin = chamber.l_beam_pipe/2,
                             ymax = chamber.l_main_y/2, color = 'red', lw = lw)
        axc.vlines(x = chamber.l_main_z/2, ymax = -chamber.l_beam_pipe/2,
                            ymin = -chamber.l_main_y/2, color = 'red', lw = lw)
        axc.set_aspect((chamber.zmax-chamber.zmin)/(chamber.xmax-chamber.xmin))

        axs[1].set_xlabel('z [m]')
        axs[1].set_ylabel('y [m]')
        axs[1].set_title('e- density')
        fig.colorbar(im2, ax = axs[1])

        figname = self.images_dir + '/%d.png' %int(self.n_step)
        plt.savefig(figname)



kwargs = {'enable_trap': enable_trap,
	'z_length': 1.,
	'nx': nx,
	'ny': ny, 
	'nz': nz,
	'n_bunches': n_bunches,
    'b_spac' : 25e-9,
    'beam_gamma': beam_gamma, 
	'sigmax': sigmax,
    'sigmay': sigmay, 
    'sigmat': sigmat,
    'bunch_intensity': 1.1e11, 
    'init_num_elecs': init_num_elecs,
    'init_num_elecs_mp': int(0.7*N_mp_max), 
    'pyecloud_nel_mp_ref': init_num_elecs/(0.7*N_mp_max),
	'dt': 25e-12,
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
	'flag_checkpointing': True,
	'checkpoints': np.linspace(1, n_bunches, n_bunches),
    'temps_filename': 'complete_temp.mat',
    'flag_output': True,
    'bunch_macro_particles': 1e5,
    't_offs': 3*sigmat+1e-10,
    'output_filename': 'complete_out.mat',
    'images_dir': 'images',
    'flag_relativ_tracking': True,
    'lattice_elem': lattice_elem,
    'chamber': chamber,
    'custom_plot': plots_crab,
    'stride_imgs': 1
}

sim = warp_pyecloud_sim(**kwargs)

sim.all_steps()

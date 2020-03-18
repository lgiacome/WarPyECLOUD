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
from lattice_elements import CrabFields
import matplotlib.pyplot as plt

enable_trap = True

N_mp_max = 6000000
init_num_elecs = 2*10**7
dh = 3.e-4

nx = 20/2 
ny = 15/2
nz = 20/2

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
file_em_fields = fields_folder + '/ey_cavity.mat'
em_scale_fac = 1e6

def plots_crab(self, l_force = 0):
    fontsz = 16
    plt.rcParams['axes.labelsize'] = fontsz
    plt.rcParams['axes.titlesize'] = fontsz
    plt.rcParams['xtick.labelsize'] = fontsz
    plt.rcParams['ytick.labelsize'] = fontsz
    plt.rcParams['legend.fontsize'] = fontsz
    plt.rcParams['legend.title_fontsize'] = fontsz

    chamber = self.chamber
    k_antenna = int((self.source_z - zmin)/em.dz)
    j_mid_waveguide = int((chamber.ycen6 - ymin)/em.dy)
    flist = ['Ex','Ey','Ez','Bx','By','Bz']
    flist = ['Ey']
    if pw.top.it%10==0 or l_force:
        for ffstr in flist:
            if ffstr == 'Ex': ff = em.gatherex()
            if ffstr == 'Ey': ff = em.gatherey()
            if ffstr == 'Ez': ff = em.gatherez()
            if ffstr == 'Bx': ff = em.gatherbx()
            if ffstr == 'By': ff = em.gatherby()
            if ffstr == 'Bz': ff = em.gatherbz()
            if me==0:
                maxe =35 #np.max(ey[:,:,:])
                mine = -35 #np.min(ey[:,:,:])
                cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
                cmap = mpl.cm.jet
                Nx = nx + 1
                Ny = ny + 1
                ey00 = ff[:,:,chamber.k_antenna]
                ey01 = ff[:,chamber.j_mid_waveguide,:]
                ey10 = ff[:,int(Ny/2),:]
                ey11 = ff[int(Nx/2),:,:]
                ax = plt.subplot2grid((2,2), (0,0))
                ax.imshow(np.flipud(ey00.T),cmap = cmap, vmin=mine,vmax = maxe, extent=[xmin, xmax, ymin, ymax], aspect = 'auto')
                lw = 1
                plt.vlines(x = chamber.l_beam_pipe/2, ymin = -chamber.l_beam_pipe/2, ymax = chamber.l_beam_pipe/2, color='black',lw=lw)
                plt.vlines(x = -chamber.l_beam_pipe/2, ymin = -chamber.l_beam_pipe/2, ymax = chamber.l_beam_pipe/2, color='black',lw=lw)
                plt.hlines(y = chamber.l_beam_pipe/2, xmin = -chamber.l_beam_pipe/2, xmax = chamber.l_beam_pipe/2, color='black',lw=lw)
                plt.hlines(y = -chamber.l_beam_pipe/2, xmin = -chamber.l_beam_pipe/2, xmax = chamber.l_beam_pipe/2, color='black',lw=lw)
                plt.vlines(x = chamber.x_min_wg, ymin = chamber.y_min_wg, ymax = chamber.y_max_wg, color='black',lw=lw)
                plt.vlines(x = chamber.x_max_wg, ymin = chamber.y_min_wg, ymax = chamber.y_max_wg, color='black',lw=lw)
                plt.hlines(y = chamber.y_min_wg, xmin = chamber.x_min_wg, xmax = chamber.x_max_wg, color='black',lw=lw)
                plt.hlines(y = chamber.y_max_wg, xmin = chamber.x_min_wg, xmax = chamber.x_max_wg, color='black',lw=lw)
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                ax = plt.subplot2grid((2,2), (0,1))
                ax.imshow(np.flipud(ey01.T), cmap = cmap, vmin=mine,vmax = maxe, extent=[xmin, xmax, zmin, zmax], aspect = 'auto')
                plt.vlines(x = chamber.x_min_wg, ymin = z_min_wg, ymax = chamber.z_rest, color='black',lw=lw)
                plt.vlines(x = chamber.x_max_wg, ymin = z_min_wg, ymax = chamber.z_rest, color='black',lw=lw)
                plt.vlines(x = chamber.x_min_wg_rest, ymin = chamber.z_rest, ymax = -chamber.l_main_z/2, color='black',lw=lw)
                plt.vlines(x = chamber.x_max_wg_rest, ymin = chamber.z_rest, ymax = -chamber.l_main_z/2, color='black',lw=lw)
                plt.hlines(y = chamber.z_rest, xmin = chamber.x_min_wg, xmax = chamber.x_min_wg_rest, color='black',lw=lw)
                plt.hlines(y = chamber.z_rest, xmin = chamber.x_max_wg_rest, xmax = chamber.x_max_wg, color='black',lw=lw)
                plt.vlines(x = -chamber.l_main_x/2, ymin = -chamber.l_main_z/2, ymax = chamber.l_main_z/2, color='black',lw=lw)
                plt.vlines(x = chamber.l_main_x/2, ymin = -chamber.l_main_z/2, ymax = chamber.l_main_z/2, color='black',lw=lw)
                plt.hlines(y = -chamber.l_main_z/2, xmin = -chamber.l_main_x/2, xmax = chamber.x_min_wg_rest, color='black',lw=lw)
                plt.hlines(y = -chamber.l_main_z/2, xmin = chamber.x_max_wg_rest, xmax = chamber.l_main_x/2, color='black',lw=lw)
                plt.hlines(y = chamber.l_main_z/2, xmin = -chamber.l_main_x/2, xmax = chamber.l_main_x/2, color='black',lw=lw)
                plt.hlines(y = chamber.l_main_int_z, xmin = -chamber.l_main_int_x, xmax = chamber.l_main_int_x, color='black',lw=lw)
                plt.hlines(y = -chamber.l_main_int_z, xmin = -chamber.l_main_int_x, xmax = chamber.l_main_int_x, color='black',lw=lw)
                plt.vlines(x = -chamber.l_main_int_x, ymin = -chamber.l_main_int_z, ymax = chamber.l_main_int_z, color='black',lw=lw)
                plt.vlines(x = chamber.l_main_int_x, ymin = -chamber.l_main_int_z, ymax = chamber.l_main_int_z, color='black',lw=lw)           
                ax.set_xlabel('x')
                ax.set_ylabel('z')
                ax = plt.subplot2grid((2,2), (1,0))
                im = ax.imshow(np.flipud(ey10.T), cmap = cmap, vmin=mine,vmax = maxe, extent=[xmin, xmax, zmin, zmax], aspect = 'auto')
                plt.vlines(x = chamber.l_beam_pipe/2, ymin = chamber.l_main_z/2, ymax = zmax)
                plt.vlines(x = -chamber.l_beam_pipe/2, ymin = chamber.l_main_z/2, ymax = zmax)
                plt.vlines(x = chamber.l_beam_pipe/2, ymin = zmin, ymax = -chamber.l_main_z/2)
                plt.vlines(x = -chamber.l_beam_pipe/2, ymin = zmin, ymax = -chamber.l_main_z/2)
                plt.vlines(x = chamber.l_main_x/2, ymin = -chamber.l_main_z/2, ymax = chamber.l_main_z/2)
                plt.vlines(x = -chamber.l_main_x/2, ymin = -chamber.l_main_z/2, ymax = chamber.l_main_z/2)           
                plt.hlines(y = chamber.l_main_z/2, xmin = chamber.l_beam_pipe/2, xmax = chamber.l_main_x/2)
                plt.hlines(y = -chamber.l_main_z/2, xmin = chamber.l_beam_pipe/2, xmax = chamber.l_main_x/2)
                plt.hlines(y = chamber.l_main_z/2, xmin = -chamber.l_main_x/2, xmax = -chamber.l_beam_pipe/2)
                plt.hlines(y = -chamber.l_main_z/2, xmin = -chamber.l_main_x/2, xmax = -chamber.l_beam_pipe/2)
                plt.hlines(y = chamber.l_main_int_z, xmin = -chamber.l_main_int_x, xmax = chamber.l_main_int_x, color='black',lw=lw, linestyle = 'dashed')
                plt.hlines(y = -chamber.l_main_int_z, xmin = -chamber.l_main_int_x, xmax = chamber.l_main_int_x, color='black',lw=lw, linestyle = 'dashed')
                plt.vlines(x = -chamber.l_main_int_x, ymin = -chamber.l_main_int_z, ymax = chamber.l_main_int_z, color='black',lw=lw, linestyle = 'dashed')
                plt.vlines(x = chamber.l_main_int_x, ymin = -chamber.l_main_int_z, ymax = chamber.l_main_int_z, color='black',lw=lw, linestyle = 'dashed')
                ax.set_xlabel('x')
                ax.set_ylabel('z')
                ax = plt.subplot2grid((2,2), (1,1))
                im = ax.imshow(np.flipud(ey11), cmap = cmap, vmin=mine,vmax = maxe, extent=[zmin, zmax, ymin, ymax], aspect = 'auto')
                plt.hlines(y = -chamber.l_beam_pipe/2, xmin = zmin, xmax = -chamber.l_main_z/2)
                plt.hlines(y = chamber.l_beam_pipe/2, xmin = zmin, xmax = -chamber.l_main_z/2)
                plt.hlines(y = -chamber.l_beam_pipe/2, xmin = chamber.l_main_z/2, xmax = zmax)
                plt.hlines(y = chamber.l_beam_pipe/2, xmin = chamber.l_main_z/2, xmax = zmax)
                plt.hlines(y = chamber.y_min_wg, xmin = zmin, xmax = -chamber.l_main_z/2)
                plt.hlines(y = chamber.y_max_wg, xmin = zmin, xmax = -chamber.l_main_z/2)
                plt.hlines(y = -chamber.l_main_y/2, xmin = -chamber.l_main_z/2, xmax = -chamber.l_main_int_z)
                plt.hlines(y = chamber.l_main_y/2, xmin = -chamber.l_main_z/2, xmax = -chamber.l_main_int_z)
                plt.hlines(y = -chamber.l_main_y/2, xmin = chamber.l_main_int_z, xmax = chamber.l_main_z/2)
                plt.hlines(y = chamber.l_main_y/2, xmin = chamber.l_main_int_z, xmax = chamber.l_main_z/2)
                plt.hlines(y = chamber.l_beam_pipe/2, xmin = -chamber.l_main_int_z, xmax = chamber.l_main_int_z)
                plt.hlines(y = -chamber.l_beam_pipe/2, xmin = -chamber.l_main_int_z, xmax = chamber.l_main_int_z)
                plt.vlines(x = -chamber.l_main_z/2, ymin = chamber.l_beam_pipe/2, ymax = chamber.y_min_wg)
                plt.vlines(x = -chamber.l_main_z/2, ymin = chamber.y_max_wg, ymax = chamber.l_main_y/2)
                plt.vlines(x = -chamber.l_main_z/2, ymin = -chamber.l_main_y/2, ymax = -chamber.l_beam_pipe/2)
                plt.vlines(x = chamber.l_main_z/2, ymin = -chamber.l_main_y/2, ymax = -chamber.l_beam_pipe/2)
                plt.vlines(x = chamber.l_main_z/2, ymin = chamber.l_beam_pipe/2, ymax = chamber.l_main_y/2)
                plt.vlines(x = chamber.l_main_int_z, ymin = -chamber.l_main_y/2, ymax = -chamber.l_beam_pipe/2)
                plt.vlines(x = chamber.l_main_int_z, ymin = chamber.l_beam_pipe/2, ymax = chamber.l_main_y/2)
                plt.vlines(x = -chamber.l_main_int_z, ymin = -chamber.l_main_y/2, ymax = -chamber.l_beam_pipe/2)
                plt.vlines(x = -chamber.l_main_int_z, ymin = chamber.l_beam_pipe/2, ymax = chamber.l_main_y/2)
                ax.set_xlabel('z')
                ax.set_xlabel('z')
                ax.set_ylabel('y')
                #plt.subplot(2,2,4)
                #laser_func_mat = laser_func_plot(xx,yy,pw.top.time)
                #ax = plt.gca()
                #im = ax.imshow(laser_func_mat, cmap = cmap, vmin=mine,vmax = maxe, extent=[ -width/2, width/2,-height/2, height/2])
                #ax.set_xlabel('x')
                #ax.set_ylabel('y')
                fig.subplots_adjust(left = 0.15, right=0.8, hspace = 0.4, wspace = 0.4)
                fig.colorbar(im, cax=cbar_ax)
                fig.suptitle('Ey, t = %1.6e' %pw.top.time )
                #fig.tight_layout()
                if not os.path.exists('images_cavity'):
                    os.mkdir('images_cavity')
                filename = 'images_cavity/it_' + str(pw.top.it).zfill(5) + '.png'
                plt.savefig(filename, dpi=150)
                fig.clf()



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
    'flag_checkpointing': True,
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
    'em_scale_fac': em_scale_fac
}

sim = warp_pyecloud_sim(**kwargs)

sim.all_steps()

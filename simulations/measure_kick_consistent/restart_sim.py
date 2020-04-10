import sys
import os
from pathlib import Path
import matplotlib as mpl
#mpl.use('agg')
import matplotlib.pyplot as plt
from warp import restart, me, picmi

BIN = os.path.expanduser("../../")
if BIN not in sys.path:
    sys.path.append(BIN)

import numpy as np
#from warp_pyecloud_sim import warp_pyecloud_sim
#from chamber import CrabCavityWaveguide
#from plots import plot_field_crab

base_folder = str(Path(os.getcwd()).parent.parent)
cwd = str(Path(os.getcwd()))
folder = base_folder + '/dumps'
dumpfile = folder+ '/cavity.%d.dump' %me 


def my_gaussian_time_prof(self):
    t = picmi.warp.top.time
    val = 0
    for i in range(0,self.n_bunches):
        val += (self.bunch_macro_particles*1.
               /np.sqrt(2*np.pi*self.sigmat**2)
               *np.exp(-(t-i*self.b_spac-self.t_offs)**2
               /(2*self.sigmat**2))*picmi.warp.top.dt)
    sim.bunch_profile[picmi.warp.top.it] = val
    return val

def plot_bunch(self, l_force=0):
    fontsz = 16
    plt.rcParams['axes.labelsize'] = fontsz
    plt.rcParams['axes.titlesize'] = fontsz
    plt.rcParams['xtick.labelsize'] = fontsz
    plt.rcParams['ytick.labelsize'] = fontsz
    plt.rcParams['legend.fontsize'] = fontsz
    plt.rcParams['legend.title_fontsize'] = fontsz

    chamber = self.chamber
    if l_force or picmi.warp.top.it%10 == 0: #self.stride_imgs == 0:
        (Nx, Ny, Nz) = np.shape(self.ecloud.wspecies.get_density())
        fig, axs = plt.subplots(1, 2, figsize = (13.5, 5))
        fig.subplots_adjust(left = 0.1, bottom = 0.07, right = 0.99,
                            top = 0.87)
        d = (self.beam.wspecies.get_density())
        im1 = axs[0].imshow(d[:, :, int(Nz/2)] .T, cmap = 'jet',
              origin = 'lower', vmin = 0,
#              vmax = 1e13,
              extent = [chamber.xmin, chamber.xmax ,
                        chamber.ymin, chamber.ymax])
        axs[0].set_xlabel('x [m]')
        axs[0].set_ylabel('y [m]')
        axs[0].set_title('rho')
        fig.colorbar(im1, ax = axs[0])
        im2 = axs[1].imshow(d[int(Nx/2), :, :], cmap = 'jet',
                            origin = 'lower',
                            vmin = 0,
#                            vmax = 1e13,
                            extent=[chamber.zmin, chamber.zmax,
                                    chamber.ymin, chamber.ymax])

        axs[1].set_aspect((chamber.zmax-chamber.zmin)/(chamber.xmax-chamber.xmin))
        axs[1].set_xlabel('z [m]')
        axs[1].set_ylabel('y [m]')
        axs[1].set_title('rho')
        fig.suptitle('t = %1.6e' %picmi.warp.top.time, fontsize=fontsz)
        fig.colorbar(im2, ax = axs[1])

        figname = 'images_bunch'+ '/' + repr(int(picmi.warp.top.it)).zfill(4) + '.png'
        plt.savefig(figname)
        plt.close('all')

def noplot(self, l_force = 0):
    pass

restart(dumpfile)
sim.bunch_profile = np.zeros(10000)
sim.t_offs = 5.1355E-08
sim.n_bunches = 1
sim.bunch_macro_particles = 1e4
sim.reinit(laser_func, noplot ,custom_time_prof = my_gaussian_time_prof)
sim.tot_nsteps = 400
sim.saver.extend_probe_vectors(sim.tot_nsteps)
#sim.all_steps_no_ecloud()


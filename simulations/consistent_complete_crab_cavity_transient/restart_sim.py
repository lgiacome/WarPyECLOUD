import faulthandler; faulthandler.enable()
import sys
import os
from pathlib import Path
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
from warp import restart, me, picmi

BIN = os.path.expanduser("../../")
if BIN not in sys.path:
    sys.path.append(BIN)
from dump_restart import reinit

import warp_pyecloud_sim
import numpy as np

def plot_dens(self, l_force=0):
    fontsz = 16
    plt.rcParams['axes.labelsize'] = fontsz
    plt.rcParams['axes.titlesize'] = fontsz
    plt.rcParams['xtick.labelsize'] = fontsz
    plt.rcParams['ytick.labelsize'] = fontsz
    plt.rcParams['legend.fontsize'] = fontsz
    plt.rcParams['legend.title_fontsize'] = fontsz

    chamber = self.chamber
    if l_force or picmi.warp.top.it%1 == 0: #self.stride_imgs == 0:
        (Nx, Ny, Nz) = np.shape(self.ecloud.wspecies.get_density())
        fig, axs = plt.subplots(1, 2, figsize = (13.5, 5))
        fig.subplots_adjust(left = 0.1, bottom = 0.07, right = 0.99,
                            top = 0.87)
        d = (self.beam.wspecies.get_density()+self.ecloud.wspecies.get_density())
        im1 = axs[0].imshow(d[:, :, int(Nz/2)] .T, cmap = 'jet',
              origin = 'lower', vmin = 0,
              vmax = 1e10,
              extent = [chamber.xmin, chamber.xmax ,
                        chamber.ymin, chamber.ymax])
        axs[0].set_xlabel('x [m]')
        axs[0].set_ylabel('y [m]')
        axs[0].set_title('rho')
        fig.colorbar(im1, ax = axs[0])
        im2 = axs[1].imshow(d[int(Nx/2), :, :], cmap = 'jet',
                            origin = 'lower',
                            vmin = 0,
                            vmax = 1e10,
                            extent=[chamber.zmin, chamber.zmax,
                                    chamber.ymin, chamber.ymax])

        axs[1].set_aspect((chamber.zmax-chamber.zmin)/(chamber.xmax-chamber.xmin))
        axs[1].set_xlabel('z [m]')
        axs[1].set_ylabel('y [m]')
        axs[1].set_title('rho')
        fig.suptitle('t = %1.6e' %picmi.warp.top.time, fontsize=fontsz)
        fig.colorbar(im2, ax = axs[1])

        figname = 'images_dens'+ '/' + repr(int(picmi.warp.top.it)).zfill(4) + '.png'
        plt.savefig(figname)
        del fig
        plt.close('all')
        #print(np.max(self.solver.solver.fields.Ey))

def plots_crab(self, l_force = 0):
    #fontsz = 16
    #plt.rcParams['axes.labelsize'] = fontsz
    #plt.rcParams['axes.titlesize'] = fontsz
    #plt.rcParams['xtick.labelsize'] = fontsz
    #plt.rcParams['ytick.labelsize'] = fontsz
    #plt.rcParams['legend.fontsize'] = fontsz
    #plt.rcParams['legend.title_fontsize'] = fontsz

    chamber = self.chamber
    if self.laser_source_z is None: here_laser_source_z = chamber.zmin + 0.5*(-chamber.l_main_z/2-chamber.zmin)
    else: here_laser_source_z = self.laser_source_z
    em = self.solver.solver
    k_antenna = int((here_laser_source_z - chamber.zmin)/em.dz)
    j_mid_waveguide = int((chamber.ycen6 - chamber.ymin)/em.dy)

    flist = ['Ex','Ey','Ez','Bx','By','Bz']
    flist = ['Jy']
    pw = picmi.warp
    if pw.top.it%10==0 or l_force:
        #fig = plt.figure( figsize=(7,7))
        for ffstr in flist:
            if ffstr == 'Ex': ff = em.gatherex()
            if ffstr == 'Ey':
                ff = em.gatherey()
                maxe =35e6*self.em_scale_fac #np.max(ey[:,:,:])
                mine = -35e6*self.em_scale_fac #np.min(ey[:,:,:])
            if ffstr == 'Ez': ff = em.gatherez()
            if ffstr == 'Bx': ff = em.gatherbx()
            if ffstr == 'By': ff = em.gatherby()
            if ffstr == 'Bz': ff = em.gatherbz()
            if ffstr == 'elecs':
                ff = self.ecloud.wspecies.get_density()
                maxe = 5e9 #np.max(ey[:,:,:])
                mine = 0 #np.min(ey[:,:,:])
            if ffstr == 'Jy': 
                ff = em.gatherjy()
                maxe = np.max(ff[:,:,:])
                mine = np.min(ff[:,:,:])
            if me==0:
                plot_field_crab(ff, ffstr, mine, maxe, k_antenna, j_mid_waveguide, chamber)


restart('cavity.0.dump')
reinit(sim, laser_func, plot_dens)
sim.tot_nsteps = 3600
sim.saver.tot_nsteps = 3600
sim.saver.extend_probe_vectors(sim.tot_nsteps)
sim.all_steps_no_ecloud()


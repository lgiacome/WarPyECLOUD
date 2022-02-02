import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from warp import picmi
import os

from cmcrameri import cm

def plot_field_crab(ff, ffstr, mine, maxe, k_antenna, j_mid_waveguide, chamber, images_dir = 'images_cavity', wg = True, lines = True):
    pw = picmi.warp
    fig = plt.figure( figsize=(7,7))
    xmin = chamber.xmin
    xmax = chamber.xmax
    ymin = chamber.ymin
    ymax = chamber.ymax
    zmin = chamber.zmin
    zmax = chamber.zmax
    cmap = mpl.cm.jet
    Nx, Ny, Nz = np.shape(ff)
    ey00 = ff[:,:,k_antenna]
    ey01 = ff[:,j_mid_waveguide,:]
    ey10 = ff[:,int(Ny/2),:]
    ey11 = ff[int(Nx/2),:,:]
    ax = plt.subplot2grid((2,2), (0,0))
    ax.imshow(np.flipud(ey00.T),cmap = cmap, vmin=mine,vmax = maxe, extent=[xmin, xmax, ymin, ymax], aspect = 'auto')
    lw = 1
    if lines:
        if wg:
            plt.vlines(x = chamber.l_beam_pipe/2, ymin = -chamber.l_beam_pipe/2, ymax = chamber.l_beam_pipe/2, color='black',lw=lw)
            plt.vlines(x = -chamber.l_beam_pipe/2, ymin = -chamber.l_beam_pipe/2, ymax = chamber.l_beam_pipe/2, color='black',lw=lw)
            plt.hlines(y = chamber.l_beam_pipe/2, xmin = -chamber.l_beam_pipe/2, xmax = chamber.l_beam_pipe/2, color='black',lw=lw)
            plt.hlines(y = -chamber.l_beam_pipe/2, xmin = -chamber.l_beam_pipe/2, xmax = chamber.l_beam_pipe/2, color='black',lw=lw)
            plt.vlines(x = chamber.x_min_wg, ymin = chamber.y_min_wg, ymax = chamber.y_max_wg, color='black',lw=lw)
            plt.vlines(x = chamber.x_max_wg, ymin = chamber.y_min_wg, ymax = chamber.y_max_wg, color='black',lw=lw)
            plt.hlines(y = chamber.y_min_wg, xmin = chamber.x_min_wg, xmax = chamber.x_max_wg, color='black',lw=lw)
            plt.hlines(y = chamber.y_max_wg, xmin = chamber.x_min_wg, xmax = chamber.x_max_wg, color='black',lw=lw)
        else:
            plt.hlines(y = -chamber.l_main_y/2, xmin = -chamber.l_main_x/2, xmax = -chamber.l_main_int_x)
            plt.hlines(y = chamber.l_main_y/2, xmin = -chamber.l_main_x/2, xmax = -chamber.l_main_int_x)
            plt.hlines(y = -chamber.l_main_y/2, xmin = chamber.l_main_int_x, xmax = chamber.l_main_x/2)
            plt.hlines(y = chamber.l_main_y/2, xmin = chamber.l_main_int_x, xmax = chamber.l_main_x/2)
            plt.hlines(y = -chamber.l_beam_pipe/2, xmin = -chamber.l_main_int_x, xmax = chamber.l_main_int_x)
            plt.hlines(y = chamber.l_beam_pipe/2, xmin = -chamber.l_main_int_x, xmax = chamber.l_main_int_x)
            plt.vlines(x = chamber.l_main_int_x, ymin = chamber.l_beam_pipe/2, ymax = chamber.l_main_y/2)
            plt.vlines(x = -chamber.l_main_int_x, ymin = chamber.l_beam_pipe/2, ymax = chamber.l_main_y/2)
            plt.vlines(x = chamber.l_main_int_x, ymin = -chamber.l_main_y/2, ymax = -chamber.l_beam_pipe/2)
            plt.vlines(x = -chamber.l_main_int_x, ymin = -chamber.l_main_y/2, ymax = -chamber.l_beam_pipe/2)
            plt.vlines(x = chamber.l_main_x/2, ymin = -chamber.l_main_y/2, ymax = chamber.l_main_y/2)
            plt.vlines(x = -chamber.l_main_x/2, ymin = -chamber.l_main_y/2, ymax = chamber.l_main_y/2)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax = plt.subplot2grid((2,2), (0,1))
    ax.imshow(np.flipud(ey01.T), cmap = cmap, vmin=mine,vmax = maxe, extent=[xmin, xmax, zmin, zmax], aspect = 'auto')
    if lines:
        if wg:
            plt.vlines(x = chamber.x_min_wg, ymin = chamber.zmin, ymax = chamber.z_rest, color='black',lw=lw)
            plt.vlines(x = chamber.x_max_wg, ymin = chamber.zmin, ymax = chamber.z_rest, color='black',lw=lw)
            plt.vlines(x = chamber.x_min_wg_rest, ymin = chamber.z_rest, ymax = -chamber.l_main_z/2, color='black',lw=lw)
            plt.vlines(x = chamber.x_max_wg_rest, ymin = chamber.z_rest, ymax = -chamber.l_main_z/2, color='black',lw=lw)
            plt.hlines(y = chamber.z_rest, xmin = chamber.x_min_wg, xmax = chamber.x_min_wg_rest, color='black',lw=lw)
            plt.hlines(y = chamber.z_rest, xmin = chamber.x_max_wg_rest, xmax = chamber.x_max_wg, color='black',lw=lw)
            plt.hlines(y = -chamber.l_main_z/2, xmin = -chamber.l_main_x/2, xmax = chamber.x_min_wg_rest, color='black',lw=lw)
            plt.hlines(y = -chamber.l_main_z/2, xmin = chamber.x_max_wg_rest, xmax = chamber.l_main_x/2, color='black',lw=lw)
        else:
            plt.hlines(y = -chamber.l_main_z/2, xmin = -chamber.l_main_x/2, xmax = chamber.l_main_x/2, color='black',lw=lw)
        plt.vlines(x = -chamber.l_main_x/2, ymin = -chamber.l_main_z/2, ymax = chamber.l_main_z/2, color='black',lw=lw)
        plt.vlines(x = chamber.l_main_x/2, ymin = -chamber.l_main_z/2, ymax = chamber.l_main_z/2, color='black',lw=lw)
        plt.hlines(y = chamber.l_main_z/2, xmin = -chamber.l_main_x/2, xmax = chamber.l_main_x/2, color='black',lw=lw)
        plt.hlines(y = chamber.l_main_int_z, xmin = -chamber.l_main_int_x, xmax = chamber.l_main_int_x, color='black',lw=lw)
        plt.hlines(y = -chamber.l_main_int_z, xmin = -chamber.l_main_int_x, xmax = chamber.l_main_int_x, color='black',lw=lw)
        plt.vlines(x = -chamber.l_main_int_x, ymin = -chamber.l_main_int_z, ymax = chamber.l_main_int_z, color='black',lw=lw)
        plt.vlines(x = chamber.l_main_int_x, ymin = -chamber.l_main_int_z, ymax = chamber.l_main_int_z, color='black',lw=lw)
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax = plt.subplot2grid((2,2), (1,0))
    im = ax.imshow(np.flipud(ey10.T), cmap = cmap, vmin=mine,vmax = maxe, extent=[xmin, xmax, zmin, zmax], aspect = 'auto')
    if lines:
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
    if lines:
        plt.hlines(y = -chamber.l_beam_pipe/2, xmin = zmin, xmax = -chamber.l_main_z/2)
        plt.hlines(y = chamber.l_beam_pipe/2, xmin = zmin, xmax = -chamber.l_main_z/2)
        plt.hlines(y = -chamber.l_beam_pipe/2, xmin = chamber.l_main_z/2, xmax = zmax)
        plt.hlines(y = chamber.l_beam_pipe/2, xmin = chamber.l_main_z/2, xmax = zmax)
        if wg:
            plt.hlines(y = chamber.y_min_wg, xmin = zmin, xmax = -chamber.l_main_z/2)
            plt.hlines(y = chamber.y_max_wg, xmin = zmin, xmax = -chamber.l_main_z/2)
            plt.vlines(x = -chamber.l_main_z/2, ymin = chamber.l_beam_pipe/2, ymax = chamber.y_min_wg)
            plt.vlines(x = -chamber.l_main_z/2, ymin = chamber.y_max_wg, ymax = chamber.l_main_y/2)
        else:
            plt.vlines(x = -chamber.l_main_z/2, ymin = chamber.l_beam_pipe/2, ymax = chamber.l_main_y/2)
        plt.hlines(y = -chamber.l_main_y/2, xmin = -chamber.l_main_z/2, xmax = -chamber.l_main_int_z)
        plt.hlines(y = chamber.l_main_y/2, xmin = -chamber.l_main_z/2, xmax = -chamber.l_main_int_z)
        plt.hlines(y = -chamber.l_main_y/2, xmin = chamber.l_main_int_z, xmax = chamber.l_main_z/2)
        plt.hlines(y = chamber.l_main_y/2, xmin = chamber.l_main_int_z, xmax = chamber.l_main_z/2)
        plt.hlines(y = chamber.l_beam_pipe/2, xmin = -chamber.l_main_int_z, xmax = chamber.l_main_int_z)
        plt.hlines(y = -chamber.l_beam_pipe/2, xmin = -chamber.l_main_int_z, xmax = chamber.l_main_int_z)
        plt.vlines(x = -chamber.l_main_z/2, ymin = -chamber.l_main_y/2, ymax = -chamber.l_beam_pipe/2)
        plt.vlines(x = chamber.l_main_z/2, ymin = -chamber.l_main_y/2, ymax = -chamber.l_beam_pipe/2)
        plt.vlines(x = chamber.l_main_z/2, ymin = chamber.l_beam_pipe/2, ymax = chamber.l_main_y/2)
        plt.vlines(x = chamber.l_main_int_z, ymin = -chamber.l_main_y/2, ymax = -chamber.l_beam_pipe/2)
        plt.vlines(x = chamber.l_main_int_z, ymin = chamber.l_beam_pipe/2, ymax = chamber.l_main_y/2)
        plt.vlines(x = -chamber.l_main_int_z, ymin = -chamber.l_main_y/2, ymax = -chamber.l_beam_pipe/2)
        plt.vlines(x = -chamber.l_main_int_z, ymin = chamber.l_beam_pipe/2, ymax = chamber.l_main_y/2)
    ax.set_xlabel('z')
    ax.set_ylabel('y')
    #plt.subplot(2,2,4)
    #laser_func_mat = laser_func_plot(xx,yy,pw.top.time)
    #ax = plt.gca()
    #im = ax.imshow(laser_func_mat, cmap = cmap, vmin=mine,vmax = maxe, extent=[ -width/2, width/2,-height/2, height/2])
    #ax.set_xlabel('x')
    #ax.set_ylabel('y')
    cbar_ax = fig.add_axes([0.85, 0.15, 0.03, 0.7]) #[0.8, -0.3, 0.03, 1.8])
    fig.colorbar(im, cax=cbar_ax)
    fig.suptitle(ffstr + ', t = %1.6e' %pw.top.time )
    fig.subplots_adjust(left = 0.15, right=0.8, hspace = 0.4, wspace = 0.4)
    #fig.tight_layout()
    if not os.path.exists(images_dir):
        os.mkdir(images_dir)
    if not os.path.exists(images_dir + '/' + ffstr):
        os.mkdir(images_dir + '/' + ffstr)
    filename = images_dir +'/'+ ffstr + '/it_' + str(pw.top.it).zfill(5) + '.png'
    plt.savefig(filename, dpi=150)
    plt.close(fig)

def plot_fields(ff, ffstr, mine, maxe, chamber, images_dir, l_force=0):
    (Nx, Ny, Nz) = np.shape(ff)
    pw = picmi.warp
    fig, axs = plt.subplots(1, 3, figsize = (15, 4.5))
    fig.subplots_adjust(left = 0.05, bottom = 0.1, right = 0.97, 
                        top = 0.94, wspace = 0.15)
    #d = (self.ecloud.wspecies.get_density()
    #   + self.beam.wspecies.get_density())
    #d2  = (self.ecloud.wspecies.get_density())
    
    cmap = mpl.cm.jet

    im1 = axs[0].imshow(ff[:, :, int(Nz/2)].T, cmap = cmap, 
                        origin = 'lower',
                        vmin = mine,
                        vmax = maxe,
                        extent = [chamber.xmin, chamber.xmax ,
                                  chamber.ymin, chamber.ymax])
    axs[0].set_xlabel('x [m]')
    axs[0].set_ylabel('y [m]')
#    axs[0].set_title(ffstr)
    fig.colorbar(im1, ax = axs[0])
    im2 = axs[1].imshow(ff[int(Nx/2), :, :], cmap = cmap, 
                        origin = 'lower', 
                        vmin = mine, 
                        vmax = maxe,
                        extent=[chamber.zmin, chamber.zmax, 
                                chamber.ymin, chamber.ymax], 
                        aspect = 'auto')
    axs[1].set_xlabel('z [m]')
    axs[1].set_ylabel('y [m]')
#    axs[1].set_title(ffstr)
    fig.colorbar(im2, ax = axs[1])
    im3 = axs[2].imshow(ff[:, int(Ny/2), :], cmap = cmap,
                        origin = 'lower',
                        vmin = mine,
                        vmax = maxe,
                        extent=[chamber.zmin, chamber.zmax,
                                chamber.xmin, chamber.xmax],
                        aspect = 'auto')
    axs[2].set_xlabel('z [m]')
    axs[2].set_ylabel('x [m]')
#    axs[2].set_title(ffstr)
    fig.colorbar(im2, ax = axs[2])
    if not os.path.exists(images_dir):
        os.mkdir(images_dir)
    if not os.path.exists(images_dir + '/' + ffstr):
        os.mkdir(images_dir + '/' + ffstr)
    figname = images_dir +'/'+ ffstr + '/it_' + str(pw.top.it).zfill(5) + '.png'
    #figname = self.images_dir + '/%d.png' %int(self.n_step)
    fig.suptitle(ffstr + ', t = %1.6e' %pw.top.time )
    if not os.path.exists(images_dir):
        os.mkdir(images_dir)
    if not os.path.exists(images_dir + '/' + ffstr):
        os.mkdir(images_dir + '/' + ffstr)
        box_body_v = picmi.warp.Box(xsize=self.L_body-2*self.r_body, ysize=self.L_body, zsize=self.Lz_body)
    plt.savefig(figname)
    plt.close(fig)


def ellipse(xx, a, b, xcent=0, ycent=0):
    return b*np.sqrt(1 - np.square(xx - xcent)/(a*a)) + ycent

def plot_fields_DQW(ff, ffstr, mine, maxe, chamber, images_dir, l_force=0):
    (Nx, Ny, Nz) = np.shape(ff)
    pw = picmi.warp
    fig, axs = plt.subplots(1, 3, figsize = (15, 4.5))
    fig.subplots_adjust(left = 0.05, bottom = 0.1, right = 0.97, 
                        top = 0.94, wspace = 0.15)
    #d = (self.ecloud.wspecies.get_density()
    #   + self.beam.wspecies.get_density())
    #d2  = (self.ecloud.wspecies.get_density())
    
    cmap = mpl.cm.jet

    im1 = axs[0].imshow(ff[:, :, int(Nz/2)].T, cmap = cmap, 
                        origin = 'lower',
                        vmin = mine,
                        vmax = maxe,
                        extent = [chamber.xmin, chamber.xmax ,
                                  chamber.ymin, chamber.ymax])

    xcav = np.array([-chamber.L_out_tot_x,-chamber.L_out_tot_x-chamber.L_slope_ext_x, -chamber.L_int_x-chamber.L_slope_int_x, -chamber.L_int_x, chamber.L_int_x, chamber.L_int_x+chamber.L_slope_int_x, chamber.L_out_tot_x+chamber.L_slope_ext_x, chamber.L_out_tot_x])
    ycav =  np.array([0, chamber.H_out/2, chamber.H_out/2,chamber.r_beam_pipe, chamber.r_beam_pipe, chamber.H_out/2, chamber.H_out/2, 0.])
    axs[0].plot(np.concatenate([xcav, xcav[::-1]]), np.concatenate([ycav, -ycav[::-1]]), color='black')


    axs[0].set_xlabel('x [m]')
    axs[0].set_ylabel('y [m]')
#    axs[0].set_title(ffstr)
    fig.colorbar(im1, ax = axs[0])
    im2 = axs[1].imshow(ff[int(Nx/2), :, :], cmap = cmap, 
                        origin = 'lower', 
                        vmin = mine, 
                        vmax = maxe,
                        extent=[chamber.zmin, chamber.zmax, 
                                chamber.ymin, chamber.ymax], 
                        aspect = 'auto')
    zcav = np.array([chamber.zmin, -0.177,-chamber.L_out_z-chamber.L_slope_ext_z, -chamber.L_int_z-chamber.L_slope_int_z, -chamber.L_int_z, chamber.L_int_z, chamber.L_int_z+chamber.L_slope_int_z, chamber.L_out_z+chamber.L_slope_ext_z, 0.177, chamber.zmax])
    ycav =  np.array([chamber.r_beam_pipe, chamber.r_beam_pipe, chamber.H_out/2, chamber.H_out/2,chamber.r_beam_pipe, chamber.r_beam_pipe, chamber.H_out/2, chamber.H_out/2, chamber.r_beam_pipe, chamber.r_beam_pipe]) 
    axs[1].plot(np.concatenate([zcav, zcav[::-1]]), np.concatenate([ycav, -ycav[::-1]]), color='black')
    axs[1].set_xlabel('z [m]')
    axs[1].set_ylabel('y [m]')
#    axs[1].set_title(ffstr)
    fig.colorbar(im2, ax = axs[1])
    im3 = axs[2].imshow(ff[:, int(Ny/2), :], cmap = cmap,
                        origin = 'lower',
                        vmin = mine,
                        vmax = maxe,
                        extent=[chamber.zmin, chamber.zmax,
                                chamber.xmin, chamber.xmax],
                        aspect = 'auto')

    lb = -chamber.L_out_z*np.sqrt(1-(chamber.r_beam_pipe**2)/(chamber.L_out_tot_x**2))
    ub = -lb
    tt = np.linspace(lb, ub, 100)
    axs[2].plot(tt, ellipse(tt, chamber.L_out_z, chamber.L_out_tot_x), color='black')
    axs[2].plot(tt, -ellipse(tt, chamber.L_out_z, chamber.L_out_tot_x), color='black')
    axs[2].plot([chamber.zmin, lb], [chamber.r_beam_pipe, chamber.r_beam_pipe], color='black')
    axs[2].plot([chamber.zmin, lb], [-chamber.r_beam_pipe, -chamber.r_beam_pipe], color='black')
    axs[2].plot([ub, chamber.zmax], [chamber.r_beam_pipe, chamber.r_beam_pipe], color='black')
    axs[2].plot([ub, chamber.zmax], [-chamber.r_beam_pipe, -chamber.r_beam_pipe], color='black')

    axs[2].set_xlabel('z [m]')
    axs[2].set_ylabel('x [m]')
#    axs[2].set_title(ffstr)
    fig.colorbar(im2, ax = axs[2])

    if not os.path.exists(images_dir):
        os.mkdir(images_dir)
    if not os.path.exists(images_dir + '/' + ffstr):
        os.mkdir(images_dir + '/' + ffstr)
    figname = images_dir +'/'+ ffstr + '/it_' + str(pw.top.it).zfill(5) + '.png'
    #figname = self.images_dir + '/%d.png' %int(self.n_step)
    fig.suptitle(ffstr + ', t = %1.6e' %pw.top.time )
    if not os.path.exists(images_dir):
        os.mkdir(images_dir)
    if not os.path.exists(images_dir + '/' + ffstr):
        os.mkdir(images_dir + '/' + ffstr)
    plt.savefig(figname)
    plt.close(fig)


def plot_fields_RFD(ff, ffstr, mine, maxe, chamber, images_dir, l_force=0):
    (Nx, Ny, Nz) = np.shape(ff)
    pw = picmi.warp
    fig, axs = plt.subplots(1, 3, figsize = (15, 4.5))
    fig.subplots_adjust(left = 0.05, bottom = 0.1, right = 0.97, 
                        top = 0.94, wspace = 0.15)
    #d = (self.ecloud.wspecies.get_density()
    #   + self.beam.wspecies.get_density())
    #d2  = (self.ecloud.wspecies.get_density())
    
    cmap = mpl.cm.jet

    im1 = axs[0].imshow(ff[:, :, int(Nz/2)].T, cmap = cmap, 
                        origin = 'lower',
                        vmin = mine,
                        vmax = maxe,
                        extent = [chamber.xmin, chamber.xmax ,
                                  chamber.ymin, chamber.ymax])

    ycav_b = np.array([-chamber.L_body/2, -chamber.L_body/2])
    xcav_b = np.array([-chamber.L_body/2+chamber.r_body, chamber.L_body/2-chamber.r_body])
    ycav_t = np.array([chamber.L_body/2, chamber.L_body/2])
    xcav_t = np.array([-chamber.L_body/2+chamber.r_body, chamber.L_body/2-chamber.r_body])


    xend_tr = ellipse(chamber.Ly_pole/2, chamber.r_body, chamber.r_body,
                            xcent=chamber.L_body/2-chamber.r_body,
                            ycent=chamber.L_body/2-chamber.r_body)
    xx = np.linspace(chamber.L_body/2-chamber.r_body, xend_tr, 30)
    

    axs[0].plot(xcav_b, ycav_b, color='black')
    axs[0].plot(xcav_t, ycav_t, color='black')
    axs[0].plot(xx, ellipse(xx, chamber.r_body, chamber.r_body, 
                            xcent=chamber.L_body/2-chamber.r_body, 
                            ycent=chamber.L_body/2-chamber.r_body), 
                color='black')

    axs[0].plot(np.array([xend_tr, chamber.L_body/2-chamber.h_pole]), 
                np.array([chamber.Ly_pole/2, chamber.Ly_pole/2]), color='black')
    
    xx = np.linspace(chamber.L_body/2-chamber.h_pole, chamber.r_beam_pipe, 15)

    axs[0].plot(np.array([chamber.L_body/2-chamber.h_pole, chamber.L_body/2-chamber.h_pole]), 
                np.array([chamber.Ly_pole/2, ellipse(xx[0], chamber.r_beam_pipe, 
                                                     chamber.r_beam_pipe)]), 
                color='black')

    axs[0].plot(np.array([chamber.L_body/2-chamber.h_pole, chamber.L_body/2-chamber.h_pole]), 
                np.array([-ellipse(xx[0], chamber.r_beam_pipe, 
                                   chamber.r_beam_pipe),
                          -chamber.Ly_pole/2]),  
                color='black')


    axs[0].plot(xx, ellipse(xx, chamber.r_beam_pipe, chamber.r_beam_pipe), color='black')   
    axs[0].plot(xx, -ellipse(xx, chamber.r_beam_pipe, chamber.r_beam_pipe), color='black')   

    axs[0].plot(np.array([xend_tr, chamber.L_body/2-chamber.h_pole]), 
                np.array([-chamber.Ly_pole/2, -chamber.Ly_pole/2]), color='black')

    xend_tr = ellipse(chamber.Ly_pole/2, chamber.r_body, chamber.r_body,
                                xcent=chamber.L_body/2-chamber.r_body,
                                ycent=chamber.L_body/2-chamber.r_body)

    xx = np.linspace(chamber.L_body/2-chamber.r_body, xend_tr, 30)
    axs[0].plot(xx, -ellipse(xx, chamber.r_body, chamber.r_body, 
                             xcent=chamber.L_body/2-chamber.r_body, 
                             ycent=chamber.L_body/2-chamber.r_body), 
                color='black')

    xx = np.linspace(-xend_tr, -chamber.L_body/2+chamber.r_body, 30)
    axs[0].plot(xx, -ellipse(xx, chamber.r_body, chamber.r_body, 
                             xcent=-chamber.L_body/2+chamber.r_body, 
                             ycent=chamber.L_body/2-chamber.r_body), 
                color='black')


    xx = np.linspace(-xend_tr, -chamber.L_body/2+chamber.r_body, 30)
    axs[0].plot(xx, ellipse(xx, chamber.r_body, chamber.r_body,
                            xcent=-chamber.L_body/2+chamber.r_body,
                            ycent=chamber.L_body/2-chamber.r_body),
                color='black')

    xx = np.linspace(chamber.L_body/2-chamber.h_pole, chamber.r_beam_pipe, 15)

    axs[0].plot(-np.array([chamber.L_body/2-chamber.h_pole, chamber.L_body/2-chamber.h_pole]),
                np.array([chamber.Ly_pole/2, ellipse(xx[0], chamber.r_beam_pipe,
                                                     chamber.r_beam_pipe)]),
                color='black')

    axs[0].plot(-np.array([chamber.L_body/2-chamber.h_pole, chamber.L_body/2-chamber.h_pole]),
                np.array([-ellipse(xx[0], chamber.r_beam_pipe,
                                   chamber.r_beam_pipe),
                          -chamber.Ly_pole/2]),
                color='black')


    axs[0].plot(-xx, ellipse(-xx, chamber.r_beam_pipe, chamber.r_beam_pipe), color='black')
    axs[0].plot(-xx, -ellipse(-xx, chamber.r_beam_pipe, chamber.r_beam_pipe), color='black') 

    axs[0].plot(-np.array([xend_tr, chamber.L_body/2-chamber.h_pole]), 
                 np.array([chamber.Ly_pole/2, chamber.Ly_pole/2]), color='black')

    axs[0].plot(-np.array([xend_tr, chamber.L_body/2-chamber.h_pole]), 
                -np.array([chamber.Ly_pole/2, chamber.Ly_pole/2]), color='black')
 

    axs[0].set_xlabel('x [m]')
    axs[0].set_ylabel('y [m]')
#    axs[0].set_title(ffstr)
    fig.colorbar(im1, ax = axs[0])
        
    im2 = axs[1].imshow(ff[int(Nx/2), :, :], cmap = cmap, 
                        origin = 'lower', 
                        vmin = mine, 
                        vmax = maxe,
                        extent=[chamber.zmin, chamber.zmax, 
                                chamber.ymin, chamber.ymax], 
                        aspect = 'auto')

    axs[1].plot(np.array([-chamber.l_beam_pipe/2, -chamber.Lz_body/2]), np.array([chamber.r_beam_pipe, chamber.r_beam_pipe]), color='black')
    axs[1].plot(np.array([-chamber.l_beam_pipe/2, -chamber.Lz_body/2]), -np.array([chamber.r_beam_pipe, chamber.r_beam_pipe]), color='black')
    axs[1].plot(-np.array([-chamber.l_beam_pipe/2, -chamber.Lz_body/2]), np.array([chamber.r_beam_pipe, chamber.r_beam_pipe]), color='black')
    axs[1].plot(-np.array([-chamber.l_beam_pipe/2, -chamber.Lz_body/2]), -np.array([chamber.r_beam_pipe, chamber.r_beam_pipe]), color='black')


    axs[1].plot(np.array([-chamber.Lz_body/2, chamber.Lz_body/2]), np.array([chamber.L_body/2, chamber.L_body/2]), color='black')
    axs[1].plot(np.array([-chamber.Lz_body/2, chamber.Lz_body/2]), -np.array([chamber.L_body/2, chamber.L_body/2]), color='black')

    axs[1].plot(np.array([-chamber.Lz_body/2, -chamber.Lz_body/2]), np.array([chamber.r_beam_pipe, chamber.L_body/2]), color='black')
    axs[1].plot(np.array([-chamber.Lz_body/2, -chamber.Lz_body/2]), -np.array([chamber.r_beam_pipe, chamber.L_body/2]), color='black')
    axs[1].plot(np.array([chamber.Lz_body/2, chamber.Lz_body/2]), np.array([chamber.r_beam_pipe, chamber.L_body/2]), color='black')
    axs[1].plot(np.array([chamber.Lz_body/2, chamber.Lz_body/2]), -np.array([chamber.r_beam_pipe, chamber.L_body/2]), color='black')

    #zcav = np.array([chamber.zmin, -0.177,-chamber.L_out_z-chamber.L_slope_ext_z, -chamber.L_int_z-chamber.L_slope_int_z, -chamber.L_int_z, chamber.L_int_z, chamber.L_int_z+chamber.L_slope_int_z, chamber.L_out_z+chamber.L_slope_ext_z, 0.177, chamber.zmax])
    #ycav =  np.array([chamber.r_beam_pipe, chamber.r_beam_pipe, chamber.H_out/2, chamber.H_out/2,chamber.r_beam_pipe, chamber.r_beam_pipe, chamber.H_out/2, chamber.H_out/2, chamber.r_beam_pipe, chamber.r_beam_pipe]) 
    #axs[1].plot(np.concatenate([zcav, zcav[::-1]]), np.concatenate([ycav, -ycav[::-1]]), color='black')
    axs[1].set_xlabel('z [m]')
    axs[1].set_ylabel('y [m]')
    
#    axs[1].set_title(ffstr)
    fig.colorbar(im2, ax = axs[1])
    im3 = axs[2].imshow(ff[:, int(Ny/2), :], cmap = cmap,
                        origin = 'lower',
                        vmin = mine,
                        vmax = maxe,
                        extent=[chamber.zmin, chamber.zmax,
                                chamber.xmin, chamber.xmax],
                        aspect = 'auto')

    axs[2].plot(np.array([-chamber.l_beam_pipe/2, -chamber.Lz_body/2]), np.array([chamber.r_beam_pipe, chamber.r_beam_pipe]), color='black')
    axs[2].plot(np.array([-chamber.l_beam_pipe/2, -chamber.Lz_body/2]), -np.array([chamber.r_beam_pipe, chamber.r_beam_pipe]), color='black')
    axs[2].plot(-np.array([-chamber.l_beam_pipe/2, -chamber.Lz_body/2]), np.array([chamber.r_beam_pipe, chamber.r_beam_pipe]), color='black')
    axs[2].plot(-np.array([-chamber.l_beam_pipe/2, -chamber.Lz_body/2]), -np.array([chamber.r_beam_pipe, chamber.r_beam_pipe]), color='black')


    axs[2].plot(np.array([-chamber.Lz_body/2, -chamber.Lz_body/2]), np.array([chamber.r_beam_pipe, chamber.L_body/2]), color='black')
    axs[2].plot(np.array([-chamber.Lz_body/2, -chamber.Lz_body/2]), -np.array([chamber.r_beam_pipe, chamber.L_body/2]), color='black')
    axs[2].plot(np.array([chamber.Lz_body/2, chamber.Lz_body/2]), np.array([chamber.r_beam_pipe, chamber.L_body/2]), color='black')
    axs[2].plot(np.array([chamber.Lz_body/2, chamber.Lz_body/2]), -np.array([chamber.r_beam_pipe, chamber.L_body/2]), color='black')


    axs[2].plot(np.array([-chamber.Lz_pole/2, chamber.Lz_pole/2]), np.array([chamber.r_beam_pipe, chamber.r_beam_pipe]), color='black')
    axs[2].plot(np.array([-chamber.Lz_pole/2, chamber.Lz_pole/2]), -np.array([chamber.r_beam_pipe, chamber.r_beam_pipe]), color='black')

 
    axs[2].plot(np.array([-chamber.Lz_pole/2, -chamber.Lz_pole/2]), np.array([chamber.r_beam_pipe, chamber.L_body/2]), color='black')
    axs[2].plot(np.array([-chamber.Lz_pole/2, -chamber.Lz_pole/2]), -np.array([chamber.r_beam_pipe, chamber.L_body/2]), color='black')
    axs[2].plot(np.array([chamber.Lz_pole/2, chamber.Lz_pole/2]), np.array([chamber.r_beam_pipe, chamber.L_body/2]), color='black')
    axs[2].plot(np.array([chamber.Lz_pole/2, chamber.Lz_pole/2]), -np.array([chamber.r_beam_pipe, chamber.L_body/2]), color='black')


    axs[2].plot(np.array([-chamber.Lz_body/2, -chamber.Lz_pole/2]), np.array([chamber.L_body/2, chamber.L_body/2]), color='black')
    axs[2].plot(np.array([-chamber.Lz_body/2, -chamber.Lz_pole/2]), -np.array([chamber.L_body/2, chamber.L_body/2]), color='black')
 
    axs[2].plot(np.array([chamber.Lz_body/2, chamber.Lz_pole/2]), np.array([chamber.L_body/2, chamber.L_body/2]), color='black')
    axs[2].plot(np.array([chamber.Lz_body/2, chamber.Lz_pole/2]), -np.array([chamber.L_body/2, chamber.L_body/2]), color='black')

    axs[2].set_xlabel('z [m]')
    axs[2].set_ylabel('x [m]')
#    axs[2].set_title(ffstr)
    fig.colorbar(im2, ax = axs[2])
    
    if not os.path.exists(images_dir):
        os.mkdir(images_dir)
    if not os.path.exists(images_dir + '/' + ffstr):
        os.mkdir(images_dir + '/' + ffstr)
    figname = images_dir +'/'+ ffstr + '/it_' + str(pw.top.it).zfill(5) + '.png'
    #figname = self.images_dir + '/%d.png' %int(self.n_step)
    fig.suptitle(ffstr + ', t = %1.6e' %pw.top.time )
    if not os.path.exists(images_dir):
        os.mkdir(images_dir)
    if not os.path.exists(images_dir + '/' + ffstr):
        os.mkdir(images_dir + '/' + ffstr)
    plt.savefig(figname)
    plt.close(fig)

def plot_fields_RFD_cyl(ff, ffstr, mine, maxe, chamber, images_dir, l_force=0):
    (Nx, Ny, Nz) = np.shape(ff)
    pw = picmi.warp
    fig, axs = plt.subplots(1, 3, figsize = (15, 4.5))
    fig.subplots_adjust(left = 0.05, bottom = 0.1, right = 0.97, 
                        top = 0.94, wspace = 0.15)
    #d = (self.ecloud.wspecies.get_density()
    #   + self.beam.wspecies.get_density())
    #d2  = (self.ecloud.wspecies.get_density())
    
    cmap = mpl.cm.jet

    im1 = axs[0].imshow(ff[:, :, int(Nz/2)].T, cmap = cmap, 
                        origin = 'lower',
                        vmin = mine,
                        vmax = maxe,
                        extent = [chamber.xmin, chamber.xmax ,
                                  chamber.ymin, chamber.ymax])

    ycav_b = np.array([-chamber.L_body/2, -chamber.L_body/2])
    xcav_b = np.array([-0.0395, 0.0395])
    ycav_t = np.array([chamber.L_body/2, chamber.L_body/2])
    xcav_t = np.array([-0.0395, 0.0395])


    xend_tr = ellipse(chamber.Ly_pole/2, chamber.r_body, chamber.r_body)
    xx = np.linspace(0.0395, xend_tr, 30)
    

    axs[0].plot(xcav_b, ycav_b, color='black')
    axs[0].plot(xcav_t, ycav_t, color='black')
    axs[0].plot(xx, ellipse(xx, chamber.r_body, chamber.r_body), 
                color='black')

    axs[0].plot(np.array([xend_tr, chamber.L_body/2-chamber.h_pole]), 
                np.array([chamber.Ly_pole/2, chamber.Ly_pole/2]), color='black')
    
    xx = np.linspace(chamber.L_body/2-chamber.h_pole, chamber.r_beam_pipe, 15)

    axs[0].plot(np.array([chamber.L_body/2-chamber.h_pole, chamber.L_body/2-chamber.h_pole]), 
                np.array([chamber.Ly_pole/2, ellipse(xx[0], chamber.r_beam_pipe, 
                                                     chamber.r_beam_pipe)]), 
                color='black')

    axs[0].plot(np.array([chamber.L_body/2-chamber.h_pole, chamber.L_body/2-chamber.h_pole]), 
                np.array([-ellipse(xx[0], chamber.r_beam_pipe, 
                                   chamber.r_beam_pipe),
                          -chamber.Ly_pole/2]),  
                color='black')


    axs[0].plot(xx, ellipse(xx, chamber.r_beam_pipe, chamber.r_beam_pipe), color='black')   
    axs[0].plot(xx, -ellipse(xx, chamber.r_beam_pipe, chamber.r_beam_pipe), color='black')   

    axs[0].plot(np.array([xend_tr, chamber.L_body/2-chamber.h_pole]), 
                np.array([-chamber.Ly_pole/2, -chamber.Ly_pole/2]), color='black')

    xend_tr = ellipse(chamber.Ly_pole/2, chamber.r_body, chamber.r_body)

    xx = np.linspace(0.0395, xend_tr, 30)
    axs[0].plot(xx, -ellipse(xx, chamber.r_body, chamber.r_body), 
                color='black')

    xx = np.linspace(-xend_tr, -0.0395, 30)
    axs[0].plot(xx, -ellipse(xx, chamber.r_body, chamber.r_body), 
                color='black')


    xx = np.linspace(-xend_tr, -0.0395, 30)
    axs[0].plot(xx, ellipse(xx, chamber.r_body, chamber.r_body),
                color='black')

    xx = np.linspace(chamber.L_body/2-chamber.h_pole, chamber.r_beam_pipe, 15)

    axs[0].plot(-np.array([chamber.L_body/2-chamber.h_pole, chamber.L_body/2-chamber.h_pole]),
                np.array([chamber.Ly_pole/2, ellipse(xx[0], chamber.r_beam_pipe,
                                                     chamber.r_beam_pipe)]),
                color='black')

    axs[0].plot(-np.array([chamber.L_body/2-chamber.h_pole, chamber.L_body/2-chamber.h_pole]),
                np.array([-ellipse(xx[0], chamber.r_beam_pipe,
                                   chamber.r_beam_pipe),
                          -chamber.Ly_pole/2]),
                color='black')


    axs[0].plot(-xx, ellipse(-xx, chamber.r_beam_pipe, chamber.r_beam_pipe), color='black')
    axs[0].plot(-xx, -ellipse(-xx, chamber.r_beam_pipe, chamber.r_beam_pipe), color='black') 

    axs[0].plot(-np.array([xend_tr, chamber.L_body/2-chamber.h_pole]), 
                 np.array([chamber.Ly_pole/2, chamber.Ly_pole/2]), color='black')

    axs[0].plot(-np.array([xend_tr, chamber.L_body/2-chamber.h_pole]), 
                -np.array([chamber.Ly_pole/2, chamber.Ly_pole/2]), color='black')
 

    axs[0].set_xlabel('x [m]')
    axs[0].set_ylabel('y [m]')
#    axs[0].set_title(ffstr)
    fig.colorbar(im1, ax = axs[0])
        
    im2 = axs[1].imshow(ff[int(Nx/2), :, :], cmap = cmap, 
                        origin = 'lower', 
                        vmin = mine, 
                        vmax = maxe,
                        extent=[chamber.zmin, chamber.zmax, 
                                chamber.ymin, chamber.ymax], 
                        aspect = 'auto')

    axs[1].plot(np.array([-chamber.l_beam_pipe/2, -chamber.Lz_body/2]), np.array([chamber.r_beam_pipe, chamber.r_beam_pipe]), color='black')
    axs[1].plot(np.array([-chamber.l_beam_pipe/2, -chamber.Lz_body/2]), -np.array([chamber.r_beam_pipe, chamber.r_beam_pipe]), color='black')
    axs[1].plot(-np.array([-chamber.l_beam_pipe/2, -chamber.Lz_body/2]), np.array([chamber.r_beam_pipe, chamber.r_beam_pipe]), color='black')
    axs[1].plot(-np.array([-chamber.l_beam_pipe/2, -chamber.Lz_body/2]), -np.array([chamber.r_beam_pipe, chamber.r_beam_pipe]), color='black')


    axs[1].plot(np.array([-chamber.Lz_body/2, chamber.Lz_body/2]), np.array([chamber.L_body/2, chamber.L_body/2]), color='black')
    axs[1].plot(np.array([-chamber.Lz_body/2, chamber.Lz_body/2]), -np.array([chamber.L_body/2, chamber.L_body/2]), color='black')

    axs[1].plot(np.array([-chamber.Lz_body/2, -chamber.Lz_body/2]), np.array([chamber.r_beam_pipe, chamber.L_body/2]), color='black')
    axs[1].plot(np.array([-chamber.Lz_body/2, -chamber.Lz_body/2]), -np.array([chamber.r_beam_pipe, chamber.L_body/2]), color='black')
    axs[1].plot(np.array([chamber.Lz_body/2, chamber.Lz_body/2]), np.array([chamber.r_beam_pipe, chamber.L_body/2]), color='black')
    axs[1].plot(np.array([chamber.Lz_body/2, chamber.Lz_body/2]), -np.array([chamber.r_beam_pipe, chamber.L_body/2]), color='black')

    #zcav = np.array([chamber.zmin, -0.177,-chamber.L_out_z-chamber.L_slope_ext_z, -chamber.L_int_z-chamber.L_slope_int_z, -chamber.L_int_z, chamber.L_int_z, chamber.L_int_z+chamber.L_slope_int_z, chamber.L_out_z+chamber.L_slope_ext_z, 0.177, chamber.zmax])
    #ycav =  np.array([chamber.r_beam_pipe, chamber.r_beam_pipe, chamber.H_out/2, chamber.H_out/2,chamber.r_beam_pipe, chamber.r_beam_pipe, chamber.H_out/2, chamber.H_out/2, chamber.r_beam_pipe, chamber.r_beam_pipe]) 
    #axs[1].plot(np.concatenate([zcav, zcav[::-1]]), np.concatenate([ycav, -ycav[::-1]]), color='black')
    axs[1].set_xlabel('z [m]')
    axs[1].set_ylabel('y [m]')
    
#    axs[1].set_title(ffstr)
    fig.colorbar(im2, ax = axs[1])
    im3 = axs[2].imshow(ff[:, int(Ny/2), :], cmap = cmap,
                        origin = 'lower',
                        vmin = mine,
                        vmax = maxe,
                        extent=[chamber.zmin, chamber.zmax,
                                chamber.xmin, chamber.xmax],
                        aspect = 'auto')

    axs[2].plot(np.array([-chamber.l_beam_pipe/2, -chamber.Lz_body/2]), np.array([chamber.r_beam_pipe, chamber.r_beam_pipe]), color='black')
    axs[2].plot(np.array([-chamber.l_beam_pipe/2, -chamber.Lz_body/2]), -np.array([chamber.r_beam_pipe, chamber.r_beam_pipe]), color='black')
    axs[2].plot(-np.array([-chamber.l_beam_pipe/2, -chamber.Lz_body/2]), np.array([chamber.r_beam_pipe, chamber.r_beam_pipe]), color='black')
    axs[2].plot(-np.array([-chamber.l_beam_pipe/2, -chamber.Lz_body/2]), -np.array([chamber.r_beam_pipe, chamber.r_beam_pipe]), color='black')


    axs[2].plot(np.array([-chamber.Lz_body/2, -chamber.Lz_body/2]), np.array([chamber.r_beam_pipe, chamber.L_body/2]), color='black')
    axs[2].plot(np.array([-chamber.Lz_body/2, -chamber.Lz_body/2]), -np.array([chamber.r_beam_pipe, chamber.L_body/2]), color='black')
    axs[2].plot(np.array([chamber.Lz_body/2, chamber.Lz_body/2]), np.array([chamber.r_beam_pipe, chamber.L_body/2]), color='black')
    axs[2].plot(np.array([chamber.Lz_body/2, chamber.Lz_body/2]), -np.array([chamber.r_beam_pipe, chamber.L_body/2]), color='black')


    axs[2].plot(np.array([-chamber.Lz_pole/2, chamber.Lz_pole/2]), np.array([chamber.r_beam_pipe, chamber.r_beam_pipe]), color='black')
    axs[2].plot(np.array([-chamber.Lz_pole/2, chamber.Lz_pole/2]), -np.array([chamber.r_beam_pipe, chamber.r_beam_pipe]), color='black')

 
    axs[2].plot(np.array([-chamber.Lz_pole/2, -chamber.Lz_pole/2]), np.array([chamber.r_beam_pipe, chamber.L_body/2]), color='black')
    axs[2].plot(np.array([-chamber.Lz_pole/2, -chamber.Lz_pole/2]), -np.array([chamber.r_beam_pipe, chamber.L_body/2]), color='black')
    axs[2].plot(np.array([chamber.Lz_pole/2, chamber.Lz_pole/2]), np.array([chamber.r_beam_pipe, chamber.L_body/2]), color='black')
    axs[2].plot(np.array([chamber.Lz_pole/2, chamber.Lz_pole/2]), -np.array([chamber.r_beam_pipe, chamber.L_body/2]), color='black')


    axs[2].plot(np.array([-chamber.Lz_body/2, -chamber.Lz_pole/2]), np.array([chamber.L_body/2, chamber.L_body/2]), color='black')
    axs[2].plot(np.array([-chamber.Lz_body/2, -chamber.Lz_pole/2]), -np.array([chamber.L_body/2, chamber.L_body/2]), color='black')
 
    axs[2].plot(np.array([chamber.Lz_body/2, chamber.Lz_pole/2]), np.array([chamber.L_body/2, chamber.L_body/2]), color='black')
    axs[2].plot(np.array([chamber.Lz_body/2, chamber.Lz_pole/2]), -np.array([chamber.L_body/2, chamber.L_body/2]), color='black')

    axs[2].set_xlabel('z [m]')
    axs[2].set_ylabel('x [m]')
#    axs[2].set_title(ffstr)
    fig.colorbar(im2, ax = axs[2])
    
    if not os.path.exists(images_dir):
        os.mkdir(images_dir)
    if not os.path.exists(images_dir + '/' + ffstr):
        os.mkdir(images_dir + '/' + ffstr)
    figname = images_dir +'/'+ ffstr + '/it_' + str(pw.top.it).zfill(5) + '.png'
    #figname = self.images_dir + '/%d.png' %int(self.n_step)
    fig.suptitle(ffstr + ', t = %1.6e' %pw.top.time )
    if not os.path.exists(images_dir):
        os.mkdir(images_dir)
    if not os.path.exists(images_dir + '/' + ffstr):
        os.mkdir(images_dir + '/' + ffstr)
    plt.savefig(figname)
    plt.close(fig)


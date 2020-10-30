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

enable_trap = False

N_mp_max = 60000
init_num_elecs = 2*10**7

nx = 101
ny = 201
nz = 201

# Compute sigmas
nemittx = 2.5e-6
nemitty = nemittx
beta_x = 3500
beta_y = 3500

beam_gamma = 7000.
beam_beta = np.sqrt(1-1/(beam_gamma**2))
sigmax = np.sqrt(beta_x*nemittx/(beam_gamma*beam_beta))
sigmay = np.sqrt(beta_y*nemitty/(beam_gamma*beam_beta))
sigmat= 1.000000e-09/4.


max_z = 0.3

disp = 2e-3
chamber = CrabCavityWaveguide(-max_z, max_z, disp)

n_bunches = 3

# Antenna parameters
laser_source_z = chamber.zmin + 0.5*(-chamber.l_main_z/2 - chamber.zmin)
laser_polangle = np.pi/2
fac = 1.511111111111111
Vt = 3.40e+06
Vt_nom = 3.4e6
laser_emax = 1e7*fac/Vt_nom*Vt
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

# Pre-computations for laser_func
freq_t = 400e6
r_time = 10*2.5e-9
phi = 0
w_t = 2*np.pi*freq_t
ww = laser_xmax - laser_xmin
r_time = 10*2.5e-9
w_z = c_light*np.sqrt((w_t/c_light)**2 - (np.pi/ww)**2)
s_z = 0             # For convenience
cwz = np.cos(w_z*s_z)
swz = np.sin(w_z*s_z)

def laser_func(y, x, t):
    sw = np.sin(-np.pi/ww*(x+ww/2))
    if t<2*r_time:
        amp = amplitude(t, laser_emax, r_time)
        return amp*sw*(np.sin(w_t*t-phi)*cwz - np.cos(w_t*t-phi)*swz)
    else:
        return 0

field_probes = np.array([[int(nx/2),int(ny/2),int(nz/2)]])
t_offs = 2.5e-9*23 - 0.3/c_light - 0.625/c_light

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
                 't_inject_elec': 1e-12}

beam_inputs = {'n_bunches': n_bunches, 'b_spac': 25e-9, 'sigmax': sigmax,
               'sigmay': sigmay, 'sigmat': sigmat, 'beam_gamma': beam_gamma,
               'bunch_intensity': 1.1e11, 'bunch_macro_particles': 10**5,
               't_offs': t_offs}

temps_filename = 'transient_temp.h5'
images_dir = 'images'
output_filename = 'transient_out.h5'

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

    flist = ['elecs']
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
                ff = self.ecloud.wspecies.get_density() + self.beam.wspecies.get_density()
                maxe = 5e9 #np.max(ey[:,:,:])
                mine = 0 #np.min(ey[:,:,:])
                self.proj_elecs_yz[:,:,self.iproj] = np.sum(ff, axis=0)
                self.proj_elecs_xz[:,:,self.iproj] = np.sum(ff, axis=1)
                self.proj_elecs_xy[:,:,self.iproj] = np.sum(ff, axis=2)
                if picmi.warp.top.it%sim.proj_stride==0:
                    dict_out = {}
                    dict_out['elecs_yz'] = self.proj_elecs_yz
                    dict_out['elecs_xz'] = self.proj_elecs_xz
                    dict_out['elecs_xy'] = self.proj_elecs_xy
                    self.iproj+=1
                    filename = sim.proj_filename
                    dict_to_h5(dict_out, filename)
            if ffstr == 'Jy':
                ff = em.gatherjy()
                maxe = np.max(ff[:,:,:])
                mine = np.min(ff[:,:,:])
            if me==0:
                plot_field_crab(ff, ffstr, mine, maxe, k_antenna, j_mid_waveguide, chamber)


saving_inputs = {'flag_checkpointing': True, 'output_filename': output_filename,
                 'checkpoints': np.linspace(1, n_bunches, n_bunches),
                 'temps_filename': temps_filename, 'flag_output': True,
                 'images_dir': images_dir, 'custom_plot': plot_pass,
                 'stride_imgs': 10, 'field_probes': field_probes,
                 'field_probes_dump_stride': 100, 'stride_output': 1000}

antenna_inputs = {}

simulation_inputs = {'enable_trap': enable_trap, 'chamber': chamber,
                     't_end': 1e-11}

sim = warp_pyecloud_sim(fieldsolver_inputs = fieldsolver_inputs,
                        beam_inputs = beam_inputs,
                        ecloud_inputs = ecloud_inputs,
                        antenna_inputs = antenna_inputs,
                        saving_inputs = saving_inputs,
                        simulation_inputs = simulation_inputs)

newsteps = 30
nnxx, nnyy, nnzz = np.shape(sim.ecloud.wspecies.get_density())
sim.proj_stride = 1
sim.proj_elecs_yz = np.zeros((nnyy, nnzz, int(np.ceil(newsteps/sim.proj_stride))))
sim.proj_elecs_xz = np.zeros((nnxx, nnzz, int(np.ceil(newsteps/sim.proj_stride))))
sim.proj_elecs_xy = np.zeros((nnxx, nnyy, int(np.ceil(newsteps/sim.proj_stride))))
sim.iproj = 0

box6 = picmi.warp.Box(zsize=chamber.z_max_wg - chamber.z_rest,
                      xsize=chamber.x_max_wg_rest - chamber.x_min_wg_rest,
                      ysize=chamber.y_max_wg - chamber.y_min_wg,
                      ycent=chamber.ycen6, zcent=chamber.zcen6)

new_conds = box6

sim.t_inject_elec = 4.87e-08

pxs = chamber.l_main_x/2*np.array([1])
pys = chamber.l_main_y/2*np.array([1])
pzs = chamber.l_main_z/2*np.array([1])
planes = []


for px in pxs:
    for py in pys:
        for pz in pzs:
            p = np.array([px, py, pz])
            n = -p/np.linalg.norm(p)
            offs = 1e-2
            theta, phi, z = plane_from_point(px, py, pz)
            if py >= 0:
                z = z - offs
                z = 2e-1
            else:
                z = z + offs
            new_conds = new_conds + picmi.warp.Plane(z0 = z, theta = theta, phi = phi, zsign = 1)

sim.part_scraper.registerconductors(new_conds)

#sim.add_es_solver()
#sim.add_ms_solver()

sim.all_steps_no_ecloud()



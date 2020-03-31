from perform_regeneration import perform_regeneration
import numpy as np
import numpy.random as random
from warp import picmi
from warp import *
from scipy.stats import gaussian_kde
from warp.particles.Secondaries import Secondaries, top, warp, time, dump, restart
import matplotlib.pyplot as plt
from io import StringIO
from scipy.constants import c as clight
import sys
import PyECLOUD.myfilemanager as mfm
import os
import PyECLOUD.sec_emission_model_ECLOUD as seec
from saver import Saver
from h5py_manager import dict_of_arrays_and_scalar_from_h5
import scipy.io as sio
from tqdm import tqdm

class warp_pyecloud_sim:
    def __init__(self, nx = None, ny = None, nz =None, 
                 solver_type = 'ES', n_bunches = None, b_spac = None, 
                 beam_gamma = None, sigmax = None, sigmay = None, 
                 sigmat = None, bunch_intensity = None, init_num_elecs = None,
                 init_num_elecs_mp = None, By = None, N_subcycle = None,
                 pyecloud_nel_mp_ref = None, dt = None, 
                 pyecloud_fact_clean = None, pyecloud_fact_split = None,
                 enable_trap = True, Emax = None, del_max = None, R0 = None, 
                 E_th = None, sigmafit = None, mufit = None,
                 secondary_angle_distribution = None, N_mp_max = None,
                 N_mp_target = None, flag_checkpointing = False, 
                 checkpoints = None, flag_output = False, 
                 bunch_macro_particles = 0, t_offs = None, width = None, 
                 height = None, output_filename = 'output.h5', 
                 flag_relativ_tracking = False, nbins = 100, radius = None, 
                 ghost = None,ghost_z = None, stride_imgs = 10, 
                 stride_output = 1000,chamber = False, lattice_elem = None, 
                 temps_filename = 'temp_mps_info.h5', custom_plot = None,
                 images_dir = None, laser_func = None, laser_source_z = None, 
                 laser_polangle = None, laser_emax = None, laser_xmin = None,
                 laser_xmax = None, laser_ymin = None, laser_ymax = None, 
                 init_em_fields = False, file_em_fields = None, em_scale_fac = 1,
                 EM_method = 'Yee', cfl = 1.0, ecloud_sim = True,
                 folder_em_fields = None): 

        # Construct PyECLOUD secondary emission object
        sey_mod = seec.SEY_model_ECLOUD(Emax = Emax, del_max = del_max, R0 = R0,
                                       E_th = E_th, sigmafit = sigmafit,
                                       mufit = mufit,
                                       secondary_angle_distribution='cosine_3D')

        self.nbins = nbins
        self.N_mp_target = N_mp_target
        self.N_mp_max = N_mp_max
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.flag_checkpointing = flag_checkpointing
        self.checkpoints = checkpoints 
        self.flag_output = flag_output
        self.output_filename = output_filename 
        self.stride_imgs = stride_imgs
        self.stride_output = stride_output
        self.beam_gamma = beam_gamma
        self.chamber = chamber
        self.init_num_elecs_mp = init_num_elecs_mp
        self.init_num_elecs = init_num_elecs
        self.n_bunches = n_bunches
        self.bunch_macro_particles = bunch_macro_particles
        self.sigmat = sigmat
        self.b_spac = b_spac
        self.t_offs = t_offs
        self.temps_filename = temps_filename
        self.custom_plot = custom_plot
        self.images_dir = images_dir
        if not os.path.exists(images_dir):
            os.makedirs(images_dir)
        self.lattice_elem = lattice_elem
        self.laser_func = laser_func
        self.laser_source_z = laser_source_z
        self.laser_polangle = laser_polangle
        self.laser_emax = laser_emax
        self.laser_xmin = laser_xmin
        self.laser_xmax = laser_xmax
        self.laser_ymin = laser_ymin
        self.laser_ymax = laser_ymax
        self.cfl = cfl
        # Just some shortcuts
        pw = picmi.warp
        step = pw.step
        self.init_em_fields = init_em_fields
        self.folder_em_fields = folder_em_fields
        self.file_em_fields = file_em_fields
        self.em_scale_fac = em_scale_fac
        self.enable_trap = enable_trap 

        if solver_type == 'ES':
            pw.top.dt = dt

        elif solver_type == 'EM':
            if dt is not None:
                print('WARNING: dt is going to ignored for the EM solver')
            self.EM_method = EM_method
            
        if flag_relativ_tracking:
            pw.top.lrelativ = pw.true
        else:
            pw.top.lrelativ = pw.false

        # Beam parameters
        sigmaz = sigmat*picmi.clight
        if bunch_macro_particles > 0:
            self.bunch_w = bunch_intensity/bunch_macro_particles
        else:
            self.bunch_w = 0

        self.bunch_rms_size = [sigmax, sigmay, sigmaz]
        self.bunch_rms_velocity = [0., 0., 0.]
        self.bunch_centroid_position = [0, 0, chamber.zmin+1e-4]
        self.bunch_centroid_velocity = [0.,0., beam_gamma*picmi.constants.c]

        # Instantiate beam
        self.beam = picmi.Species(particle_type = 'proton',
                             particle_shape = 'linear',
                             name = 'beam')
        
        # If checkopoint is found reload it, 
        # otherwise start with uniform distribution
        if self.flag_checkpointing and os.path.exists(self.temps_filename): 
            electron_background_dist = self.load_elec_density()
        else:
            if init_num_elecs > 0: 
                electron_background_dist = self.init_uniform_density()
            else:
                electron_background_dist = None
                self.b_pass = 0

        self.flag_first_pass = True

        self.ecloud = picmi.Species(particle_type = 'electron',
                              particle_shape = 'linear',
                              name = 'Electron background',
                              initial_distribution = electron_background_dist)

        # Setup grid and boundary conditions
        if solver_type == 'ES':
            lower_bc = ['dirichlet', 'dirichlet', 'dirichlet']
            upper_bc = ['dirichlet', 'dirichlet', 'dirichlet']
        if solver_type == 'EM':
            lower_bc = ['open', 'open', 'open']
            upper_bc = ['open', 'open', 'open']

        grid = picmi.Cartesian3DGrid(number_of_cells = [self.nx,
                                                        self.ny, 
                                                        self.nz],
                                     lower_bound = [chamber.xmin, 
                                                    chamber.ymin, 
                                                    chamber.zmin],
                                     upper_bound = [chamber.xmax, 
                                                    chamber.ymax, 
                                                    chamber.zmax],
                                     lower_boundary_conditions = lower_bc,
                                     upper_boundary_conditions = upper_bc)

        if solver_type == 'ES':
            self.solver = picmi.ElectrostaticSolver(grid = grid)
        elif solver_type == 'EM':
            smoother = picmi.BinomialSmoother(n_pass = [[1], [1], [1]],
                    compensation = [[False], [False], [False]],
                    stride = [[1], [1], [1]],
                    alpha = [[0.5], [0.5], [0.5]])
            self.solver = picmi.ElectromagneticSolver(grid = grid,
                                          method = self.EM_method,
                                          cfl = self.cfl,
                                          source_smoother = smoother,
                                          warp_l_correct_num_Cherenkov = False,
                                          warp_type_rz_depose = 0,
                                          warp_l_setcowancoefs = True,
                                          warp_l_getrho = False, 
                                          warp_laser_func = self.laser_func,
                                          warp_laser_source_z = self.laser_source_z,
                                          warp_laser_polangle = self.laser_polangle,
                                          warp_laser_emax = self.laser_emax,
                                          warp_laser_xmin = self.laser_xmin,
                                          warp_laser_xmax = self.laser_xmax,
                                          warp_laser_ymin = self.laser_ymin,
                                          warp_laser_ymax = self.laser_ymax)
 
                     
        # Setup simulation
        sim = picmi.Simulation(solver = self.solver, verbose = 1, cfl = self.cfl,
                               warp_initialize_solver_after_generate = 1)


        sim.conductors = chamber.conductors

        sim.add_species(self.beam, layout = None,
                        initialize_self_field = solver_type == 'EM')

        self.ecloud_layout = picmi.PseudoRandomLayout(
                                          n_macroparticles = init_num_elecs_mp,
                                          seed = 3)

        sim.add_species(self.ecloud, layout = self.ecloud_layout,
                        initialize_self_field = solver_type == 'EM')

        if self.bunch_macro_particles > 0:
            picmi.warp.installuserinjection(self.bunched_beam)
        

        sim.step(1)
        self.tot_nsteps = int(np.round(b_spac*(n_bunches)/top.dt))
        self.saver = Saver(flag_output, flag_checkpointing, 
                      self.tot_nsteps, n_bunches, nbins, 
                      temps_filename = temps_filename,
                      output_filename = output_filename)
 
        self.solver.solver.installconductor(sim.conductors, 
                                       dfill = picmi.warp.largepos)
        sim.step(1)

        # Initialize the EM fields
        if self.init_em_fields:
            em = self.solver.solver
            me = pw.me
          
            fields = dict_of_arrays_and_scalar_from_h5_serial(self.folder_em_fields+'/'+str(picmi.warp.me)+'/'+self.file_em_fields) 

            em.fields.Ex = fields['ex']*self.em_scale_fac
            em.fields.Ey = fields['ey']*self.em_scale_fac
            em.fields.Ez = fields['ez']*self.em_scale_fac
            em.fields.Bx = fields['bx']*self.em_scale_fac
            em.fields.By = fields['by']*self.em_scale_fac
            em.fields.Bz = fields['bz']*self.em_scale_fac      
            em.setebp()
  
        # Setup secondary emission stuff       
        pp = warp.ParticleScraper(sim.conductors, lsavecondid = 1, 
                                  lsaveintercept = 1,lcollectlpdata = 1)

        self.sec=Secondaries(conductors = sim.conductors, l_usenew = 1,
                        pyecloud_secemi_object = sey_mod,
                        pyecloud_nel_mp_ref = pyecloud_nel_mp_ref,
                        pyecloud_fact_clean = pyecloud_fact_clean,
                        pyecloud_fact_split = pyecloud_fact_split)

        self.sec.add(incident_species = self.ecloud.wspecies,
                emitted_species  = self.ecloud.wspecies,
                conductor        = sim.conductors)

        if N_subcycle is not None:
            Subcycle(N_subcycle)
        
        if custom_plot is not None:
            plot_func = self.self_wrapped_custom_plot
        else:
            plot_func = self.myplots

        pw.installafterstep(plot_func)

        self.ntsteps_p_bunch = int(np.round(b_spac/top.dt))

        # aux variables
        self.perc = 10
        self.t0 = time.time()
        
        # trapping warp std output
        self.text_trap = {True: StringIO(), False: sys.stdout}[self.enable_trap]
        self.original = sys.stdout

        self.n_step = int(np.round(self.b_pass*self.ntsteps_p_bunch))
        plot_func(1)

    def step(self, u_steps = 1):
        for u_step in range(u_steps):
            # if a passage is starting...
            if (self.n_step%self.ntsteps_p_bunch == 0):
                self.b_pass+=1
                self.perc = 10
                # Measure the duration of the previous passage
                if not self.flag_first_pass:
                    self.t_pass_1 = time.time()
                    self.t_pass = self.t_pass_1-self.t_pass_0

                self.t_pass_0 = time.time()
                # Perform regeneration if needed
                if self.ecloud.wspecies.getn() > self.N_mp_max:
                    print('Number of macroparticles: %d' 
                          %(self.ecloud.wspecies.getn()))
                    print('MAXIMUM LIMIT OF MPS HAS BEEN RACHED')
                    perform_regeneration(self.N_mp_target, 
                                         self.ecloud.wspecies, self.sec)
                 
                # Save stuff if checkpoint
                if (self.flag_checkpointing 
                   and np.any(self.checkpoints == self.b_pass)):
                    self.saver.save_checkpoint(self.b_pass, 
                                               self.ecloud.wspecies)

                print('===========================')
                print('Bunch passage: %d' %(self.b_pass))
                print('Number of electrons: %d' %(np.sum(self.ecloud.wspecies.getw())))
                print('Number of macroparticles: %d' %(self.ecloud.wspecies.getn()))
                if not self.flag_first_pass:
                    print('Previous passage took %ds' %self.t_pass)

                self.flag_first_pass = False

            if ((self.n_step%self.ntsteps_p_bunch)/self.ntsteps_p_bunch*100 
                        > self.perc):
                print('%d%% of bunch passage' %self.perc)
                self.perc = self.perc + 10

            # Dump outputs
            if self.flag_output and self.n_step%self.stride_output == 0:
                self.saver.dump_outputs(self.chamber.xmin, self.chamber.xmax, 
                                        self.ecloud.wspecies, self.b_pass)

            # Perform a step
            sys.stdout = self.text_trap
            picmi.warp.step(1)
            sys.stdout = self.original
            #print(sum(self.ecloud.wspecies.getw()))

            # Store stuff to be saved
            if self.flag_output:
                self.saver.update_outputs(self.ecloud.wspecies, self.nz,
                                          self.n_step) 
            self.n_step += 1

            if self.n_step > self.tot_nsteps:
                # Timer
                t1 = time.time()
                totalt = t1-self.t0
                # Delete checkpoint if found
                #if flag_checkpointing and os.path.exists(self.temps_filename):
                #    os.remove(self.temps_filename)

                print('Run terminated in %ds' %totalt)

    def all_steps(self):
        self.step(self.tot_nsteps-self.n_step)

    def all_steps_no_ecloud(self, n_steps):
        for i in tqdm(range(n_steps)):
            sys.stdout = self.text_trap
            picmi.warp.step(1)
            sys.stdout = self.original

    def init_uniform_density(self):
        chamber = self.chamber
        lower_bound = chamber.lower_bound
        upper_bound = chamber.upper_bound
        init_num_elecs_mp = self.init_num_elecs_mp
        x0 = random.uniform(lower_bound[0], upper_bound[0],
                            init_num_elecs_mp)
        y0 = random.uniform(lower_bound[1], upper_bound[1],
                            init_num_elecs_mp)
        z0 = random.uniform(lower_bound[2], upper_bound[2],
                            init_num_elecs_mp)
        vx0 = np.zeros(init_num_elecs_mp)
        vy0 = np.zeros(init_num_elecs_mp)
        vz0 = np.zeros(init_num_elecs_mp)

        flag_out = chamber.is_outside(x0, y0, z0)
        Nout = np.sum(flag_out)
        while Nout>0:
            x0[flag_out] = random.uniform(lower_bound[0],upper_bound[0],Nout)
            y0[flag_out] = random.uniform(lower_bound[1],upper_bound[1],Nout)
            z0[flag_out] = random.uniform(lower_bound[2],upper_bound[2],Nout)

            flag_out = chamber.is_outside(x0, y0, z0)
            Nout = np.sum(flag_out)
            
        w0 = float(self.init_num_elecs)/float(init_num_elecs_mp)         
    
        self.b_pass = 0

        return picmi.ParticleListDistribution(x = x0, y = y0, z = z0, vx = vx0,
                                              vy = vy0, vz = vz0, weight = w0)



    def load_elec_density(self):
        print('#############################################################')
        print('Temp distribution found. Reloading it as initial distribution')
        print('#############################################################')
        dict_init_dist = dict_of_arrays_and_scalar_from_h5(self.temps_filename)
        # Load particles status
        x0 = dict_init_dist['x_mp']
        y0 = dict_init_dist['y_mp']
        z0 = dict_init_dist['z_mp']
        vx0 = dict_init_dist['vx_mp']
        z0 = dict_init_dist['z_mp']
        vx0 = dict_init_dist['vx_mp']
        vy0 = dict_init_dist['vy_mp']
        vz0 = dict_init_dist['vz_mp']
        w0 = dict_init_dist['nel_mp']
        
        self.b_pass = dict_init_dist['b_pass'] -1 
        self.n_step = int(np.round(self.b_pass*self.b_spac/picmi.warp.top.dt)) 

        return picmi.ParticleListDistribution(x = x0, y = y0, z = z0, vx = vx0,
                                              vy = vy0, vz = vz0, weight = w0)

           

    def time_prof(self, t):
        val = 0
        for i in range(0,self.n_bunches):
            val += (self.bunch_macro_particles*1.
                   /np.sqrt(2*np.pi*self.sigmat**2)
                   *np.exp(-(t-i*self.b_spac-self.t_offs)**2
                   /(2*self.sigmat**2))*picmi.warp.top.dt)
        return val

    def bunched_beam(self):
        NP = int(np.round(self.time_prof(top.time)))
        x = random.normal(self.bunch_centroid_position[0], 
                          self.bunch_rms_size[0], NP)
        y = random.normal(self.bunch_centroid_position[1], 
                          self.bunch_rms_size[1], NP)
        z = self.bunch_centroid_position[2]
        vx = random.normal(self.bunch_centroid_velocity[0], 
                           self.bunch_rms_velocity[0], NP)
        vy = random.normal(self.bunch_centroid_velocity[1], 
                           self.bunch_rms_velocity[1], NP)
        vz = picmi.warp.clight*np.sqrt(1 - 1./(self.beam_gamma**2))
        self.beam.wspecies.addparticles(x = x, y = y, z = z, vx = vx, vy = vy, 
                                        vz = vz, gi = 1./self.beam_gamma,
                                        w = self.bunch_w)

    def myplots(self, l_force=0):
        chamber = self.chamber
        if l_force or self.n_step%self.stride_imgs == 0:
            plt.close()
            (Nx, Ny, Nz) = np.shape(self.ecloud.wspecies.get_density())
            fig, axs = plt.subplots(1, 2, figsize = (12, 4.5))
            fig.subplots_adjust(left = 0.05, bottom = 0.1, right = 0.97, 
                                top = 0.94, wspace = 0.15)
            d = (self.ecloud.wspecies.get_density()
               + self.beam.wspecies.get_density())
            d2  = (self.ecloud.wspecies.get_density())
            im1 = axs[0].imshow(d[:, :, int(Nz/2)] .T, cmap = 'jet', 
                  origin = 'lower', vmin = 0.2*np.min(d2[:, :, int(Nz/2)]), 
                  vmax = 0.8*np.max(d2[:, :, int(Nz/2)]), 
                  extent = [chamber.xmin, chamber.xmax , 
                            chamber.ymin, chamber.ymax])
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
                                aspect = 'auto')
            axs[1].set_xlabel('z [m]')
            axs[1].set_ylabel('y [m]')
            axs[1].set_title('e- density')
            fig.colorbar(im2, ax = axs[1])

            figname = self.images_dir + '/%d.png' %int(self.n_step)
            plt.savefig(figname)

    def self_wrapped_custom_plot(self, l_force = 0):
        self.custom_plot(self, l_force = l_force)


    def dump(self, filename):
        self.solver.solver.laser_func = None
        del self.solver.em3dfft_args['laser_func']
        self.laser_func = None
        self.text_trap = None
        self.original = None        
        self.custom_plot = None
        #del self.chamber
        dump(filename)

    def reinit(self, laser_func, custom_plot):
        self.laser_func = laser_func
        self.custom_plot = custom_plot
        self.solver.solver.laser_func = self.laser_func
        self.solver.em3dfft_args['laser_func'] = self.laser_func
        self.text_trap = {True: StringIO(), False: sys.stdout}[self.enable_trap]
        self.original = sys.stdout
        picmi.warp.installafterstep(self.self_wrapped_custom_plot) 



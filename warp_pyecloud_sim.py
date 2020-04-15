# Self imports
from perform_regeneration import perform_regeneration
from saver import Saver
from h5py_manager import dict_of_arrays_and_scalar_from_h5

# PyECLOUD imports
import PyECLOUD.myfilemanager as mfm
import PyECLOUD.sec_emission_model_ECLOUD as seec
# Warp imports
from warp import picmi, top, time, ParticleScraper
from warp.particles.Secondaries import Secondaries
# Numpy/Matplotlib imports
import numpy as np
import numpy.random as random
import matplotlib.pyplot as plt
# System imports
from io import StringIO
import sys
import os
# Nice outputs
from tqdm import tqdm

class warp_pyecloud_sim(object):
    __fieldsolver_inputs__ = {'nx': None, 'ny': None, 'nz': None,
                              'solver_type': 'ES', 'N_subcycle': None,
                              'EM_method': 'Yee', 'cfl': 1.0,  'dt': None
    }
        
    __beam_inputs__ = {'n_bunches': None, 'b_spac': None, 'beam_gamma': None,
                       'sigmax': None, 'sigmay': None, 'sigmat': None,
                       'bunch_intensity': None, 't_offs': None,
                       'bunch_macro_particles': 0, 'custom_time_prof': None}

    __ecloud_inputs__ = {'init_num_elecs': None, 'init_num_elecs_mp': 0,
                         'pyecloud_nel_mp_ref': None,
                         'pyecloud_fact_clean': None,
                         'pyecloud_fact_split': None,
                         'Emax': None, 'del_max': None,
                         'R0': None, 'E_th': None, 'sigmafit': None,
                         'mufit': None, 'secondary_angle_distribution': None,
                         'N_mp_max': None, 'N_mp_target': None,
                         't_inject_elec': 0
    }
    
    __antenna_inputs__ = {'laser_source_z': None, 'laser_polangle': None,
                         'laser_emax': None, 'laser_xmin': None,
                         'laser_xmax': None, 'laser_ymin': None,
                         'laser_ymax': None, 'laser_func': None
    }

    __saving_inputs__ = {'flag_checkpointing': False, 'checkpoints': None,
                       'flag_output': False, 'output_filename': 'output.h5',
                       'nbins': 100, 'radius': None, 'stride_imgs': 10,
                       'stride_output': 1000,
                       'temps_filename': 'temp_mps_info.h5',
                       'custom_plot': None, 'images_dir': None,
                       'field_probes': [], 'field_probes_dump_stride': 100
    }
    __simulation_inputs__ = {'enable_trap': True,
                             'flag_relativ_tracking': True, 'chamber': False,
                             'lattice_elem': None, 'after_step_fun_list': [],
                             'tot_nsteps': None
    }
    
    ###### CHAMBER SHOULD NOT BE SELF...

    def defaultsfromdict(self,dict,kw):
        for name,defvalue in dict.items():
            if name not in self.__dict__:
                #self.__dict__[name] = kw.pop(name,getattr(top,name)) # Python2.3
                self.__dict__[name] = kw.get(name,defvalue)
            if name in kw: del kw[name]
        if len(kw.keys())!=0: breakpoint()
    
    def __init__(self, fieldsolver_inputs = None,
                 beam_inputs = None, ecloud_inputs = None,
                 antenna_inputs = None,
                 saving_inputs = None,
                 simulation_inputs = None):
                 
        self.defaultsfromdict(self.__fieldsolver_inputs__, fieldsolver_inputs)
        self.defaultsfromdict(self.__beam_inputs__, beam_inputs)
        self.defaultsfromdict(self.__ecloud_inputs__, ecloud_inputs)
        self.defaultsfromdict(self.__antenna_inputs__, antenna_inputs)
        self.defaultsfromdict(self.__saving_inputs__, saving_inputs)
        self.defaultsfromdict(self.__simulation_inputs__, simulation_inputs)
        
        # Construct PyECLOUD secondary emission object
        self.sey_mod = seec.SEY_model_ECLOUD(Emax = self.Emax,
                                             del_max = self.del_max,
                                             R0 = self.R0, E_th = self.E_th,
                                             sigmafit = self.sigmafit,
                                             mufit = self.mufit,
                                       secondary_angle_distribution='cosine_3D')

        self.beam_beta = np.sqrt(1-1/(self.beam_gamma**2))

        if self.n_bunches is not None and self.tot_nsteps is not None:
            print("""WARNING: if both n_bunches and tot_nsteps are specified 
                   tot_nsteps is going to be ignored and the number of steps is 
                   going to be determined basing on n_bunches and dt""")
                   
        if not os.path.exists(self.images_dir) and picmi.warp.me==0:
            os.makedirs(self.images_dir)

        # Just some shortcuts
        pw = picmi.warp
        step = pw.step

        if self.custom_time_prof is None:
            self.time_prof = self.gaussian_time_prof
        else:
            self.time_prof = self.self_wrapped_custom_time_prof

        if self.solver_type == 'ES':
            pw.top.dt = dt
        elif self.solver_type == 'EM':
            if self.dt is not None:
                print('WARNING: dt is going to be ignored for the EM solver')
            #self.EM_method = EM_method
            
        if self.flag_relativ_tracking:
            pw.top.lrelativ = pw.true
        else:
            pw.top.lrelativ = pw.false

        # Beam parameters
        self.sigmaz = self.sigmat*picmi.clight
        if self.bunch_macro_particles > 0:
            self.bunch_w = self.bunch_intensity/self.bunch_macro_particles
        else:
            self.bunch_w = 0

        self.bunch_rms_size = [self.sigmax, self.sigmay, self.sigmaz]
        self.bunch_rms_velocity = [0., 0., 0.]
        self.bunch_centroid_position = [0, 0, self.chamber.zmin + 1e-4]
        self.bunch_centroid_velocity = [0.,0., self.beam_beta*picmi.constants.c]

        # Instantiate beam
        
        self.beam = picmi.Species(particle_type = 'proton',
                                  particle_shape = 'linear',
                                  name = 'beam')
        self.flag_first_pass = True

        if self.flag_checkpointing and os.path.exists(self.temps_filename): 
            self.ecloud = picmi.Species(particle_type = 'electron',
                                        particle_shape = 'linear',
                                        name = 'Electron background',
                                initial_distribution = self.load_elec_density())
        else:
            self.ecloud = picmi.Species(particle_type = 'electron',
                                        particle_shape = 'linear',
                                        name = 'Electron background')
            self.b_pass = 0

    
        # Setup grid and boundary conditions
        if self.solver_type == 'ES':
            lower_bc = ['dirichlet', 'dirichlet', 'dirichlet']
            upper_bc = ['dirichlet', 'dirichlet', 'dirichlet']
        if self.solver_type == 'EM':
            lower_bc = ['open', 'open', 'open']
            upper_bc = ['open', 'open', 'open']

        number_of_cells = [self.nx, self.ny, self.nz]
        lower_bound = [self.chamber.xmin, self.chamber.ymin, self.chamber.zmin]
        upper_bound = [self.chamber.xmax, self.chamber.ymax, self.chamber.zmax]
        
        grid = picmi.Cartesian3DGrid(number_of_cells = number_of_cells,
                                     lower_bound = lower_bound,
                                     upper_bound = upper_bound,
                                     lower_boundary_conditions = lower_bc,
                                     upper_boundary_conditions = upper_bc)

        if self.solver_type == 'ES':
            self.solver = picmi.ElectrostaticSolver(grid = grid)
        elif self.solver_type == 'EM':
            n_pass = [[1], [1], [1]]
            stride = [[1], [1], [1]]
            compensation = [[False], [False], [False]]
            alpha = [[0.5], [0.5], [0.5]]
            smoother = picmi.BinomialSmoother(n_pass = n_pass,
                                     compensation = compensation,
                                     stride = stride,
                                     alpha = alpha)
                                     
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
        sim = picmi.Simulation(solver = self.solver, verbose = 1,
                               cfl = self.cfl,
                               warp_initialize_solver_after_generate = 1)

        sim.conductors = self.chamber.conductors

        sim.add_species(self.beam, layout = None,
                        initialize_self_field = False)

        self.ecloud_layout = picmi.PseudoRandomLayout(
                                      n_macroparticles = self.init_num_elecs_mp,
                                      seed = 3)
        
        sim.add_species(self.ecloud, layout = self.ecloud_layout,
                        initialize_self_field = False)

        if self.bunch_macro_particles > 0:
            picmi.warp.installuserinjection(self.bunched_beam)

        if self.init_num_elecs_mp > 0:
            picmi.warp.installuserinjection(self.init_uniform_density)
        sim.step(1)
        
        if self.tot_nsteps is None and self.n_bunches is not None:
            self.tot_nsteps = int(np.round(b_spac*(self.n_bunches)/top.dt))
        elif self.tot_nsteps is None and self.n_bunches is None:
            raise Exception('One between n_bunches and tot_nsteps has to be specified')

        self.saver = Saver(self.flag_output, self.flag_checkpointing,
                           self.tot_nsteps, self.n_bunches, self.nbins,
                           self.solver, temps_filename = self.temps_filename,
                           output_filename = self.output_filename)
 
        self.solver.solver.installconductor(sim.conductors, 
                                            dfill = picmi.warp.largepos)
        sim.step(1)

        # Initialize the EM fields
        #if self.init_em_fields:
        #    em = self.solver.solver
        #    me = pw.me
        #  
        #    fields = dict_of_arrays_and_scalar_from_h5_serial(self.folder_em_fields+'/'+str(picmi.warp.me)+'/'+self.file_em_fields) 

        #    em.fields.Ex = fields['ex']*self.em_scale_fac
        #    em.fields.Ey = fields['ey']*self.em_scale_fac
        #    em.fields.Ez = fields['ez']*self.em_scale_fac
        #    em.fields.Bx = fields['bx']*self.em_scale_fac
        #    em.fields.By = fields['by']*self.em_scale_fac
        #    em.fields.Bz = fields['bz']*self.em_scale_fac      
        #    em.setebp()
  
        # Setup secondary emission stuff       
        pp = ParticleScraper(sim.conductors, lsavecondid = 1,
                                  lsaveintercept = 1,lcollectlpdata = 1)

        self.sec=Secondaries(conductors = sim.conductors, l_usenew = 1,
                             pyecloud_secemi_object = self.sey_mod,
                             pyecloud_nel_mp_ref = self.pyecloud_nel_mp_ref,
                             pyecloud_fact_clean = self.pyecloud_fact_clean,
                             pyecloud_fact_split = self.pyecloud_fact_split)
        #self.sec=Secondaries(conductors = sim.conductors, l_usenew = 1)

        self.sec.add(incident_species = self.ecloud.wspecies,
                     emitted_species = self.ecloud.wspecies,
                     conductor = sim.conductors)

        if self.N_subcycle is not None:
            Subcycle(self.N_subcycle)
        
        if self.custom_plot is not None:
            plot_func = self.self_wrapped_custom_plot
            pw.installafterstep(plot_func)

        # Install field probes
        if len(self.field_probes)>0:
            self.saver.init_field_probes(np.shape(self.field_probes)[0], 
                                         self.tot_nsteps, 
                                         self.field_probes_dump_stride)

        for i, pos_probe in enumerate(self.field_probes):
            self.pos_probe = pos_probe
            self.ind_probe = i
            pw.installafterstep(self.self_wrapped_probe_fun_i)

        # Install other user-specified functions
        for fun in self.after_step_fun_list:
            pw.installafterstep(fun)

        self.ntsteps_p_bunch = int(np.round(self.b_spac/top.dt))

        # aux variables
        self.perc = 10
        self.t0 = time.time()
        
        # trapping warp std output
        self.text_trap = {True: StringIO(), False: sys.stdout}[self.enable_trap]
        self.original = sys.stdout

        self.n_step = int(np.round(self.b_pass*self.ntsteps_p_bunch))
        plot_func(1)

    def self_wrapped_probe_fun_i(self): 
        self.saver.field_probe(self.ind_probe, self.pos_probe)

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

            # Store stuff to be saved
            if self.flag_output:
                self.saver.update_outputs(self.ecloud.wspecies, self.nz,
                                          self.n_step) 
            self.n_step += 1

            if self.n_step > self.tot_nsteps:
                # Timer
                t1 = time.time()
                totalt = t1-self.t0

                print('Run terminated in %ds' %totalt)

    def all_steps(self):
        for i in tqdm(range(self.tot_nsteps-self.n_step)):
            self.step()
            self.n_step += 1

    def all_steps_no_ecloud(self):
        if picmi.warp.me == 0:
            for i in tqdm(range(self.tot_nsteps)):
                sys.stdout = self.text_trap
                picmi.warp.step(1)
                sys.stdout = self.original
                self.n_step += 1
        else:
            for i in range(self.tot_nsteps):
                sys.stdout = self.text_trap
                picmi.warp.step(1)
                sys.stdout = self.original
                self.n_step += 1

    def init_uniform_density(self):
        pwt = picmi.warp.top
        if np.isclose(pwt.time, self.t_inject_elec, rtol=0, atol=pwt.dt):
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
            gi0 = np.ones(init_num_elecs_mp)

            flag_out = chamber.is_outside(x0, y0, z0)
            Nout = np.sum(flag_out)
            while Nout>0:
                x0[flag_out] = random.uniform(lower_bound[0],
                                              upper_bound[0], Nout)
                y0[flag_out] = random.uniform(lower_bound[1],
                                              upper_bound[1], Nout)
                z0[flag_out] = random.uniform(lower_bound[2],
                                              upper_bound[2], Nout)

                flag_out = chamber.is_outside(x0, y0, z0)
                Nout = np.sum(flag_out)
                
            w0 = float(self.init_num_elecs)/float(init_num_elecs_mp)         

            self.ecloud.wspecies.addparticles(x = x0, y = y0, z = z0, vx = vx0,
                                              vy = vy0, vz = vz0, gi = gi0,
                                              w = w0)

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
        vy0 = dict_init_dist['vy_mp']
        vz0 = dict_init_dist['vz_mp']
        w0 = dict_init_dist['nel_mp']

        self.b_pass = dict_init_dist['b_pass'] -1 
        self.n_step = int(np.round(self.b_pass*self.b_spac/picmi.warp.top.dt)) 

        return picmi.ParticleListDistribution(x = x0, y = y0, z = z0, vx = vx0,
                                              vy = vy0, vz = vz0, weight = w0)

    def gaussian_time_prof(self):
        t = picmi.warp.top.time
        val = 0
        for i in range(0,self.n_bunches):
            val += (self.bunch_macro_particles*1.
                   /np.sqrt(2*np.pi*self.sigmat**2)
                   *np.exp(-(t-i*self.b_spac-self.t_offs)**2
                   /(2*self.sigmat**2))*picmi.warp.top.dt)
        return val

    def bunched_beam(self):
        NP = int(np.round(self.time_prof()))
        if NP > 0:
            x = random.normal(self.bunch_centroid_position[0],
                                self.bunch_rms_size[0], NP)
            y = random.normal(self.bunch_centroid_position[1],
                              self.bunch_rms_size[1], NP)
            z = self.bunch_centroid_position[2]
            vx = random.normal(self.bunch_centroid_velocity[0],
                               self.bunch_rms_velocity[0], NP)
            vy = random.normal(self.bunch_centroid_velocity[1],
                               self.bunch_rms_velocity[1], NP)
            vz = picmi.warp.clight*self.beam_beta
            self.beam.wspecies.addparticles(x = x, y = y, z = z, vx = vx,
                                            vy = vy, vz = vz,
                                            gi = 1./self.beam_gamma,
                                            w = self.bunch_w)

    def self_wrapped_custom_plot(self, l_force = 0):
        self.custom_plot(self, l_force = l_force)

    def self_wrapped_custom_time_prof(self):
       return self.custom_time_prof(self)




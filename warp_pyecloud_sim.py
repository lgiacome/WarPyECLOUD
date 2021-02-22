# Self imports
from perform_regeneration import perform_regeneration
from saver import Saver
from h5py_manager import dict_of_arrays_and_scalar_from_h5

# PyECLOUD imports
import PyECLOUD.myfilemanager as mfm
import PyECLOUD.sec_emission_model_ECLOUD as seec
# Warp imports
from warp import picmi, top, time, ParticleScraper, registersolver, pprint
from warp import dump as warpdump
from warp.particles.Secondaries import Secondaries
# Numpy/Matplotlib imports
import numpy as np
import numpy.random as random
import matplotlib.pyplot as plt
from scipy.constants import c as c_light
# System imports
from io import StringIO
import sys
import os
# Nice outputs
from tqdm import tqdm

from mpi4py import MPI


class warp_pyecloud_sim(object):
    __fieldsolver_inputs__ = {'nx': None, 'ny': None, 'nz': None,
                              'solver_type': 'ES', 'N_subcycle': None,
                              'EM_method': 'Yee', 'cfl': 1.0, 'dt': None,
                              'source_smoothing': True}

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
                         't_inject_elec': 0, 'init_ecloud_fields': False,
                         }

    __antenna_inputs__ = {'laser_source_z': None, 'laser_polangle': None,
                          'laser_emax': None, 'laser_xmin': None,
                          'laser_xmax': None, 'laser_ymin': None,
                          'laser_ymax': None, 'laser_func': None
                          }

    __saving_inputs__ = {'flag_checkpointing': False, 'checkpoints': None,
                         'flag_output': True, 'output_filename': 'output.h5',
                         'nbins': 100, 'radius': None, 'stride_imgs': 10,
                         'stride_output': 1000,
                         'temps_filename': 'temp_mps_info.h5',
                         'custom_plot': None, 'images_dir': None,
                         'field_probes': [], 'field_probes_dump_stride': 1000,
                         'probe_filename': 'probe.h5'
                         }

    __simulation_inputs__ = {'enable_trap': True,
                             'flag_relativ_tracking': True, 'chamber': False,
                             'lattice_elem': None, 'after_step_fun_list': [],
                             'tot_nsteps': None, 't_end': None
                             }

    def __init__(self, fieldsolver_inputs=None,
                 beam_inputs=None, ecloud_inputs=None,
                 antenna_inputs=None,
                 saving_inputs=None,
                 simulation_inputs=None):

        self.defaultsfromdict(self.__fieldsolver_inputs__, fieldsolver_inputs)
        self.defaultsfromdict(self.__beam_inputs__, beam_inputs)
        self.defaultsfromdict(self.__ecloud_inputs__, ecloud_inputs)
        self.defaultsfromdict(self.__antenna_inputs__, antenna_inputs)
        self.defaultsfromdict(self.__saving_inputs__, saving_inputs)
        self.defaultsfromdict(self.__simulation_inputs__, simulation_inputs)

        # Construct PyECLOUD secondary emission object
        self.sey_mod = seec.SEY_model_ECLOUD(Emax=self.Emax,
                                             del_max=self.del_max,
                                             R0=self.R0, E_th=self.E_th,
                                             sigmafit=self.sigmafit,
                                             mufit=self.mufit,
                                             secondary_angle_distribution='cosine_3D')

        self.beam_beta = np.sqrt(1 - 1 / (self.beam_gamma ** 2))

        if self.n_bunches is not None and self.tot_nsteps is not None:
            print("""WARNING: if both n_bunches and tot_nsteps are specified 
                   tot_nsteps is going to be ignored and the number of steps is 
                   going to be determined basing on n_bunches and dt""")

        if self.n_bunches is not None and self.t_end is not None:
            print("""WARNING: if both n_bunches and t_end are specified
                   tot_nsteps is going to be ignored and the number of steps is
                   going to be determined basing on n_bunches and dt""")

        if not os.path.exists(self.images_dir) and picmi.warp.me == 0:
            os.makedirs(self.images_dir)

        # Just some shortcuts
        pw = picmi.warp

        if self.custom_time_prof is None:
            self.time_prof = self.gaussian_time_prof
        else:
            self.time_prof = self.self_wrapped_custom_time_prof

        if self.solver_type == 'EM':
            if self.dt is not None:
                print('WARNING: dt is going to be ignored for the EM solver')
        else:
            pw.top.dt = self.dt

        if self.flag_relativ_tracking:
            pw.top.lrelativ = pw.true
        else:
            pw.top.lrelativ = pw.false

        # Beam parameters
        self.sigmaz = self.sigmat * picmi.clight
        if self.bunch_macro_particles > 0:
            self.bunch_w = self.bunch_intensity / self.bunch_macro_particles
        else:
            self.bunch_w = 0

        self.bunch_rms_size = [self.sigmax, self.sigmay, self.sigmaz]
        self.bunch_rms_velocity = [0., 0., 0.]
        self.bunch_centroid_position = [0, 0, self.chamber.z_inj_beam]
        self.bunch_centroid_velocity = [0., 0., self.beam_beta * picmi.constants.c]

        self.species_names = ['beam', 'ecloud']

        # Instantiate beam

        if self.flag_checkpointing and os.path.exists(self.temps_filename):
            self.ecloud = picmi.Species(particle_type='electron',
                                        particle_shape='linear',
                                        name=self.species_names[1],
                                        initial_distribution=self.load_elec_density())
        elif self.t_inject_elec == 0:
            
            lower_bound = self.chamber.lower_bound
            upper_bound = self.chamber.upper_bound
            vol = np.prod(np.array(upper_bound) - np.array(lower_bound))
            dens = self.init_num_elecs/vol           
            #unif_dist = picmi.UniformDistribution(dens,
            #                                      lower_bound = lower_bound,
            #                                      upper_bound = upper_bound) 
 
            x0, y0, z0, vx0, vy0, vz0, gi0, w0 = self.uniform_density() 
            unif_dist = picmi.ParticleListDistribution(x=x0, y=y0,
                                                          z=z0, ux=vx0/gi0,
                                                          uy=vy0/gi0, uz=vz0/gi0,
                                                          weight=w0)
            self.ecloud = picmi.Species(particle_type='electron',
                                        particle_shape='linear',
                                        name=self.species_names[1],
                                        initial_distribution = unif_dist)
        else:
            self.ecloud = picmi.Species(particle_type='electron',
                                        particle_shape='linear',
                                        name=self.species_names[1])

        self.beam = picmi.Species(particle_type='proton',
                                  particle_shape='linear',
                                  name=self.species_names[0],
                                  warp_fselfb = self.bunch_centroid_velocity[2])


        self.b_pass = 0
        # Setup grid and boundary conditions
        self.dir_bc = ['dirichlet', 'dirichlet', 'dirichlet']
        self.pml_bc = ['open', 'open', 'open']
        #self.pml_bc = ['dirichlet', 'dirichlet', 'dirichlet']

        self.number_of_cells = [self.nx, self.ny, self.nz]
        self.lower_bound = [self.chamber.xmin, self.chamber.ymin, self.chamber.zmin]
        self.upper_bound = [self.chamber.xmax, self.chamber.ymax, self.chamber.zmax]
        
        self.grid_EM = picmi.Cartesian3DGrid(number_of_cells=self.number_of_cells,
                                        lower_bound=self.lower_bound,
                                        upper_bound=self.upper_bound,
                                        lower_boundary_conditions=self.pml_bc,
                                        upper_boundary_conditions=self.pml_bc)


        grid_ES = picmi.Cartesian3DGrid(number_of_cells=self.number_of_cells,
                                        lower_bound=self.lower_bound,
                                        upper_bound=self.upper_bound,
                                        lower_boundary_conditions=self.dir_bc,
                                        upper_boundary_conditions=self.dir_bc)

        if self.solver_type == 'ES':
            self.solver = picmi.ElectrostaticSolver(grid=grid_ES,
                                                    warp_conductors = self.chamber.conductors,
                                                    warp_conductor_dfill = picmi.warp.largepos)

        elif self.solver_type == 'EM':
            if self.source_smoothing:
                n_pass = [[1], [1], [1]]
                stride = [[1], [1], [1]]
                compensation = [[False], [False], [False]]
                alpha = [[0.5], [0.5], [0.5]]
                smoother = picmi.BinomialSmoother(n_pass=n_pass,
                                                  compensation=compensation,
                                                  stride=stride,
                                                  alpha=alpha)
            else:
                smoother = None

            if hasattr(self, 'laser_func'):
                self.solver = picmi.ElectromagneticSolver(grid=self.grid_EM,
                                                      method=self.EM_method, cfl=self.cfl,
                                                      source_smoother=smoother,
                                                      warp_l_correct_num_Cherenkov=False,
                                                      warp_type_rz_depose=0,
                                                      warp_l_setcowancoefs=True,
                                                      warp_l_getrho=False,
                                                      warp_laser_func=self.laser_func,
                                                      warp_laser_source_z=self.laser_source_z,
                                                      warp_laser_polangle=self.laser_polangle,
                                                      warp_laser_emax=self.laser_emax,
                                                      warp_laser_xmin=self.laser_xmin,
                                                      warp_laser_xmax=self.laser_xmax,
                                                      warp_laser_ymin=self.laser_ymin,
                                                      warp_laser_ymax=self.laser_ymax,
                                                      warp_conductors = self.chamber.conductors,
                                                      warp_conductor_dfill = picmi.warp.largepos,
                                                      warp_deposition_species =[self.ecloud.wspecies],
                                                      warp_iselfb_list = [0]) 
            else:
                self.solver = picmi.ElectromagneticSolver(grid=self.grid_EM,
                                                      method=self.EM_method, cfl=self.cfl,
                                                      source_smoother=smoother,
                                                      warp_l_correct_num_Cherenkov=False,
                                                      warp_type_rz_depose=0,
                                                      warp_l_setcowancoefs=True,
                                                      warp_l_getrho=False,
                                                      warp_conductors = self.chamber.conductors,
                                                      warp_conductor_dfill = picmi.warp.largepos,
                                                      warp_deposition_species =[self.ecloud.wspecies],
                                                      warp_iselfb_list = [0])     


        # Setup simulation
        self.sim = picmi.Simulation(solver=self.solver, verbose=1,
                               warp_initialize_solver_after_generate=1)

        self.sim.add_species(self.beam, layout=None,
                        initialize_self_field=False)

        self.ecloud_layout = picmi.PseudoRandomLayout(
            n_macroparticles=self.init_num_elecs_mp,
            seed=3)

        #tot_cells = 2 #self.nx*self.ny*self.nz
        #self.ecloud_layout = picmi.GriddedLayout(grid=self.grid_EM,
        #    n_macroparticle_per_cell= np.array([2, 2, 2])) #np.array([int(self.init_num_elecs_mp/tot_cells), 
                                               #int(self.init_num_elecs_mp/tot_cells), 
                                               #int(self.init_num_elecs_mp/tot_cells)]))

        self.sim.add_species(self.ecloud, layout=self.ecloud_layout,
                        initialize_self_field= self.init_ecloud_fields) 

        
        #init_field = self.solver_type=='EM'


#        if self.init_num_elecs_mp > 0:
#            picmi.warp.installuserinjection(self.init_uniform_density)
       
        self.sim.step(1)

        if self.tot_nsteps is None and self.n_bunches is not None:
            self.tot_nsteps = int(np.round(self.b_spac*(self.n_bunches)/top.dt))
        if self.tot_nsteps is None and self.t_end is not None:
            self.tot_nsteps = int(np.round(self.t_end / top.dt))
        # to be fixed
        if self.t_end is not None:
            self.tot_nsteps = int(np.round(self.t_end / top.dt))
        elif self.tot_nsteps is None and self.n_bunches is None:
            raise Exception('One between n_bunches, tot_nsteps, t_end has to be specified')
        
        # needed to be consistent with the conductors
        self.solver.solver.current_cor = False 

        self.flag_first_pass = True

        if self.bunch_macro_particles > 0:
            picmi.warp.installuserinjection(self.bunched_beam)     
        # Initialize the EM fields
        # if self.init_em_fields:
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
        self.part_scraper = ParticleScraper(self.chamber.conductors, lsavecondid=1, lsaveintercept=1, lcollectlpdata=1)

        self.sec = Secondaries(conductors=self.chamber.conductors, l_usenew=1,
                               pyecloud_secemi_object=self.sey_mod,
                               pyecloud_nel_mp_ref=self.pyecloud_nel_mp_ref,
                               pyecloud_fact_clean=self.pyecloud_fact_clean,
                               pyecloud_fact_split=self.pyecloud_fact_split)
 
        self.sec.add(incident_species=self.ecloud.wspecies,
                     emitted_species=self.ecloud.wspecies,
                     conductor=self.chamber.conductors)
        
        self.saver = Saver(self.flag_output, self.flag_checkpointing,
                           self.nbins,
                           self.solver, None, temps_filename=self.temps_filename,
                           output_filename=self.output_filename,
                           probe_filename = self.probe_filename,
                           tot_nsteps = self.tot_nsteps, n_bunches = self.n_bunches)


        self.ntsteps_p_bunch = int(np.round(self.b_spac / top.dt))
        self.n_step = int(np.round(self.b_pass * self.ntsteps_p_bunch))
        if self.N_subcycle is not None:
            Subcycle(self.N_subcycle)

        if self.custom_plot is not None:
            pw.installafterstep(self.self_wrapped_custom_plot)

        # Install field probes
        if len(self.field_probes) > 0:
            self.saver.init_field_probes(np.shape(self.field_probes)[0],
                                         self.tot_nsteps,
                                         self.field_probes_dump_stride)

            pw.installafterstep(self.self_wrapped_probe_fun)

        # Install other user-specified functions
        for fun in self.after_step_fun_list:
            pw.installafterstep(fun)

        # aux variables
        self.perc = 10
        self.t0 = time.time()

        # trapping warp std output
        self.text_trap = {True: StringIO(), False: sys.stdout}[self.enable_trap]
        self.original = sys.stdout
        self.print_solvers_info()

    def print_solvers_info(self):
        # printing info about the sim
        print('dx = %e' %self.solver.solver.dx) 
        print('dy = %e' %self.solver.solver.dy) 
        print('dz = %e' %self.solver.solver.dz) 
        dtCFL = self.cfl/(c_light*np.sqrt(1/self.solver.solver.dx**2 + 
                                     1/self.solver.solver.dy**2 + 
                                     1/self.solver.solver.dz**2))

        print('EM solver: dt = %e' %dtCFL)
        print('ES solver: dt = %e' %picmi.warp.top.dt)

    def add_es_solver(self, deposition_species = []):
        grid_ES = picmi.Cartesian3DGrid(number_of_cells=self.number_of_cells,
                                        lower_bound=self.lower_bound,
                                        upper_bound=self.upper_bound,
                                        lower_boundary_conditions=self.dir_bc,
                                        upper_boundary_conditions=self.dir_bc)

        self.ES_solver = picmi.ElectrostaticSolver(grid=grid_ES, 
                                                   warp_deposition_species=deposition_species,
                                                   warp_conductors = self.chamber.conductors,
                                                   warp_conductor_dfill = picmi.warp.largepos)

        self.ES_solver.initialize_solver_inputs()
        registersolver(self.ES_solver.solver)

    def add_em_solver(self, deposition_species = []):
        grid_EM = picmi.Cartesian3DGrid(number_of_cells=self.number_of_cells,
                                        lower_bound=self.lower_bound,
                                        upper_bound=self.upper_bound,
                                        lower_boundary_conditions=self.pml_bc,
                                        upper_boundary_conditions=self.pml_bc)

        n_pass = [[1], [1], [1]]
        stride = [[1], [1], [1]]
        compensation = [[False], [False], [False]]
        alpha = [[0.5], [0.5], [0.5]]
        smoother = picmi.BinomialSmoother(n_pass=n_pass,
                                          compensation=compensation,
                                          stride=stride,
                                          alpha=alpha)

        self.EM_solver = picmi.ElectromagneticSolver(grid=self.grid_EM,
                                                     method=self.EM_method, cfl=self.cfl,
                                                     source_smoother=smoother,
                                                     warp_l_correct_num_Cherenkov=False,
                                                     warp_type_rz_depose=0,
                                                     warp_l_setcowancoefs=True,
                                                     warp_l_getrho=False,
                                                     warp_deposition_species=deposition_species,
                                                     warp_conductors = self.chamber.conductors,
                                                     warp_conductor_dfill = picmi.warp.largepos)

        self.EM_solver.initialize_solver_inputs()
        registersolver(self.EM_solver.solver)

    def add_ms_solver(self, deposition_species = []):
        grid_MS = picmi.Cartesian3DGrid(number_of_cells=self.number_of_cells,
                                        lower_bound=self.lower_bound,
                                        upper_bound=self.upper_bound,
                                        lower_boundary_conditions=self.dir_bc,
                                        upper_boundary_conditions=self.dir_bc,
                                        warp_conductors = self.chamber.conductors,
                                        warp_conductor_dfill = picmi.warp.largepos)

        self.MS_solver = picmi.MagnetostaticSolver(grid=grid_MS, warp_deposition_species=deposition_species)

        self.MS_solver.initialize_solver_inputs()
        registersolver(self.MS_solver.solver)

    def distribute_species(self, primary_species=None, es_species=None, em_species=None, ms_species=None):
        if primary_species is not None:
            self.solver.solver.deposition_species = primary_species
        if es_species is not None:
            self.ES_solver.solver.deposition_species = es_species
        if em_species is not None:
            self.EM_solver.solver.deposition_species = em_species
        if ms_species is not None:
            self.MS_solver.solver.deposition_species = ms_species

    def self_wrapped_probe_fun(self):
        self.saver.update_field_probes(self.field_probes)

    def step(self, u_steps=1):
        for u_step in range(u_steps):
            # if a passage is starting...
            if self.n_step % self.ntsteps_p_bunch == 0:
                self.b_pass += 1
                self.perc = 10
                # Measure the duration of the previous passage
                if not self.flag_first_pass:
                    self.t_pass_1 = time.time()
                    self.t_pass = self.t_pass_1 - self.t_pass_0

                self.t_pass_0 = time.time()
                # Perform regeneration if needed
                if self.ecloud.wspecies.getn() > self.N_mp_max:
                    print('Number of macroparticles: %d'
                          % (self.ecloud.wspecies.getn()))
                    print('MAXIMUM LIMIT OF MPS HAS BEEN RACHED')
                    perform_regeneration(self.N_mp_target,
                                         self.ecloud.wspecies, self.sec)

                # Save stuff if checkpoint
                if (self.flag_checkpointing
                        and np.any(self.checkpoints == self.b_pass)):
                    self.saver.save_checkpoint(self.b_pass,
                                               self.ecloud.wspecies)

                print('===========================')
                print('Bunch passage: %d' %self.b_pass)
                print('Number of electrons: %d' % (np.sum(self.ecloud.wspecies.getw())))
                print('Number of macroparticles: %d' % (self.ecloud.wspecies.getn()))
                if not self.flag_first_pass:
                    print('Previous passage took %ds' % self.t_pass)

                self.flag_first_pass = False

            if ((self.n_step % self.ntsteps_p_bunch) / self.ntsteps_p_bunch * 100
                    > self.perc):
                print('%d%% of bunch passage' % self.perc)
                self.perc = self.perc + 10

            # Dump outputs
            if self.flag_output and self.n_step % self.stride_output == 0:
                self.saver.dump_outputs(self.chamber.xmin, self.chamber.xmax,
                                        self.ecloud.wspecies, self.b_pass)

            # Perform a step
            sys.stdout = self.text_trap
            picmi.warp.step()
            sys.stdout = self.original

            # Store stuff to be saved
            if self.flag_output:
                self.saver.update_outputs(self.ecloud.wspecies, self.beam.wspecies, self.nz,
                                          self.n_step)

            if self.n_step > self.tot_nsteps:
                # Timer
                t1 = time.time()
                totalt = t1 - self.t0

                print('Run terminated in %ds' % totalt)

    def all_steps(self):
        for i in range(self.n_step, self.tot_nsteps):
            self.step()
            self.n_step += 1
            #print(self.beam.wspecies.getn())

    def all_steps_no_ecloud(self):
        if picmi.warp.me == 0:
            for i in tqdm(range(self.n_step, self.tot_nsteps)):
                sys.stdout = self.text_trap
                picmi.warp.step(1)
                sys.stdout = self.original
                if self.flag_output:
                    self.saver.update_outputs(self.ecloud.wspecies, self.beam.wspecies, 
                                                self.nz, self.n_step)
                if self.flag_output and self.n_step % self.stride_output == 0:
                    self.saver.dump_outputs(self.chamber.xmin, self.chamber.xmax, self.ecloud.wspecies, self.b_pass)
                # Perform regeneration if needed
                if self.ecloud.wspecies.getn() > self.N_mp_max:
                    print('Number of macroparticles: %d' %self.ecloud.wspecies.getn())
                    print('MAXIMUM LIMIT OF MPS HAS BEEN RACHED')
                    perform_regeneration(self.N_mp_target, self.ecloud.wspecies, self.sec)

                self.n_step += 1
        else:
            for i in range(self.n_step, self.tot_nsteps):
                sys.stdout = self.text_trap
                picmi.warp.step()
                sys.stdout = self.original
                if self.flag_output:
                    self.saver.update_outputs(self.ecloud.wspecies, self.beam.wspecies,
                                              self.nz, self.n_step)
                if self.flag_output and self.n_step % self.stride_output == 0:
                    self.saver.dump_outputs(self.chamber.xmin, self.chamber.xmax, self.ecloud.wspecies, self.b_pass)
                # Perform regeneration if needed
                if self.ecloud.wspecies.getn() > self.N_mp_max:
                    print('Number of macroparticles: %d' %self.ecloud.wspecies.getn())
                    print('MAXIMUM LIMIT OF MPS HAS BEEN RACHED')
                    perform_regeneration(self.N_mp_target, self.ecloud.wspecies, self.sec)

                self.n_step += 1

    def uniform_density(self):
        pwt = picmi.warp.top
        init_num_elecs_mp = self.init_num_elecs_mp

        x0 = np.empty(init_num_elecs_mp, dtype=float)
        y0 = np.empty(init_num_elecs_mp, dtype=float)
        z0 = np.empty(init_num_elecs_mp, dtype=float)
        vx0 = np.empty(init_num_elecs_mp, dtype=float)
        vy0 = np.empty(init_num_elecs_mp, dtype=float)
        vz0 = np.empty(init_num_elecs_mp, dtype=float)
        gi0 = np.empty(init_num_elecs_mp, dtype=float)

        if picmi.warp.me == 0:
            chamber = self.chamber
            lower_bound = chamber.lower_bound
            upper_bound = chamber.upper_bound
            x0 = random.uniform(lower_bound[0], upper_bound[0],
                                init_num_elecs_mp)
            y0 = random.uniform(lower_bound[1], upper_bound[1],
                                init_num_elecs_mp)
            z0 = random.uniform(lower_bound[2], upper_bound[2],
                                init_num_elecs_mp)
            vx0 = np.zeros(init_num_elecs_mp)
            vy0 = np.zeros(init_num_elecs_mp)
            vz0 = np.zeros(init_num_elecs_mp)
            gi0 = c_light/np.sqrt(c_light**2 - np.square(vx0) - np.square(vy0) - np.square(vz0))
            flag_out = chamber.is_outside(x0, y0, z0)
            Nout = np.sum(flag_out)
            while Nout > 0:
                x0[flag_out] = random.uniform(lower_bound[0],
                                              upper_bound[0], Nout)
                y0[flag_out] = random.uniform(lower_bound[1],
                                              upper_bound[1], Nout)
                z0[flag_out] = random.uniform(lower_bound[2],
                                              upper_bound[2], Nout)

                flag_out = chamber.is_outside(x0, y0, z0)
                Nout = np.sum(flag_out)

        comm = MPI.COMM_WORLD

        comm.Bcast(x0, root=0)
        comm.Bcast(y0, root=0)
        comm.Bcast(z0, root=0)
        comm.Bcast(vx0, root=0)
        comm.Bcast(vy0, root=0)
        comm.Bcast(vz0, root=0)
        comm.Bcast(gi0, root=0)

        w0 = float(self.init_num_elecs) / float(init_num_elecs_mp)
        return x0, y0, z0, vx0, vy0, vz0, gi0, w0

    def init_uniform_density(self):
        if np.isclose(pwt.time, self.t_inject_elec, rtol=0, atol=pwt.dt) and (pwt.time - self.t_inject_elec) > 0 and self.t_inject_elec!=0:
            x0, y0, z0, vx0, vy0, vz0, gi0, w0 = self.unif_dist()

            self.ecloud.wspecies.addparticles(x=x0, y=y0, z=z0, vx=vx0,
                                              vy=vy0, vz=vz0, gi=gi0,
                                              w=w0)
            print('injected %d electrons' % np.sum(self.ecloud.wspecies.getw()))
            print('injected %d MPs' % self.ecloud.wspecies.getn())

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

        self.b_pass = dict_init_dist['b_pass'] - 1

    def gaussian_time_prof(self):
        t = picmi.warp.top.time
        val = 0
        for i in range(0, self.n_bunches):
            val += (self.bunch_macro_particles * 1.
                    / np.sqrt(2 * np.pi * self.sigmat ** 2)
                    * np.exp(-(t - i * self.b_spac - self.t_offs) ** 2
                             / (2 * self.sigmat ** 2)) * picmi.warp.top.dt)
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
            vz = picmi.warp.clight * self.beam_beta
            self.beam.wspecies.addparticles(x=x, y=y, z=z, vx=vx,
                                            vy=vy, vz=vz,
                                            gi=1. / self.beam_gamma,
                                            w=self.bunch_w)

    def self_wrapped_custom_plot(self, l_force=0):
        self.custom_plot(self, l_force=l_force)

    def self_wrapped_custom_time_prof(self):
        return self.custom_time_prof(self)

    def defaultsfromdict(self, dic, kw):
        if kw is not None:
            for name, defvalue in dic.items():
                if name not in self.__dict__:
                    self.__dict__[name] = kw.get(name, defvalue)
                if name in kw: del kw[name]
            if kw:
                raise TypeError("""Keyword argument 
                                '%s' is out of place""" % list(kw)[0])

    def dump(self, filename):
        #self.solver.solver.laser_func = None
        if hasattr(self, 'laser_func'):
            del self.solver.em3dfft_args['laser_func']
            self.laser_func = None
            self.solver.solver.laser_func = None
        self.text_trap = None
        self.original = None
        self.custom_plot = None
        self.custom_time_prof = None
        warpdump(filename)

import numpy as np
import os
from h5py_manager import dict_of_arrays_and_scalar_from_h5, dict_to_h5, dict_to_h5_serial
from warp import *
from warp import picmi

class Saver:

    def __init__(self, flag_output, flag_checkpointing, tot_nsteps, n_bunches,
                 nbins, solver, output_filename= None, temps_filename = None, probe_filename = 'probe.h5'):
        self.flag_checkpointing = flag_checkpointing
        self.flag_output = flag_output
        self.temps_filename = temps_filename
        self.tot_nsteps = tot_nsteps
        self.n_bunches = n_bunches
        self.nbins = nbins
        self.output_filename = output_filename
        self.solver = solver
        self.probe_filename = probe_filename
        if self.flag_output:
            if not self.flag_checkpointing or not os.path.exists(self.temps_filename):
                self.init_empty_outputs()
            else:
                self.restore_outputs_from_file()

    def init_empty_outputs(self):
        self.numelecs = np.zeros(self.tot_nsteps)
        self.numelecs_tot = np.zeros(self.tot_nsteps)
        self.N_mp = np.zeros(self.tot_nsteps)
        if self.n_bunches is not None:
            self.xhist = np.zeros((self.n_bunches,self.nbins))
        self.bins = np.zeros(self.nbins)

    def restore_outputs_from_file(self):
        dict_init_dist = dict_of_arrays_and_scalar_from_h5(self.temps_filename)
        if self.flag_output:
            self.numelecs = dict_init_dist['numelecs']
            self.N_mp = dict_init_dist['N_mp']
            self.numelecs_tot = dict_init_dist['numelecs_tot']
            self.xhist = dict_init_dist['xhist']
            self.bins = dict_init_dist['bins']
            self.b_pass = dict_init_dist['b_pass']  
    
    def save_checkpoint(self, b_pass, elecbw):
        dict_out_temp = {}
        print('Saving a checkpoint!')
        dict_out_temp['x_mp'] = elecbw.getx()
        dict_out_temp['y_mp'] = elecbw.gety()
        dict_out_temp['z_mp'] = elecbw.getz()
        dict_out_temp['vx_mp'] = elecbw.getvx()
        dict_out_temp['vy_mp'] = elecbw.getvy()
        dict_out_temp['vz_mp'] = elecbw.getvz()
        dict_out_temp['nel_mp'] = elecbw.getw()
        if self.flag_output:
            dict_out_temp['numelecs'] = self.numelecs
            dict_out_temp['numelecs_tot'] = self.numelecs_tot
            dict_out_temp['N_mp'] = self.N_mp
            dict_out_temp['xhist'] = self.xhist
            dict_out_temp['bins'] = self.bins

        dict_out_temp['b_pass'] = b_pass 
        dict_out_temp['ecloud_density'] = elecbw.get_density()
    
        dict_to_h5(dict_out_temp, self.temps_filename)

    def update_outputs(self, ew, nz, n_step):
        elecb_w = ew.getw()
        elecs_density = ew.get_density(l_dividebyvolume=0)[:,:,int(nz/2.)]
        elecs_density_tot = ew.get_density(l_dividebyvolume=0)[:,:,:]
        # resize outputs if needed, this could happen in the event
        # of a restart with a different simulation length
        if self.tot_nsteps > len(self.numelecs):
            N_add = self.tot_nsteps - len(self.numelecs)
            self.numelecs = np.pad(self.numelecs, (0,N_add), 'constant')
            self.numelecs_tot = np.pad(self.numelecs_tot, (0,N_add), 'constant')
            self.N_mp = np.pad(self.N_mp, (0,N_add), 'constant')
        self.numelecs[n_step] = np.sum(elecs_density)
        self.numelecs_tot[n_step] = np.sum(elecs_density_tot)
        self.N_mp[n_step] = len(elecb_w)

    def dump_outputs(self, xmin, xmax, elecbw, b_pass):
        dict_out = {}
        dict_out['numelecs'] = self.numelecs
        dict_out['numelecs_tot'] = self.numelecs_tot
        dict_out['N_mp'] = self.N_mp
        # Compute the x-position histogram
        if self.n_bunches is not None:
            (self.xhist[b_pass-1], self.bins) = np.histogram(elecbw.getx(), 
                                                     range = (xmin,xmax), 
                                                     bins = self.nbins, 
                                                     weights = elecbw.getw(), 
                                                     density = False)
        dict_out['bins'] = self.bins
        dict_out['xhist'] = self.xhist
        dict_to_h5(dict_out, self.output_filename)

    def dump_em_fields(em, folder, filename):
        if not os.path.exists(folder+'/'+str(picmi.warp.me)):
            os.makedirs(folder+'/'+str(picmi.warp.me))
        dict_out = {}
        dict_out['ex'] = em.getexg(guards=1)
        dict_out['ey'] = em.geteyg(guards=1)
        dict_out['ez'] = em.getezg(guards=1)
        dict_out['bx'] = em.getbxg(guards=1)
        dict_out['by'] = em.getbyg(guards=1)
        dict_out['bz'] = em.getbzg(guards=1)
        dict_to_h5_serial(dict_out, folder+'/'+str(picmi.warp.me)+'/'+filename) 

    def init_field_probes(self, Nprobes, tot_nsteps, field_probes_dump_stride):
        self.Nprobes = Nprobes
        self.e_x_vec = np.zeros((Nprobes, tot_nsteps))
        self.e_y_vec = np.zeros((Nprobes, tot_nsteps))
        self.e_z_vec = np.zeros((Nprobes, tot_nsteps))
        self.b_x_vec = np.zeros((Nprobes, tot_nsteps))
        self.b_y_vec = np.zeros((Nprobes, tot_nsteps))
        self.b_z_vec = np.zeros((Nprobes, tot_nsteps))
        self.t_probes = np.zeros(tot_nsteps)
        self.field_probes_dump_stride = field_probes_dump_stride

    def field_probe(self, probe_i, pp):
        pw = picmi.warp
        em = self.solver.solver
        ex = em.gatherex()
        ey = em.gatherey()
        ez = em.gatherez()
        bx = em.gatherbx()
        by = em.gatherby()
        bz = em.gatherbz()
        if me ==0:
            self.e_x_vec[probe_i, pw.top.it-3] = ex[pp[0],pp[1],pp[2]]
            self.e_y_vec[probe_i, pw.top.it-3] = ey[pp[0],pp[1],pp[2]]
            self.e_z_vec[probe_i, pw.top.it-3] = ez[pp[0],pp[1],pp[2]]
            self.b_x_vec[probe_i, pw.top.it-3] = bx[pp[0],pp[1],pp[2]]
            self.b_y_vec[probe_i, pw.top.it-3] = by[pp[0],pp[1],pp[2]]
            self.b_z_vec[probe_i, pw.top.it-3] = bz[pp[0],pp[1],pp[2]]
            #Save if specified by the user and if all the probes have been processed
            stride = self.field_probes_dump_stride
            if probe_i == self.Nprobes-1:
                self.t_probes[pw.top.it-3]= pw.top.time
                if pw.top.it%stride == 0:
                    dict_out = {}
                    dict_out['ex'] = self.e_x_vec
                    dict_out['ey'] = self.e_y_vec
                    dict_out['ez'] = self.e_z_vec
                    dict_out['bx'] = self.b_x_vec
                    dict_out['by'] = self.b_y_vec
                    dict_out['bz'] = self.b_z_vec
                    dict_out['t_probes'] = self.t_probes
                    dict_to_h5_serial(dict_out, self.probe_filename)

    def extend_probe_vectors(self, n_add_steps):
        add_zeros_fields = np.zeros((self.Nprobes, n_add_steps))
        add_zeros_t = np.zeros(n_add_steps)
        self.e_x_vec = np.concatenate((self.e_x_vec, add_zeros_fields), axis = 1) 
        self.e_y_vec = np.concatenate((self.e_y_vec, add_zeros_fields), axis = 1) 
        self.e_z_vec = np.concatenate((self.e_z_vec, add_zeros_fields), axis = 1) 
        self.b_x_vec = np.concatenate((self.b_x_vec, add_zeros_fields), axis = 1) 
        self.b_y_vec = np.concatenate((self.b_y_vec, add_zeros_fields), axis = 1) 
        self.b_z_vec = np.concatenate((self.b_z_vec, add_zeros_fields), axis = 1)
        self.b_z_vec = np.concatenate((self.b_z_vec, add_zeros_fields), axis = 1)
        self.t_probes = np.concatenate((self.t_probes, add_zeros_t))


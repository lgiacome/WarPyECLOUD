import numpy as np
import os
from h5py_manager import dict_of_arrays_and_scalar_from_h5, dict_to_h5, dict_to_h5_serial
from warp import picmi, AppendableArray
from warp_parallel import parallelsum, npes

class Saver:

    def __init__(self, flag_output, flag_checkpointing,
                 nbins, solver, sec, output_filename= None, temps_filename = None, probe_filename = 'probe.h5', tot_nsteps = 1, n_bunches = 1):
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
            self.init_empty_outputs(tot_nsteps, n_bunches)
            if os.path.exists(self.temps_filename):
                self.restore_outputs_from_file()
        self.sec = sec

    def init_empty_outputs(self, tot_nsteps = 1, n_bunches = 1):
        self.numelecs = AppendableArray(initlen = tot_nsteps, typecode='d')
        self.numelecs_tot = AppendableArray(initlen = tot_nsteps, typecode='d')
        self.N_mp = AppendableArray(initlen = tot_nsteps, typecode='d')
        if self.n_bunches is not None:
            self.xhist = AppendableArray(initlen = n_bunches, typecode = 'd', unitshape = (1,self.nbins))
        self.bins = np.zeros(self.nbins)
        self.tt = AppendableArray(initlen = tot_nsteps, typecode='d')
        self.costhav = AppendableArray(initlen = tot_nsteps, typecode='d')
        self.ek0av = AppendableArray(initlen = tot_nsteps, typecode='d')
        self.htime = AppendableArray(initlen = tot_nsteps, typecode='d')

    def restore_outputs_from_file(self):
        dict_init_dist = dict_of_arrays_and_scalar_from_h5(self.temps_filename)
        if self.flag_output:
            self.numelecs.append(dict_init_dist['numelecs'])
            self.N_mp.append(dict_init_dist['N_mp'])
            self.numelecs_tot.append(dict_init_dist['numelecs_tot'])
            for xhist in dict_init_dist['xhist']:
                self.xhist.append(xhist)
            self.bins = dict_init_dist['bins']
            self.tt.append(dict_init_dist['tt']) 
            
            self.costhav = dict_out['costhav']
            self.ek0av = dict_out['ek0av']
            self.htime = dict_out['t_imp']
            
    
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
        self.numelecs.append(np.sum(elecs_density))
        self.numelecs_tot.append(np.sum(elecs_density_tot))
        self.N_mp.append(ew.getn())
        self.tt.append(picmi.warp.top.time)
        if len(self.sec.costhav) > 0:
            costhav = parallelsum(self.sec.costhav[-1])/npes
            ek0av = paralellsum(self.sec.ek0av[-1])/npes
            htime = paralellsum(self.sec.htime[-1])/npes
            self.costhav.append(costhav)
            self.ek0av.append(ek0av)
            self.htime.append(htime)

    def dump_outputs(self, xmin, xmax, elecbw, b_pass):
        dict_out = {}
        dict_out['numelecs'] = self.numelecs
        dict_out['numelecs_tot'] = self.numelecs_tot
        dict_out['N_mp'] = self.N_mp
        # Compute the x-position histogram
        if self.n_bunches is not None:
            (xhist, self.bins) = np.histogram(elecbw.getx(), 
                                                     range = (xmin,xmax), 
                                                     bins = self.nbins, 
                                                     weights = elecbw.getw(), 
                                                     density = False)
        self.xhist.append(xhist)
        
        dict_out['bins'] = self.bins
        dict_out['xhist'] = self.xhist
        dict_out['tt'] = self.tt
        dict_out['costhav'] = self.costhav
        dict_out['ek0av'] = self.ek0av
        dict_out['t_imp'] = self.htime
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
        dict_to_h5(dict_out, folder+'/'+str(picmi.warp.me)+'/'+filename) 

    def init_field_probes(self, Nprobes, tot_nsteps, field_probes_dump_stride):
        self.Nprobes = Nprobes
        self.e_x_vec = AppendableArray(typecode = 'd', unitshape = (Nprobes, 1))
        self.e_y_vec = AppendableArray(typecode = 'd', unitshape = (Nprobes, 1))
        self.e_z_vec = AppendableArray(typecode = 'd', unitshape = (Nprobes, 1))
        self.b_x_vec = AppendableArray(typecode = 'd', unitshape = (Nprobes, 1))
        self.b_y_vec = AppendableArray(typecode = 'd', unitshape = (Nprobes, 1))
        self.b_z_vec = AppendableArray(typecode = 'd', unitshape = (Nprobes, 1))
        self.t_probes = AppendableArray(typecode = 'd')
        self.field_probes_dump_stride = field_probes_dump_stride

    def update_field_probes(self, pp):
        pw = picmi.warp
        em = self.solver.solver
        ex = em.gatherex()
        ey = em.gatherey()
        ez = em.gatherez()
        bx = em.gatherbx()
        by = em.gatherby()
        bz = em.gatherbz()
        
        self.e_x_vec.append(ex[pp[...,0],pp[...,1],pp[...,2]])
        self.e_y_vec.append(ey[pp[...,0],pp[...,1],pp[...,2]])
        self.e_z_vec.append(ez[pp[...,0],pp[...,1],pp[...,2]])
        self.b_x_vec.append(bx[pp[...,0],pp[...,1],pp[...,2]])
        self.b_y_vec.append(by[pp[...,0],pp[...,1],pp[...,2]])
        self.b_z_vec.append(bz[pp[...,0],pp[...,1],pp[...,2]])
       
        self.t_probes.append(pw.top.time)

        #Save if specified by the user and if all the probes have been processed
        stride = self.field_probes_dump_stride 
        if pw.top.it%stride == 0:
            self.dump_probes()

    def dump_probes(self):
        pw = picmi.warp
        dict_out = {}
        dict_out['ex'] = self.e_x_vec
        dict_out['ey'] = self.e_y_vec
        dict_out['ez'] = self.e_z_vec
        dict_out['bx'] = self.b_x_vec
        dict_out['by'] = self.b_y_vec
        dict_out['bz'] = self.b_z_vec
        dict_out['t_probes'] = self.t_probes
        dict_to_h5(dict_out, self.probe_filename)


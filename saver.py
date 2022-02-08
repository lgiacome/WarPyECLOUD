import numpy as np
import os
from h5py_manager import dict_of_arrays_and_scalar_from_h5, dict_to_h5, dict_to_h5_serial
from warp import AppendableArray
from warp import picmi
from time import sleep


class Saver:
    """
    WarPyECLOUD saver class
    - flag_output: save outputs to file
    - nbins: number of bins for the histograms
    - solver: Warp solver object
    - temps_filename: name of checkpoint file
    - probe_filename: name of field probe file
    - tot_nsteps: total number of steps of the simulation
    - n_nunches: number of the simulated bunches
    - flag_save_ek_impacts: if True save the kinetic energy deposited on the conductors by the electrons
    """

    def __init__(self, flag_output, nbins, solver, sec, output_filename=None, temps_filename=None,
                 probe_filename='probe.h5', tot_nsteps=1, n_bunches=1, flag_save_ek_impacts=False):

        self.flag_output = flag_output
        self.temps_filename = temps_filename
        self.tot_nsteps = tot_nsteps
        self.n_bunches = n_bunches
        self.nbins = nbins
        self.output_filename = output_filename
        self.solver = solver
        self.sec = sec
        self.flag_save_ek_impacts = flag_save_ek_impacts
        self.probe_filename = probe_filename
        if self.flag_output:
            self.init_empty_outputs(tot_nsteps, n_bunches)
            if os.path.exists(self.temps_filename):
                self.restore_outputs_from_file()

    @staticmethod
    def save_h5_safe(dict_out, filename, serial=False):
        """
        Save a dictionary into a h5 file in a safe way, which keep retrying to save until it works
        - dict_save: dictionary to save
        - filename: name of the h5 file
        - serial: if True use MPI support
        """
        count = 0
        if picmi.warp.me == 0 and os.path.exists(filename):
            os.remove(filename)
        while 1:
            try:
                if not serial:
                    dict_to_h5(dict_out, filename)
                else:
                    dict_to_h5_serial(dict_out, filename)
                break
            except:
                count += 1
                print('Failed saving ' + filename + ' %d times. Retrying in 5 seconds' % count)
                sleep(5)
                pass

    def init_empty_outputs(self, tot_nsteps=1, n_bunches=1):
        """
        Save a dictionary into a h5 file in a safe way, which keep retrying to save until it works
        - dict_save: dictionary to save
        - filename: name of the h5 file
        - serial: if True use MPI support
        """
        self.numelecs = AppendableArray(initlen=tot_nsteps, typecode='d')
        self.numelecs_tot = AppendableArray(initlen=tot_nsteps, typecode='d')
        self.numelecs_body_cav = AppendableArray(initlen=tot_nsteps, typecode='d')
        self.numpro = AppendableArray(initlen=tot_nsteps, typecode='d')
        self.N_mp = AppendableArray(initlen=tot_nsteps, typecode='d')
        if self.n_bunches is not None:
            self.xhist = AppendableArray(initlen=n_bunches, typecode='d', unitshape=(1, self.nbins))
        self.bins = np.zeros(self.nbins)
        self.tt = AppendableArray(initlen=tot_nsteps, typecode='d')

        self.ex_applied = AppendableArray(initlen=tot_nsteps, typecode='d')

    def restore_outputs_from_file(self):
        """
        When restarting from a checkpoint restore the output arrays
        """
        dict_init_dist = dict_of_arrays_and_scalar_from_h5(self.temps_filename)
        if self.flag_output:
            self.numelecs.append(dict_init_dist['numelecs'])
            self.numpro.append(dict_init_dist['numpro'])
            self.N_mp.append(dict_init_dist['N_mp'])
            self.numelecs_tot.append(dict_init_dist['numelecs_tot'])
            for xhist in dict_init_dist['xhist']:
                self.xhist.append(xhist)
            self.bins = dict_init_dist['bins']
            self.tt.append(dict_init_dist['tt'])
            if self.flag_save_ek_impacts:
                self.sec.ek0av.append(dict_init_dist['ek0av'])
                self.sec.power_diff.append(dict_init_dist['power_diff'])
                self.sec.htime.append(dict_init_dist['t_imp'])

    def save_checkpoint(self, b_pass, elecbw):
        """
        Save a checkpoint
        - b_pass: bunch passage number
        - elecbw: electrons species object
        """
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
            dict_out_temp['tt'] = self.numelecs
            dict_out_temp['numpro'] = self.numpro
            dict_out_temp['numelecs_tot'] = self.numelecs_tot
            dict_out_temp['N_mp'] = self.N_mp
            dict_out_temp['xhist'] = self.xhist
            dict_out_temp['bins'] = self.bins
            if self.flag_save_ek_impacts:
                dict_out_temp['ek0av'] = self.sec.ek0av
                dict_out_temp['power_diff'] = self.sec.power_diff
                dict_out_temp['t_imp'] = self.sec.htime

        dict_out_temp['b_pass'] = b_pass
        dict_out_temp['ecloud_density'] = elecbw.get_density()
        if picmi.warp.me == 0:
            self.save_h5_safe(dict_out_temp, self.temps_filename, serial=True)

    def update_outputs(self, ew, bw, nz, chamber):
        """
        Update the output arrays
        - ew: electrons species object
        - bw: beam species object
        - nz: number of cells in z-direction
        - chamber: chamber object
        """
        elecs_density = ew.get_density(l_dividebyvolume=0)[:, :, int(nz / 2.)]
        pro_density_tot = bw.get_density(l_dividebyvolume=0)[:, :, :]
        self.numelecs.append(np.sum(elecs_density))
        self.numpro.append(np.sum(pro_density_tot))
        self.numelecs_tot.append(np.sum(np.sum(ew.getw())))
        l_main_z = 354e-3
        z_mp = ew.getz()
        flag_in_body = np.logical_and(z_mp <= l_main_z / 2, z_mp > - l_main_z / 2)
        self.numelecs_body_cav.append(np.sum(np.sum(ew.getw()[flag_in_body])))
        self.N_mp.append(ew.getn())
        self.tt.append(picmi.warp.top.time)

        (ex, ey, ez, bx, by, bz) = picmi.warp.getappliedfieldsongrid(
            nx=10, ny=10, nz=10,
            xmin=chamber.xmin, xmax=chamber.xmax,
            ymin=chamber.ymin, ymax=chamber.ymax,
            zmin=chamber.zmin, zmax=chamber.zmax)

        self.ex_applied.append(ex[5, 5, 5])

    def dump_outputs(self, xmin, xmax, elecbw):
        """
        Dump the output file
        - xmin: min of the domain in x
        - xmax: max of the domain in x
        - elecbw: electrons species object
        """
        dict_out = {'numelecs': self.numelecs, 'numpro': self.numpro, 'numelecs_tot': self.numelecs_tot,
                    'numelecs_body_cav': self.numelecs_body_cav, 'N_mp': self.N_mp}
        # Compute the x-position histogram
        if self.n_bunches is not None:
            (xhist, self.bins) = np.histogram(elecbw.getx(), range=(xmin, xmax), bins=self.nbins, weights=elecbw.getw(),
                                              density=False)
            self.xhist.append(xhist)

        dict_out['bins'] = self.bins
        dict_out['xhist'] = self.xhist
        dict_out['tt'] = self.tt
        dict_out['ex_applied'] = self.ex_applied
        if self.flag_save_ek_impacts:
            dict_out['ek0av'] = self.sec.ek0av
            dict_out['power_diff'] = self.sec.power_diff
            dict_out['t_imp'] = self.sec.htime
        if picmi.warp.me == 0:
            self.save_h5_safe(dict_out, self.output_filename, serial=True)

    def dump_em_fields(self, em, folder, filename):
        """
        Dump the super-imposed electromagnetic fields to an h5 file
        - em: electromagnetic solver object
        - folder: folder where to save the file
        - filename: name of the file
        """
        if not os.path.exists(folder + '/' + str(picmi.warp.me)):
            os.makedirs(folder + '/' + str(picmi.warp.me))
        dict_out = {'ex': em.getexg(guards=1), 'ey': em.geteyg(guards=1), 'ez': em.getezg(guards=1),
                    'bx': em.getbxg(guards=1), 'by': em.getbyg(guards=1), 'bz': em.getbzg(guards=1)}
        filename_tot = folder + '/' + str(picmi.warp.me) + '/' + filename
        self.save_h5_safe(dict_out, filename_tot)

    def init_field_probes(self, n_probes, field_probes_dump_stride):
        """
        Initialize the arrays of the field probes
        - n_probes: number of probes
        - field_probes_dump_stride: stride between two probe saves
        """
        self.n_probes = n_probes
        self.e_x_vec = AppendableArray(typecode='d', unitshape=(n_probes, 1))
        self.e_y_vec = AppendableArray(typecode='d', unitshape=(n_probes, 1))
        self.e_z_vec = AppendableArray(typecode='d', unitshape=(n_probes, 1))
        self.b_x_vec = AppendableArray(typecode='d', unitshape=(n_probes, 1))
        self.b_y_vec = AppendableArray(typecode='d', unitshape=(n_probes, 1))
        self.b_z_vec = AppendableArray(typecode='d', unitshape=(n_probes, 1))
        self.t_probes = AppendableArray(typecode='d')
        self.field_probes_dump_stride = field_probes_dump_stride

    def update_field_probes(self, pp):
        """
        Update the arrays of the field probes
        - pp: probes locations
        """
        pw = picmi.warp
        em = self.solver.solver
        ex = em.gatherex()
        ey = em.gatherey()
        ez = em.gatherez()
        bx = em.gatherbx()
        by = em.gatherby()
        bz = em.gatherbz()

        if picmi.warp.me == 0:
            self.e_x_vec.append(ex[pp[..., 0], pp[..., 1], pp[..., 2]])
            self.e_y_vec.append(ey[pp[..., 0], pp[..., 1], pp[..., 2]])
            self.e_z_vec.append(ez[pp[..., 0], pp[..., 1], pp[..., 2]])
            self.b_x_vec.append(bx[pp[..., 0], pp[..., 1], pp[..., 2]])
            self.b_y_vec.append(by[pp[..., 0], pp[..., 1], pp[..., 2]])
            self.b_z_vec.append(bz[pp[..., 0], pp[..., 1], pp[..., 2]])

            self.t_probes.append(pw.top.time)

            # Save if specified by the user and if all the probes have been processed
            stride = self.field_probes_dump_stride
            if pw.top.it % stride == 0:
                self.dump_probes()

    def dump_probes(self):
        """
        Dump the probe files
        """
        dict_out = {'ex': self.e_x_vec, 'ey': self.e_y_vec, 'ez': self.e_z_vec, 'bx': self.b_x_vec, 'by': self.b_y_vec,
                    'bz': self.b_z_vec, 't_probes': self.t_probes}
        self.save_h5_safe(dict_out, self.probe_filename, serial=True)

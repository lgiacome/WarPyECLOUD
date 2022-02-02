from warp import picmi
import numpy as np
from h5py_manager import dict_of_arrays_and_scalar_from_h5_serial


class Dipole:

    def __init__(self, zs_dipo, ze_dipo, by):
        picmi.warp.addnewdipo(zs=zs_dipo, ze=ze_dipo, by=by)


def interp(fx_3d):
    fxx = 0.125 * (fx_3d[0:-1, 0:-1, 0:-1]
                   + fx_3d[0:-1:, 0:-1, 1:]
                   + fx_3d[0:-1, 1:, 0:-1]
                   + fx_3d[0:-1, 1:, 1:]
                   + fx_3d[1:, 1:, 0:-1]
                   + fx_3d[1:, 1:, 1:]
                   + fx_3d[1:, 0:-1, 1:]
                   + fx_3d[1:, 0:-1, 0:-1])
    return fxx


class CrabFields:

    def __init__(self, max_rescale=1., efield_path='efield.txt',
                 hfield_path='hfield.txt', chamber=None, t_offs=None):
        get_data = picmi.getdatafromtextfile
        self.maxE = max_rescale
        self.chamber = chamber

        x, y, z, re_ex, re_ey, re_ez, im_ex, im_ey, im_ez = get_data(efield_path, nskip=1, dims=[9, None])
        _, _, _, re_hx, re_hy, re_hz, im_hx, im_hy, im_hz = get_data(hfield_path, nskip=1, dims=[9, None])

        re_bx = re_hx * picmi.mu0
        re_by = re_hy * picmi.mu0
        re_bz = re_hz * picmi.mu0
        im_bx = im_hx * picmi.mu0
        im_by = im_hy * picmi.mu0
        im_bz = im_hz * picmi.mu0

        # Interpolate them at cell centers (as prescribed by Warp doc)
        self.d = abs(x[1] - x[0])
        # Number of mesh cells
        self.NNx = int(round(2 * np.max(x) / self.d))
        self.NNy = int(round(2 * np.max(y) / self.d))
        self.NNz = int(round(2 * np.max(z) / self.d))
        # Number of mesh vertices
        self.nnx = self.NNx + 1
        self.nny = self.NNy + 1
        self.nnz = self.NNz + 1

        re_ex_3d = re_ex.reshape(self.nnz, self.nny, self.nnx).transpose(2, 1, 0)
        re_ey_3d = re_ey.reshape(self.nnz, self.nny, self.nnx).transpose(2, 1, 0)
        re_ez_3d = re_ez.reshape(self.nnz, self.nny, self.nnx).transpose(2, 1, 0)
        re_bx_3d = re_bx.reshape(self.nnz, self.nny, self.nnx).transpose(2, 1, 0)
        re_by_3d = re_by.reshape(self.nnz, self.nny, self.nnx).transpose(2, 1, 0)
        re_bz_3d = re_bz.reshape(self.nnz, self.nny, self.nnx).transpose(2, 1, 0)
        im_ex_3d = im_ex.reshape(self.nnz, self.nny, self.nnx).transpose(2, 1, 0)
        im_ey_3d = im_ey.reshape(self.nnz, self.nny, self.nnx).transpose(2, 1, 0)
        im_ez_3d = im_ez.reshape(self.nnz, self.nny, self.nnx).transpose(2, 1, 0)
        im_bx_3d = im_bx.reshape(self.nnz, self.nny, self.nnx).transpose(2, 1, 0)
        im_by_3d = im_by.reshape(self.nnz, self.nny, self.nnx).transpose(2, 1, 0)
        im_bz_3d = im_bz.reshape(self.nnz, self.nny, self.nnx).transpose(2, 1, 0)
        x3d = x.reshape(self.nnz, self.nny, self.nnx).transpose(2, 1, 0)
        y3d = y.reshape(self.nnz, self.nny, self.nnx).transpose(2, 1, 0)
        z3d = z.reshape(self.nnz, self.nny, self.nnx).transpose(2, 1, 0)
        # Rescale the fields at convenience
        self.kk = self.maxE / np.max(abs(re_ey_3d[int((self.nnx + 1) / 2), int((self.nny + 1) / 2), :]))

        re_ex_3d *= self.kk
        re_ey_3d *= self.kk
        re_ez_3d *= self.kk
        im_ex_3d *= self.kk
        im_ey_3d *= self.kk
        im_ez_3d *= self.kk
        re_bx_3d *= self.kk
        re_by_3d *= self.kk
        re_bz_3d *= self.kk
        im_bx_3d *= self.kk
        im_by_3d *= self.kk
        im_bz_3d *= self.kk

        self.ReExx = interp(re_ex_3d)
        self.ReEyy = interp(re_ey_3d)
        self.ReEzz = interp(re_ez_3d)
        self.ImExx = interp(im_ex_3d)
        self.ImEyy = interp(im_ey_3d)
        self.ImEzz = interp(im_ez_3d)

        self.ReBxx = interp(re_bx_3d)
        self.ReByy = interp(re_by_3d)
        self.ReBzz = interp(re_bz_3d)
        self.ImBxx = interp(im_bx_3d)
        self.ImByy = interp(im_by_3d)
        self.ImBzz = interp(im_bz_3d)

        self.xx = interp(x3d)
        self.yy = interp(y3d)
        self.zz = interp(z3d)

        # Lattice spatial parameters
        self.zs = np.min(z3d) - self.d / 2.
        self.ze = np.max(z3d) + self.d / 2.
        self.xs = np.min(x3d) - self.d / 2.
        self.ys = np.min(y3d) - self.d / 2.
        # Lattice temporal parameters
        self.Tf = 25e-9
        self.freq = 400 * 1e6
        self.Nt = 1000
        self.phase_disp = np.pi / 2
        delay = (self.chamber.lower_bound[2]) / picmi.clight - t_offs

        time_array = np.linspace(0., self.Tf, self.Nt)

        data_array_sin = -np.sin((time_array - delay) * self.freq * 2 * np.pi + self.phase_disp)
        data_array_cos = np.cos((time_array - delay) * self.freq * 2 * np.pi + self.phase_disp)

        # Create overlapped lattice elements to have E and B in the same region
        picmi.warp.addnewegrd(self.zs, self.ze, dx=self.d, dy=self.d, xs=self.xs, ys=self.ys, time=time_array,
                              data=data_array_cos, ex=self.ReExx, ey=self.ReEyy, ez=self.ReEzz)

        picmi.warp.addnewegrd(self.zs, self.ze, dx=self.d, dy=self.d, xs=self.xs, ys=self.ys, time=time_array,
                              data=data_array_sin, ex=self.ImExx, ey=self.ImEyy, ez=self.ImEzz)

        picmi.warp.addnewbgrd(self.zs, self.ze, dx=self.d, dy=self.d, xs=self.xs, ys=self.ys, time=time_array,
                              data=data_array_cos, bx=self.ReBxx, by=self.ReByy, bz=self.ReBzz)

        picmi.warp.addnewbgrd(self.zs, self.ze, dx=self.d, dy=self.d, xs=self.xs, ys=self.ys, time=time_array,
                              data=data_array_sin, bx=self.ImBxx, by=self.ImByy, bz=self.ImBzz)


def func_sin(t):
    freq = 400 * 1e6
    phase_disp = 0
    delay = 0
    return -np.sin((t - delay) * freq * 2 * np.pi + phase_disp)


def func_cos(t):
    freq = 400 * 1e6
    phase_disp = 0
    delay = 0
    return np.cos((t - delay) * freq * 2 * np.pi + phase_disp)


class CrabFieldsH5:

    def __init__(self, max_rescale=None, fields_path='fields.h5'):
        dict_h5 = dict_of_arrays_and_scalar_from_h5_serial(fields_path)
        self.init_self_from_dict(dict_h5)
        self.maxE = max_rescale
        nnx, nny, nnz = np.shape(self.ReEyy)
        if self.maxE is not None:
            kk = self.maxE / np.max(abs(self.ReEyy[int((nnx + 1) / 2), int((nny + 1) / 2), :]))

            self.ReExx *= kk
            self.ReEyy *= kk
            self.ReEzz *= kk
            self.ImExx *= kk
            self.ImEyy *= kk
            self.ImEzz *= kk
            self.ReBxx *= kk
            self.ReByy *= kk
            self.ReBzz *= kk
            self.ImBxx *= kk
            self.ImByy *= kk
            self.ImBzz *= kk

        # Create overlapped lattice elements to have E and B in the same region
        picmi.warp.addnewegrd(self.zs, self.ze, dx=self.d, dy=self.d, xs=self.xs, ys=self.ys, func=func_cos,
                              ex=self.ReExx, ey=self.ReEyy, ez=self.ReEzz)

        picmi.warp.addnewegrd(self.zs, self.ze, dx=self.d, dy=self.d, xs=self.xs, ys=self.ys, func=func_sin,
                              ex=self.ImExx, ey=self.ImEyy, ez=self.ImEzz)

        picmi.warp.addnewbgrd(self.zs, self.ze, dx=self.d, dy=self.d, xs=self.xs, ys=self.ys, func=func_cos,
                              bx=self.ReBxx, by=self.ReByy, bz=self.ReBzz)

        picmi.warp.addnewbgrd(self.zs, self.ze, dx=self.d, dy=self.d, xs=self.xs, ys=self.ys, func=func_sin,
                              bx=self.ImBxx, by=self.ImByy, bz=self.ImBzz)

    def init_self_from_dict(self, dic):
        for name in dic.keys():
            self.__dict__[name] = dic[name]

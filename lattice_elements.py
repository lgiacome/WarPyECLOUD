from warp import picmi
import numpy as np

class Dipole:
    
    def __init__(self, zs_dipo, ze_dipo, By):
        picmi.warp.addnewdipo(zs = zs_dipo, ze = ze_dipo, by = By)

class CrabFields:

    def __init__(self, min_z, max_rescale = 1., efield_path = 'efield.txt',
                 hfield_path = 'hfield.txt', chamber = None, t_offs = None):
        get_data = picmi.getdatafromtextfile
        self.maxE = max_rescale

        [x,y,z,ReEx,ReEy,ReEz,ImEx,ImEy,ImEz] = get_data(efield_path, nskip=1, dims=[9,None])
        [_,_,_,ReHx,ReHy,ReHz,ImHx,ImHy,ImHz] = get_data(hfield_path, nskip=1, dims=[9,None])

        ReBx = ReHx*picmi.mu0
        ReBy = ReHy*picmi.mu0
        ReBz = ReHz*picmi.mu0
        ImBx = ImHx*picmi.mu0
        ImBy = ImHy*picmi.mu0
        ImBz = ImHz*picmi.mu0

        # Interpolate them at cell centers (as prescribed by Warp doc)
        self.d = abs(x[1] - x[0])
        # Number of mesh cells
        self.NNx = int(round(2*np.max(x)/self.d))
        self.NNy = int(round(2*np.max(y)/self.d))
        self.NNz = int(round(2*np.max(z)/self.d))
        # Number of mesh vertices
        self.nnx = self.NNx + 1
        self.nny = self.NNy + 1
        self.nnz = self.NNz + 1

        ReEx3d = ReEx.reshape(self.nnz, self.nny, self.nnx).transpose(2, 1, 0)
        ReEy3d = ReEy.reshape(self.nnz, self.nny, self.nnx).transpose(2, 1, 0)
        ReEz3d = ReEz.reshape(self.nnz, self.nny, self.nnx).transpose(2, 1, 0)
        ReBx3d = ReBx.reshape(self.nnz, self.nny, self.nnx).transpose(2, 1, 0)
        ReBy3d = ReBy.reshape(self.nnz, self.nny, self.nnx).transpose(2, 1, 0)
        ReBz3d = ReBz.reshape(self.nnz, self.nny, self.nnx).transpose(2, 1, 0)
        ImEx3d = ImEx.reshape(self.nnz, self.nny, self.nnx).transpose(2, 1, 0)
        ImEy3d = ImEy.reshape(self.nnz, self.nny, self.nnx).transpose(2, 1, 0)
        ImEz3d = ImEz.reshape(self.nnz, self.nny, self.nnx).transpose(2, 1, 0)
        ImBx3d = ImBx.reshape(self.nnz, self.nny, self.nnx).transpose(2, 1, 0)
        ImBy3d = ImBy.reshape(self.nnz, self.nny, self.nnx).transpose(2, 1, 0)
        ImBz3d = ImBz.reshape(self.nnz, self.nny, self.nnx).transpose(2, 1, 0)
        x3d = x.reshape(self.nnz, self.nny, self.nnx).transpose(2, 1, 0)
        y3d = y.reshape(self.nnz, self.nny, self.nnx).transpose(2, 1, 0)
        z3d = z.reshape(self.nnz, self.nny, self.nnx).transpose(2, 1, 0)
        # Rescale the fields at convenience
        self.kk = self.maxE/np.max(abs(ReEy3d[int((self.nnx+1)/2),int((self.nny+1)/2),:]))

        ReEx3d *= self.kk
        ReEy3d *= self.kk
        ReEz3d *= self.kk
        ImEx3d *= self.kk
        ImEy3d *= self.kk
        ImEz3d *= self.kk
        ReBx3d *= self.kk
        ReBy3d *= self.kk
        ReBz3d *= self.kk
        ImBx3d *= self.kk
        ImBy3d *= self.kk
        ImBz3d *= self.kk


        self.ReExx = self.interp(ReEx3d)
        self.ReEyy = self.interp(ReEy3d)
        self.ReEzz = self.interp(ReEz3d)
        self.ImExx = self.interp(ImEx3d)
        self.ImEyy = self.interp(ImEy3d)
        self.ImEzz = self.interp(ImEz3d)

        self.ReBxx = self.interp(ReBx3d)
        self.ReByy = self.interp(ReBy3d)
        self.ReBzz = self.interp(ReBz3d)
        self.ImBxx = self.interp(ImBx3d)
        self.ImByy = self.interp(ImBy3d)
        self.ImBzz = self.interp(ImBz3d)

        self.xx = self.interp(x3d)
        self.yy = self.interp(y3d)
        self.zz = self.interp(z3d)

        # Lattice spatial parameters
        self.zs = np.min(z3d) - self.d/2.
        self.ze = np.max(z3d) + self.d/2.
        self.xs = np.min(x3d) - self.d/2.
        self.ys = np.min(y3d)-self.d/2.
        # Lattice temporal parameters
        self.Tf = 25e-9
        self.freq = 400*1e6
        self.Nt = 1000
        self.phase_disp=0
        delay = (chamber.zmin+1e-4)/picmi.clight + t_offs
        time_array = np.linspace(0., self.Tf, self.Nt)

        data_arraySinE = -np.sin((time_array-delay)*self.freq*2*np.pi+self.phase_disp)
        data_arrayCosE = np.cos((time_array-delay)*self.freq*2*np.pi+self.phase_disp)
        data_arraySinB = -np.sin((time_array-delay)*self.freq*2*np.pi+np.pi/2+self.phase_disp)
        data_arrayCosB = np.sin((time_array-delay)*self.freq*2*np.pi+np.pi/2+self.phase_disp)


        # Create overlapped lattice elements to have E and B in the same region
        iReE, ReEgrid = picmi.warp.addnewegrd(self.zs, self.ze,
                                          dx = self.d, dy = self.d,
                                          xs = self.xs, ys = self.ys,
                                          time = time_array,
                                          data = data_arrayCosE,
                                          ex = self.ReExx,
                                          ey = self.ReEyy,
                                          ez = self.ReEzz)
                                  
        iImE, ImEgrid = picmi.warp.addnewegrd(self.zs, self.ze,
                                          dx = self.d, dy = self.d,
                                          xs = self.xs, ys = self.ys,
                                          time = time_array,
                                          data = data_arraySinE,
                                          ex = self.ImExx,
                                          ey = self.ImEyy,
                                          ez = self.ImEzz)

        iReB, ReBgrid = picmi.warp.addnewbgrd(self.zs, self.ze,
                                          dx = self.d, dy = self.d,
                                          xs = self.xs, ys = self.ys,
                                          time = time_array,
                                          data = data_arrayCosB,
                                          bx = self.ReBxx,
                                          by = self.ReByy,
                                          bz = self.ReBzz)

        iImB, ImBgrid = picmi.warp.addnewbgrd(self.zs, self.ze,
                                          dx = self.d, dy = self.d,
                                          xs = self.xs, ys = self.ys,
                                          time = time_array,
                                          data = data_arraySinB,
                                          bx = self.ImBxx,
                                          by = self.ImByy,
                                          bz = self.ImBzz)
                                          
                                          
    def interp(self, Fx3d):
        Fxx = 0.125*(Fx3d[0:-1, 0:-1, 0:-1]
                            + Fx3d[0:-1:, 0:-1, 1:]
                            + Fx3d[0:-1, 1:, 0:-1]
                            + Fx3d[0:-1, 1:, 1:]
                            + Fx3d[1:, 1:, 0:-1]
                            + Fx3d[1:, 1:, 1:]
                            + Fx3d[1:, 0:-1, 1:]
                            + Fx3d[1:, 0:-1, 0:-1])
        return Fxx

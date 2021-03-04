import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy  as np
import h5py 

from scipy.constants import c as c_light

def dict_of_arrays_and_scalar_from_h5(filename):
    with h5py.File(filename, 'r') as fid:
        f_dict = {}
        for kk in list(fid.keys()):
            f_dict[kk] = np.array(fid[kk]).copy()
            if f_dict[kk].shape == ():
                f_dict[kk] = f_dict[kk].tolist()
        fid.close()
    return  f_dict

warp = dict_of_arrays_and_scalar_from_h5('cavity_triang_out.h5')
plt.semilogy(warp['tt'], warp['numelecs_tot'])
plt.xlabel('t    [ns]')
plt.ylabel('#Electrons in the crab cavity')

plt.show()

import h5py
import numpy as np
from mpi4py import MPI
from warp import *

def dict_to_h5(dict_save, filename, compression=None, compression_opts=None):
    rank = MPI.COMM_WORLD.rank  # The process ID (integer 0-3 for 4-process run)
    fid = h5py.File(filename, 'w', driver='mpio', comm=MPI.COMM_WORLD)
    for kk in list(dict_save.keys()): 
        fid.create_dataset(kk, data=dict_save[kk],
        compression=compression, compression_opts=compression_opts)
    fid.close()

def dict_to_h5_serial(dict_save, filename, compression=None, compression_opts=None):
    fid = h5py.File(filename, 'w')
    for kk in list(dict_save.keys()):
        fid.create_dataset(kk, data=dict_save[kk],
        compression=compression, compression_opts=compression_opts)
    fid.close()

def dict_of_arrays_and_scalar_from_h5_serial(filename):
    with h5py.File(filename, 'r') as fid:
        f_dict = {}
        for kk in list(fid.keys()):
            f_dict[kk] = np.array(fid[kk]).copy()
            if f_dict[kk].shape == ():
                f_dict[kk] = f_dict[kk].tolist()
        fid.close()
    return  f_dict

def dict_of_arrays_and_scalar_from_h5(filename):
    with h5py.File(filename, 'r', driver='mpio', comm=MPI.COMM_WORLD) as fid:
        f_dict = {}
        for kk in list(fid.keys()):
            f_dict[kk] = np.array(fid[kk]).copy()
            if f_dict[kk].shape == ():
                f_dict[kk] = f_dict[kk].tolist()
        fid.close()
    return  f_dict

import h5py
from mpi4py import MPI
from warp import *


def dict_to_h5(dict_save, filename, compression=None, compression_opts=None):
    """
    Save a dictionary into a h5 file with MPI support
    - dict_save: dictionary to save
    - filename: name of the h5 file
    - compression: compression strategy (as documented in h5py)
    - compression_opts: compression options (as documented in h5py)
    """
    fid = h5py.File(filename, 'w', driver='mpio', comm=MPI.COMM_WORLD)
    for kk in list(dict_save.keys()): 
        fid.create_dataset(kk, data=dict_save[kk], compression=compression, compression_opts=compression_opts)
    fid.close()


def dict_to_h5_serial(dict_save, filename, compression=None, compression_opts=None):
    """
    Save a dictionary into a h5 file without MPI support
    - dict_save: dictionary to save
    - filename: name of the h5 file
    - compression: compression strategy (as documented in h5py)
    - compression_opts: compression options (as documented in h5py)
    """
    fid = h5py.File(filename, 'w')
    for kk in list(dict_save.keys()):
        fid.create_dataset(kk, data=dict_save[kk], compression=compression, compression_opts=compression_opts)
    fid.close()


def dict_of_arrays_and_scalar_from_h5_serial(filename):
    """
    Read a h5 file into a dictionary of arrays and scalars without MPI support
    - filename: name of the h5 file
    """
    with h5py.File(filename, 'r') as fid:
        f_dict = {}
        for kk in list(fid.keys()):
            f_dict[kk] = np.array(fid[kk]).copy()
            if f_dict[kk].shape == ():
                f_dict[kk] = f_dict[kk].tolist()
        fid.close()
    return f_dict


def dict_of_arrays_and_scalar_from_h5(filename):
    """
    Read a h5 file into a dictionary of arrays and scalars with MPI support
    - filename: name of the h5 file
    """
    with h5py.File(filename, 'r', driver='mpio', comm=MPI.COMM_WORLD) as fid:
        f_dict = {}
        for kk in list(fid.keys()):
            f_dict[kk] = np.array(fid[kk]).copy()
            if f_dict[kk].shape == ():
                f_dict[kk] = f_dict[kk].tolist()
        fid.close()
    return f_dict

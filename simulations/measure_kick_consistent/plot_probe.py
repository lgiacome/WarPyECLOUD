import sys
import os
BIN = os.path.expanduser("../../")
if BIN not in sys.path:
    sys.path.append(BIN)

import numpy as np
import matplotlib.pyplot as plt

from h5py_manager import dict_of_arrays_and_scalar_from_h5

a = dict_of_arrays_and_scalar_from_h5('probes.h5')

plt.plot(a['ey'][0,:])
plt.show()

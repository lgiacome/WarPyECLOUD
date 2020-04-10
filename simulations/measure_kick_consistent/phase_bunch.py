import sys
import os
BIN = os.path.expanduser("../../")
if BIN not in sys.path:
    sys.path.append(BIN)

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c as c_light
from h5py_manager import dict_of_arrays_and_scalar_from_h5

a = dict_of_arrays_and_scalar_from_h5('probes.h5')
sigmat = sigmat= 1.000000e-09/4.
t_offs = 5.23452e-8+0.3/c_light 
t_offs_cross = 5.23452e-8
b_spac = 2.5e-9
dt = 1.3906163023765098e-11

def input(t):
    phase_delay = 0 #np.pi/2
    freq_t = 400e6
    width = 420e-3
    w_t = 2*np.pi*freq_t
    w_z = c_light*np.sqrt((w_t/c_light)**2 - (np.pi/width)**2)
    x = 0
    source_z = 0
    return np.sin(-np.pi/width*(x+width/2))*(np.sin(w_t*t-phase_delay)*np.cos(w_z*source_z) - np.cos(w_t*t-phase_delay)*np.sin(w_z*source_z))

def time_prof(t):
    val = 0
    n_bunches = 1 
    for i in range(0, n_bunches):
        val += (1./np.sqrt(2*np.pi*sigmat**2)
               *np.exp(-(t-i*b_spac-t_offs+0.3/c_light)**2
               /(2*sigmat**2))*dt)
    return val

Nt = len(a['ey'][0])

tt = dt*np.linspace(0,Nt-1,Nt)
n_steps_init = 0
tt = tt[n_steps_init:]
plt.plot(tt,a['ey'][0,n_steps_init:],'b')
plt.plot(tt,time_prof(tt)/np.max(time_prof(tt))*20e6,'r')
plt.plot(tt,np.zeros_like(tt),'g--')
plt.plot(t_offs_cross*np.ones(2), np.linspace(-20,20,2),'g--')
plt.plot(tt, 20*np.ones_like(tt),'g--')
plt.show()

import scipy.io as sio
import matplotlib.pyplot as plt

a = sio.loadmat('input.mat')

plt.plot(a['t'][0],a['sig'][0])
plt.show()

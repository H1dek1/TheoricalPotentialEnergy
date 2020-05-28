#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize

def characteristic_pole(alpha, gamma):
    x_psi = alpha + 2*gamma
    y_psi = 9 * np.sqrt(3) * gamma
    psi = np.arctan(y_psi/x_psi)
    return -psi/2

def external_field_pole(alpha, gamma):
    x_phi = (4*alpha + 5*gamma)
    y_phi = 3 * np.sqrt(3) * gamma
    phi =  np.arctan(y_phi/x_phi)
    return -phi

#alpha_arr = np.linspace(0.01, 100, 100)
#gamma_arr = np.linspace(0.01, 300, 300)

alpha_arr = np.arange(1.0, 100, 1.0)
gamma_arr = np.arange(1.0, 100, 1.0)

alpha_arr, gamma_arr = np.meshgrid(alpha_arr, gamma_arr)
diff = external_field_pole(alpha_arr, gamma_arr) - characteristic_pole(alpha_arr, gamma_arr)

fig, ax = plt.subplots(1, 1)
ax.set_title('external pole - static pole')
ax.set_xlabel('$\\alpha$', fontsize=15)
ax.set_ylabel('$\\gamma$', fontsize=15)
mappable = ax.pcolor(alpha_arr, gamma_arr, diff, cmap="coolwarm", norm=Normalize(vmin=-0.01, vmax=0.01))
pp = fig.colorbar(mappable, ax=ax, orientation='vertical')
fig.tight_layout()

plt.show()


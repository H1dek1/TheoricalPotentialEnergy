#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import matplotlib.animation as animation

from potential_energy import potentialEnergy, potentialEnergySepa
from potential_energy import gradEnergy, gradEnergySepa, GradientDescentSepa



#-----------------------------------------------main function
## matplitlib setting
fig, ax = plt.subplots(figsize=(10, 8))
ax.set_xlabel('$\\theta$', fontsize=18)
ax.set_ylabel('$U_{all}$', fontsize=18)

ax.set_xlim(-4*np.pi, 4*np.pi)
#ax.set_ylim()
ax.grid()

ax.set_xticks(np.linspace(-4*np.pi, 4*np.pi, 9))
ax.set_xticklabels([
    '$-4\\pi$',
    '$-3\\pi$',
    '$-2\\pi$',
    '$-\\pi$',
    '$0$',
    '$\\pi$',
    '$2\\pi$',
    '$3\\pi$',
    '$4\\pi$',
    ], fontsize=15)
ax.tick_params(axis='y', labelsize=15)

## paramters
alpha = 100
gamma = 1
x1 = np.linspace(-4*np.pi, 4*np.pi, 8000)
ax.set_title("$\\alpha={}, \\gamma={}$".format(alpha, gamma))

#energy = potentialEnergySepa(x=x1, a=alpha, c=gamma, f_ext=ext_field)
#grad = gradEnergySepa(x=x1, a=alpha, c=gamma, f_ext=ext_field)
#pole_x = GradientDescentSepa(x0=0, a=alpha, c=gamma, f_ext=ext_field)
#pole_y = potentialEnergySepa(x=pole_x, a=alpha, c=gamma, f_ext=ext_field)

ims = []
t = np.linspace(0, 4*np.pi, 200)
f_arr = np.cos(t)

for ext_field in f_arr:
    energy = potentialEnergySepa(x=x1, a=alpha, c=gamma, f_ext=ext_field)
    im = ax.plot(x1, energy, c='c')
    if ext_field == f_arr[0]:
        #pole_x = GradientDescent(x_init, alpha, gamma, f)
        pole_x = GradientDescentSepa(x0=0, a=alpha, c=gamma, f_ext=ext_field)
    else:
        #pole_x = GradientDescent(pole_x, alpha, gamma, f)
        pole_x = GradientDescentSepa(pole_x, a=alpha, c=gamma, f_ext=ext_field)

    #pole_y = potentialEnergy(alpha, gamma, pole_x, f)
    pole_y = potentialEnergySepa(x=pole_x, a=alpha, c=gamma, f_ext=ext_field)
    im += ax.plot(pole_x, pole_y, marker='.', markersize=20, color='r')
    ims.append(im)

ani = animation.ArtistAnimation(fig, ims, interval=100)
ani.save('anim2.mp4', writer="ffmpeg")
plt.show()

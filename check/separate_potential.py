#!/usr/bin/env python3
import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation

FLAG = 1 # 0->なし, 1->あり


d_time = 1.0e-2
omega = 2*np.pi
num_cycle = 1
max_iter = int(num_cycle / d_time)
out_time = 1.0e-2
out_iter = int(out_time / d_time)
sleep_iter = int(1 / d_time)

alpha = 1.0e+0
beta = 6.0e-4
gamma = 1.0e-1
a_l = 0.3

def all_energy(theta, ext_field):
    energy = 0
    energy += u_ext(theta, ext_field)
    energy += u_dd(theta)
    energy += u_ext_p(theta, ext_field)
    energy += u_dd_p(theta)
    return energy

def u_ext(theta, ext_field):
    moment = np.array([-np.sin(theta), np.cos(theta)]).T
    b_ext = np.array([0, ext_field])
    return -4 * alpha * np.dot(moment, b_ext)

def u_dd(theta):
    return 3 - np.cos(2*theta)

def u_ext_p(theta, ext_field):
    moment = np.array([-np.sin(theta), np.cos(theta)]).T
    arr = np.array([3*np.sqrt(3), 5])
    return -gamma * ext_field * np.dot(moment, arr)

def u_dd_p(theta):
    moment = np.array([-np.sin(theta), np.cos(theta)]).T
    arr = np.array([3*np.sqrt(3), 5])
    return gamma / alpha * (3*np.sqrt(3)*np.sin(theta - np.pi/3) + 2*np.cos(theta)) * np.dot(moment, arr)
    

theta_arr = np.linspace(-np.pi, np.pi, 100)
time = 0
external_magnetic_field = np.cos(time)
fig, ax = plt.subplots(1, 1, figsize=(10, 8))
ax.set_title('alpha={}, gamma={}'.format(alpha, gamma))
flag_legend = True
ims = []

for iter in tqdm(range(max_iter)):
    potential_ext = u_ext(theta_arr, external_magnetic_field)
    im = ax.plot(theta_arr, potential_ext, color='C0', linestyle='dashed', label='direct external field')
    
    potential_dd = u_dd(theta_arr)
    im += ax.plot(theta_arr, potential_dd, color='C1', linestyle='dashed', label='direct dipole field')
    
    potential_ext_p = u_ext_p(theta_arr, external_magnetic_field)
    im += ax.plot(theta_arr, potential_ext_p, color='C2', linestyle='dashed', label='external field through para')
    
    potential_dd_p = u_dd_p(theta_arr)
    im += ax.plot(theta_arr, potential_dd_p, color='C3', linestyle='dashed', label='dipole field through para')
    
    potential_all = all_energy(theta_arr, external_magnetic_field)
    im += ax.plot(theta_arr, potential_all, color='C4', label='sum of all energy')
    if flag_legend:
        ax.legend()
        flag_legend = False
    
    ims.append(im)

    time += d_time
    external_magnetic_field = np.cos(omega*time)
#plt.show()

ani = animation.ArtistAnimation(fig, ims, interval=(d_time*1.0e+3*5))
print('Saving animation ...')
ani.save('potential.mp4', writer='ffmpeg')
print('Success!')


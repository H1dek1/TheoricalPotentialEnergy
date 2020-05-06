#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import matplotlib.animation as animation
import matplotlib.patches as patches


alpha = 1
beta = 0.05
d_time = 1e-2
num_cycle = 2
max_iter = num_cycle / d_time
time = 0
omega = 2*np.pi
a_l = 0.3

class PermanentParticle:
    def __init__(self, angle):
        self.theta = angle
        self.moment = np.array([-np.sin(self.theta), np.cos(self.theta), 0])
        self.torque = 0

    def angle(self):
        return np.arctan2(self.moment[1], self.moment[0]) - np.pi/2

    def calcTorque(self, magnetic_field):
        self.torque = alpha * np.cross(self.moment, magnetic_field)
        return self.torque

    def update(self, dt=d_time):
        self.d_theta = (beta / (a_l**3)) * self.torque[2]
        self.theta += self.d_theta * dt
        self.moment = np.array([-np.sin(self.theta), np.cos(self.theta), 0])

    def potential(self, ext_moment, val):
        val_arr = np.array([-np.sin(val), np.cos(val), np.zeros(val.size)])
        energy_arr = -2 * alpha * np.dot(val_arr.T, ext_moment)
        energy = -2 * alpha * np.dot(self.moment, ext_moment)
        return energy_arr, energy



class ExternalMagneticField:
    def __init__(self, angle=0):
        self.psi = angle
        self.moment = np.array([-np.sin(self.psi), np.cos(self.psi), 0])

    def update(self, dt=d_time):
        rot_matrix = np.array([
            [np.cos(omega*dt), -np.sin(omega*dt), 0],
            [np.sin(omega*dt), np.cos(omega*dt), 0],
            [0, 0, 1]
            ])

        self.moment = np.dot(rot_matrix, self.moment)



#--------------------------
fig, axes = plt.subplots(2, 1, figsize=(10, 8))
axes[0].set_xlabel('$x$', fontsize=15)
axes[0].set_ylabel('$y$', fontsize=15)
#axes[0].grid()
axes[0].set_xlim(-3, 3)
axes[0].set_ylim(-1, 1)
axes[0].set_aspect('equal')
particle = patches.Circle(xy=(0, 0), radius=a_l, fill=False)
magnetic_field = patches.Circle(xy=(-2, -0.5), radius=0.5, fill=False)

axes[0].add_patch(particle)
axes[0].add_patch(magnetic_field)



axes[1].set_title('Potential Energy')

ims = []


theta = np.linspace(-2*np.pi, 2*np.pi, 400)
b_ext = ExternalMagneticField(0)
perm = PermanentParticle(0)

for i in range(int(max_iter)):
    pos_x = np.array((-2, 0))
    pos_y = np.array((-0.5, 0))
    vec_x = np.array((b_ext.moment[0], perm.moment[0]))
    vec_y = np.array((b_ext.moment[1], perm.moment[1]))

    im1 = axes[0].quiver(pos_x, pos_y, vec_x, vec_y, color=('black', 'black'), angles='xy', scale_units='xy', scale=1.5, pivot='mid')
    potential, pole_y = perm.potential(b_ext.moment, theta)
    im2 = axes[1].plot(theta, potential, c='c')
    im2 += axes[1].plot(perm.angle(), pole_y, marker='.', markersize=20, color='r')
    ims.append([im1]+im2)
    #ims.append(im2)
    b_ext.update()
    perm.calcTorque( b_ext.moment )
    perm.update()

ani = animation.ArtistAnimation(fig, ims, interval=(d_time*1e+3*3))
ani.save('test2.mp4', writer='ffmpeg')
#plt.show()

#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import matplotlib.animation as animation
import matplotlib.patches as patches


alpha = 1
beta = 1
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

    def calcTorque(self, magnetic_field):
        self.torque = np.cross(self.moment, magnetic_field)
        return self.torque

    def update(self, dt=d_time):
        self.d_theta = (beta / (a_l**3)) * self.torque[2]
        self.theta += self.d_theta * dt
        self.moment = np.array([-np.sin(self.theta), np.cos(self.theta), 0])



class ExternalMagneticField:
    def __init__(self, angle=0):
        self.psi = angle
        self.moment = alpha * np.array([-np.sin(self.psi), np.cos(self.psi), 0])

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

ims = []


b_ext = ExternalMagneticField(0)
perm = PermanentParticle(np.pi/2)

for i in range(int(max_iter)):
    #perm.calcTorque( b_ext.moment )
    #perm.update()

    im = axes[0].quiver(-2, -0.5, b_ext.moment[0], b_ext.moment[1], color='blue', angles='xy', scale_units='xy', scale=1, pivot='mid')
    ims.append([im])
    b_ext.update()

ani = animation.ArtistAnimation(fig, ims, interval=(d_time*1e+3*3))
ani.save('test2.mp4', writer='ffmpeg')
#plt.show()

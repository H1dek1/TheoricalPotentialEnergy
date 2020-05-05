#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import matplotlib.animation as animation

alpha = 100

class Particle():
    def __init__(self, angle):
        self.angle = angle

    def calcTorque(self, magnetic_field):
        moment = np.array([-np.sin(self.angle), np.cos(self.angle), 0])
        cross = alpha*np.cross(moment, magnetic_field)

        return cross

    def updateAngle(self, torque):
        omega = torque/(0.3**3)
        self.angle += omega[2]



def potentialEnergy(x, f):
    return -alpha * f * np.cos(x)



fig, ax = plt.subplots(figsize=(10, 8))
ax.set_xlabel('$\\theta$', fontsize=18)
ax.set_ylabel('$U_{all}$', fontsize=18)

ax.set_xlim(-2*np.pi, 2*np.pi)
#ax.set_ylim()
ax.grid()

ax.set_xticks(np.linspace(-2*np.pi, 2*np.pi, 5))
ax.set_xticklabels([
    '$-2\\pi$',
    '$-\\pi$',
    '$0$',
    '$\\pi$',
    '$2\\pi$',
    ], fontsize=15)
ax.tick_params(axis='y', labelsize=15)

theta = np.linspace(-2*np.pi, 2*np.pi)
t = np.linspace(0, 2*np.pi, 100)
f_ext_arr = np.cos(t)
particle = Particle(np.pi/2)
ims = []

for f_ext in f_ext_arr:
    torque = particle.calcTorque(np.array([0, f_ext, 0]))
    print(torque)
    particle.updateAngle(torque)

    energy = potentialEnergy(theta, f_ext)
    im = ax.plot(theta, energy, c='c')
    ims.append(im)

ani = animation.ArtistAnimation(fig, ims, interval=100)
plt.show()

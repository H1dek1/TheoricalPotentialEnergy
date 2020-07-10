#!/usr/bin/env python3

from tqdm import tqdm
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import matplotlib.animation as animation
import matplotlib.patches as patches


alpha = 1
beta = 0.1
d_time = 1e-3
num_cycle = 2
max_iter = int(num_cycle / d_time)
out_time = 2.0e-2
out_iter = int(out_time / d_time)
time = 0
omega = 2*np.pi
a_l = 0.3

class PermanentParticle:
    def __init__(self, angle):
        self.theta = angle
        self.moment = np.array([-np.sin(self.theta), np.cos(self.theta), 0])
        self.torque = 0

    def angle(self):
        return self.theta

    def calcTorque(self, magnetic_field):
        self.torque = alpha * np.cross(self.moment, magnetic_field)
        return self.torque

    def update(self, dt=d_time):
        self.d_theta = (beta / (a_l**3)) * self.torque[2]
        self.theta += self.d_theta * dt
        self.moment = np.array([-np.sin(self.theta), np.cos(self.theta), 0])

    def potential(self, ext_moment, val):
        val_arr = np.array([-np.sin(val), np.cos(val), 0])
        energy_arr = -2 * alpha * np.dot(val_arr.T, ext_moment)
        energy = -2 * alpha * np.dot(self.moment, ext_moment)
        return energy_arr, energy

    def gradPotential(self, x, ext_moment):
        return 2 * np.sin(x - np.arctan2(ext_moment[1], ext_moment[0]))

    def gradientDescent(self, x0, ext_moment):
        x = x0
        counter = 0
        while(True):
            rate = 0.001 / (1 + counter)
            delta = 0.001*self.gradPotential(x, ext_moment)
            if abs(delta) < 1.0e-5:
                break

            x -= delta
            counter += 1

        return x



class ExternalMagneticField:
    def __init__(self, angle=0):
        self.psi = angle
        self.moment = np.array([-np.sin(self.psi), np.cos(self.psi), 0])

    def update(self, dt=d_time):
        self.psi += omega*dt
        self.moment = np.array([-np.sin(self.psi), np.cos(self.psi), 0])



#--------------------------
fig, axes = plt.subplots(2, 1, figsize=(10, 8))
axes[0].set_xlabel('$x$', fontsize=15)
axes[0].set_ylabel('$y$', fontsize=15)
#axes[0].grid()
axes[0].set_xlim(-3, 3)
axes[0].set_ylim(-1, 1)
axes[0].set_aspect('equal')
axes[0].text(-2.8, -0.9, '$B_{ext}$', fontsize=20)
particle = patches.Circle(xy=(0, 0), radius=a_l, fc='gray', ec='gray', fill=True, zorder=1)
axes[0].add_patch(particle)
#magnetic_field = patches.Circle(xy=(-2, -0.5), radius=0.5, fill=False)
#axes[0].add_patch(magnetic_field)



axes[1].set_title('Potential Energy')
axes[1].set_xlabel('$\\theta$', fontsize=15)
axes[1].set_ylabel('$Potential Energy$', fontsize=15)
#axes[1].set_xticks(np.linspace(-2*np.pi, 2*np.pi, 5))
axes[1].set_xticks([-2*np.pi, -np.pi, 0, np.pi, 2*np.pi])
axes[1].set_xticklabels(['$-2\\pi$', '$-\\pi$', 0, '$\\pi$', '$2\\pi$'])

ims = []


theta = np.linspace(-2*np.pi, 2*np.pi, 400)
b_ext = ExternalMagneticField(0)
perm = PermanentParticle(0)

for i in tqdm(range(max_iter)):
    if i%out_iter == 0:
        pos_x = np.array([-2, 0])
        pos_y = np.array([-0.5, 0])
        vec_x = np.array( [0.8*(b_ext.moment[0]/np.linalg.norm(b_ext.moment)), 0.8*2*a_l*(perm.moment[0]/np.linalg.norm(b_ext.moment))] )
        vec_y = np.array( [0.8*(b_ext.moment[1]/np.linalg.norm(b_ext.moment)), 0.8*2*a_l*(perm.moment[1]/np.linalg.norm(b_ext.moment))] )

        im1 = axes[0].quiver(pos_x, pos_y, vec_x, vec_y, color=('black', 'black'), angles='xy', scale_units='xy', scale=1, pivot='mid', zorder=2)
        potential, theta_0 = perm.potential(b_ext.moment, theta)
        im2 = axes[1].plot(theta, potential, color='C0')
        im2 += axes[1].plot(perm.angle(), theta_0, marker='.', markersize=10, color='red')
        im3 = axes[1].axvline(perm.angle(), color='red')

        pole_x = perm.gradientDescent(0, b_ext.moment)
        _, pole_y = perm.potential(b_ext.moment, pole_x)
        #im4 = axes[1].plot(pole_x, pole_y, marker='.', markersize=10, color='C3')

        #ims.append([im1]+im2+[im3]+im4)
        ims.append([im1]+im2+[im3])

    b_ext.update()
    perm.calcTorque( b_ext.moment )
    perm.update()

print("extreme value :{}".format(perm.angle()))
print("pole value :{}".format(pole_x))
ani = animation.ArtistAnimation(fig, ims, interval=(out_time*1e+3*3))
print("Saving animation ...")
ani.save('slipping.mp4', writer='ffmpeg')
#plt.show()

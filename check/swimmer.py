#!/usr/bin/env python3
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches

d_time = 1.0e-4
omega = 2*np.pi
num_cycle = 3
max_iter = num_cycle / d_time
out_time = 1.0e-2
out_iter = int(out_time / d_time)
sleep_iter = int(1 / d_time)

alpha = 1.0e+2
beta = 1.0e-2
gamma = 1.0e+1
a_l = 0.3

class ExternalMagneticField:
    def __init__(self, angle=0):
        self.psi = angle
        self.moment = np.array([0, np.cos(self.psi), 0])

    def update(self, dt=d_time):
        self.psi += omega * dt
        self.moment = np.array([0, np.cos(self.psi), 0])


class Swimmer:
    def __init__(self, position, init_angle):
        self.pos = position
        self.theta = init_angle
        self.permanent_moment = np.array([-np.sin(self.theta), np.cos(self.theta), 0.0])
        self.para_moment = np.zeros(3)
        self.torque = np.zeros(3)

        self.nx = np.array([1, 0, 0])
        self.npara = np.array([np.cos(np.pi/3), np.sin(np.pi/3), 0])

    def calcParamagneticMoment(self, ext_field):
        self.para_moment = gamma * ext_field

    def calcTorque(self, ext_field):
        b_all = alpha * ext_field
        the_other_moment = np.array([
            -self.permanent_moment[0],
            self.permanent_moment[1],
            self.permanent_moment[2]
            ])
        b_all += 3*np.dot(the_other_moment, self.nx)*self.nx - the_other_moment
        b_all += 3*np.dot(self.para_moment, self.npara)*self.npara - self.para_moment

        self.torque = np.cross(self.permanent_moment, b_all)

    def update(self, dt=d_time):
        self.theta += beta * ( 1/(a_l**3) + 1/2 ) * self.torque[2] * dt
        self.permanent_moment = np.array([-np.sin(self.theta), np.cos(self.theta), 0.0])

    def particlePosition(self):
        return np.array([
            self.pos - 0.5*self.nx,
            self.pos + 0.5*self.nx,
            self.pos + np.array([0, np.sin(np.pi/3), 0])
            ]).T

    def particleMoment(self):
        the_other_moment = np.array([
            -self.permanent_moment[0],
            self.permanent_moment[1],
            self.permanent_moment[2]
            ])
        return np.array([
            self.permanent_moment,
            the_other_moment,
            self.para_moment/gamma
            ]).T




#-------------------------------------------
init_position = np.array([0, 0, 0])

swimmer = Swimmer(init_position, 0)
magnetic_field = ExternalMagneticField()

for i in range(int(sleep_iter)):
    swimmer.calcParamagneticMoment(magnetic_field.moment)
    swimmer.calcTorque(magnetic_field.moment)
    swimmer.update()
    #magnetic_field.update()


fig, axes = plt.subplots(2, 1, figsize=(10, 8))
axes[0].set_xlabel('$x/l$', fontsize=15)
axes[0].set_ylabel('$y/l$', fontsize=15)
axes[0].set_xlim(-3, 3)
axes[0].set_ylim(-1, 1.5)
axes[0].set_aspect('equal')
particle1 = patches.Circle(xy=(-0.5, 0), radius=a_l, fill=False)
particle2 = patches.Circle(xy=(0.5, 0), radius=a_l, fill=False)
particle3 = patches.Circle(xy=(0, np.sqrt(3)/2), radius=a_l, fill=False)
axes[0].add_patch(particle1)
axes[0].add_patch(particle2)
axes[0].add_patch(particle3)
ims = []

for i in tqdm(range(int(max_iter))):
    swimmer.calcParamagneticMoment(magnetic_field.moment)
    swimmer.calcTorque(magnetic_field.moment)
    #####

    if i%out_iter == 0:
        positions = swimmer.particlePosition()
        moments = 0.5*swimmer.particleMoment()
        im1 = axes[0].quiver(positions[0], positions[1], moments[0], moments[1], color='black', angles='xy', scale_units='xy', scale=1, pivot='mid', width=5.0e-3)
        ims.append([im1])

    #####
    swimmer.update()
    magnetic_field.update()

print("final particle angle: {}".format(swimmer.theta))
ani = animation.ArtistAnimation(fig, ims, interval=(out_time*1.0e+3*5))
print('Saving animation ...')
ani.save('sample.mp4', writer='ffmpeg')

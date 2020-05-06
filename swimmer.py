#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import matplotlib.animation as animation
import matplotlib.patches as patches


alpha = 1.0e+2
beta = 1.0e-2
gamma = 1.0e+1

d_time = 1e-2
num_cycle = 2
max_iter = num_cycle / d_time
time = 0
omega = 2*np.pi
a_l = 0.3

class PermanentParticle:
    def __init__(self, angle, pos):
        self.pos = pos
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
        val_arr = np.array([-np.sin(val), np.cos(val), np.zeros(val.size)])
        energy_arr = -2 * alpha * np.dot(val_arr.T, ext_moment)
        energy = -2 * alpha * np.dot(self.moment, ext_moment)
        return energy_arr, energy



class ExternalMagneticField:
    def __init__(self, angle=0):
        self.psi = angle
        self.moment = np.array([0, np.cos(self.psi), 0])

    def update(self, dt=d_time):
        self.psi += omega*dt
        self.moment = np.array([0, np.cos(self.psi), 0])



class Swimmer:
    def __init__(self, pos, particle_angle):
        self.pos = pos
        self.theta = particle_angle
        self.permanent_magnetic_moment = np.array([-np.sin(self.theta), np.cos(self.theta), 0])
        self.paramagnetic_moment = np.zeros(3)
        self.torque = 0

    def calcParamagneticMoment(self, ext_field):
        magnetic_field_para = ext_field
        #magnetic_field_para = ext_field \
        #        - (1/alpha)*(3*np.sqrt(3)*np.sin(self.theta - np.pi/3) + 2*np.cos(self.theta))*np.array([0, 1, 0])

        self.paramagnetic_moment = gamma * magnetic_field_para

    def calcTorque(self, ext_field):
        b_ext = ext_field
        other_magnetic_moment = np.array([
            -self.permanent_magnetic_moment[0],
            self.permanent_magnetic_moment[1],
            self.permanent_magnetic_moment[2]
            ])
        n1_vec = np.array([1, 0, 0])
        b_dd = 3*np.dot(other_magnetic_moment, n1_vec)*n1_vec - other_magnetic_moment
        n2_vec = np.array([0.5, np.sqrt(3)/2, 0])
        b_para = 3*np.dot(self.paramagnetic_moment, n2_vec)*n2_vec - self.paramagnetic_moment

        self.torque = np.cross(self.permanent_magnetic_moment, (alpha*b_ext + b_dd + gamma*b_para))

    def updateAngle(self, dt=d_time):
        self.theta += beta*self.torque[2]*(1/(a_l**3) + 1/2)*dt
        self.permanent_magnetic_moment = np.array([-np.sin(self.theta), np.cos(self.theta), 0])

    def posParticle(self, id_particle):
        if id_particle == 1:
            return self.pos - np.array([0.5, 0, 0])
        elif id_particle == 2:
            return self.pos + np.array([0.5, 0, 0])
        elif id_particle == 3:
            return self.pos + np.array([0, np.sqrt(3)/2, 0])
        else:
            return np.zeros(3)

    def momentParticle(self, id_particle):
        if id_particle == 1:
            return self.permanent_magnetic_moment
        elif id_particle == 2:
            return np.array([
                -self.permanent_magnetic_moment[0],
                self.permanent_magnetic_moment[1],
                self.permanent_magnetic_moment[2]
                ])
        elif id_particle == 3:
            return self.paramagnetic_moment
        else:
            return np.zeros(3)


#--------------------------
fig, axes = plt.subplots(2, 1, figsize=(10, 8))
axes[0].set_xlabel('$x$', fontsize=15)
axes[0].set_ylabel('$y$', fontsize=15)
#axes[0].grid()
axes[0].set_xlim(-3, 3)
axes[0].set_ylim(-1, 1.5)
axes[0].set_aspect('equal')
permanent1 = patches.Circle(xy=(-0.5, 0), radius=a_l, fill=False)
permanent2 = patches.Circle(xy=(0.5, 0), radius=a_l, fill=False)
paramagnetic = patches.Circle(xy=(0, 0.866), radius=a_l, fill=False)
magnetic_field = patches.Circle(xy=(-2, -0.5), radius=0.2, fill=False)

axes[0].add_patch(permanent1)
axes[0].add_patch(permanent2)
axes[0].add_patch(paramagnetic)
axes[0].add_patch(magnetic_field)

ims = []

swimmer = Swimmer(np.array([0.0, 0.0, 0.0]), 0)
b_ext = ExternalMagneticField(0)
for i in range(int(max_iter)):
    if (i+1) % 100 == 0: print("Drawing {}/{}".format(i+1, max_iter))
    swimmer.calcParamagneticMoment(b_ext.moment)
    pos_x = np.array([
        -2, 
        swimmer.posParticle(1)[0], 
        swimmer.posParticle(2)[0], 
        swimmer.posParticle(3)[0]])
    pos_y = np.array([
        -0.5, 
        swimmer.posParticle(1)[1], 
        swimmer.posParticle(2)[1], 
        swimmer.posParticle(3)[1]])

    vec_x = np.array( [
        0.3*b_ext.moment[0], 
        0.8*2*a_l*(swimmer.momentParticle(1)[0]/np.linalg.norm(swimmer.momentParticle(1))),
        0.8*2*a_l*(swimmer.momentParticle(2)[0]/np.linalg.norm(swimmer.momentParticle(2))),
        0.8*2*a_l*(swimmer.momentParticle(3)[0]/np.linalg.norm(swimmer.momentParticle(3)))] )

    vec_y = np.array( [
        0.3*b_ext.moment[1], 
        0.8*2*a_l*(swimmer.momentParticle(1)[1]/np.linalg.norm(swimmer.momentParticle(1))),
        0.8*2*a_l*(swimmer.momentParticle(2)[1]/np.linalg.norm(swimmer.momentParticle(2))),
        0.8*2*a_l*(swimmer.momentParticle(3)[1]/np.linalg.norm(swimmer.momentParticle(3)))] )
    im1 = axes[0].quiver(pos_x, pos_y, vec_x, vec_y, color='black', angles='xy', scale_units='xy', scale=1, pivot='mid')
    ims.append([im1])

    swimmer.calcTorque(b_ext.moment)
    swimmer.updateAngle(d_time)
    b_ext.update(d_time)
#
#
#
#axes[1].set_title('Potential Energy')
#axes[1].set_xlabel('$\\theta$', fontsize=15)
#axes[1].set_ylabel('$Potential Energy$', fontsize=15)
#axes[1].set_xticks(np.linspace(-2*np.pi, 2*np.pi, 5))
#
#ims = []
#
#
#theta = np.linspace(-2*np.pi, 2*np.pi, 400)
#b_ext = ExternalMagneticField(0)
#perm1 = PermanentParticle(0, np.array([-0.5, 0]))
#perm2 = PermanentParticle(0, np.array([0.5, 0]))
#
#for i in range(int(max_iter)):
#    if (i+1) % 100 == 0: print("plotting {}/{}".format(i+1, max_iter))
#
#    pos_x = np.array([-2, 0])
#    pos_y = np.array([-0.5, 0])
#    vec_x = np.array( [0.8*(b_ext.moment[0]/np.linalg.norm(b_ext.moment)), 0.8*2*a_l*(perm.moment[0]/np.linalg.norm(b_ext.moment))] )
#    vec_y = np.array( [0.8*(b_ext.moment[1]/np.linalg.norm(b_ext.moment)), 0.8*2*a_l*(perm.moment[1]/np.linalg.norm(b_ext.moment))] )
#
#    im1 = axes[0].quiver(pos_x, pos_y, vec_x, vec_y, color=('black', 'black'), angles='xy', scale_units='xy', scale=1, pivot='mid')
#    potential, pole_y = perm.potential(b_ext.moment, theta)
#    im2 = axes[1].plot(theta, potential, c='c')
#    im2 += axes[1].plot(perm.angle(), pole_y, marker='.', markersize=20, color='r')
#    im3 = axes[1].axvline(perm.angle(), color='r')
#    ims.append([im1]+im2+[im3])
#
#    b_ext.update()
#    perm.calcTorque( b_ext.moment )
#    perm.update()
#
ani = animation.ArtistAnimation(fig, ims, interval=(d_time*1e+3*3))
print("Saving animation ...")
ani.save('swimmer.mp4', writer='ffmpeg')
##plt.show()

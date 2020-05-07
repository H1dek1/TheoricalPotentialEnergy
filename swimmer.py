#!/usr/bin/env python3

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import matplotlib.animation as animation
import matplotlib.patches as patches


alpha = 1.0e+2
beta = 4.0e-3
gamma = 1.0e+1

d_time = 1e-3
num_cycle = 1
max_iter = num_cycle / d_time
time = 0
omega = 2*np.pi
a_l = 0.3

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
        self.torque = np.zeros(3)

    def calcParamagneticMoment(self, ext_field):
        #magnetic_field_para = ext_field
        magnetic_field_para = ext_field \
                - (1/alpha)*(3*np.sqrt(3)*np.sin(self.theta - np.pi/3) + 2*np.cos(self.theta))*np.array([0, 1, 0])

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

        n_para_moment = self.paramagnetic_moment / np.linalg.norm(self.paramagnetic_moment)
        b_para = 3*np.dot(n_para_moment, n2_vec)*n2_vec - n_para_moment

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

    def potentialEnergy(self, ext_field, theta_arr):
        local_energy_arr = self.extEnergy(theta_arr, ext_field[1]) \
                + self.dipoleEnergy(theta_arr) \
                + self.paraEnergy(theta_arr, ext_field[1])
        local_energy = self.extEnergy(self.theta, ext_field[1]) \
                + self.dipoleEnergy(self.theta) \
                + self.paraEnergy(self.theta, ext_field[1])
        #local_energy_arr = (gamma/(2*alpha) - 1)*np.cos(2*theta_arr)
        #local_energy_arr += 3 - 13*gamma/alpha
        #local_energy_arr += (15*np.sqrt(3)*gamma/(2*alpha))*np.sin(2*theta_arr)
        #local_energy_arr += ext_field[1]*(-(2*alpha+5*gamma)*np.cos(theta_arr) + 3*np.sqrt(3)*gamma*np.sin(theta_arr))
        #local_energy = (gamma/(2*alpha) - 1)*np.cos(2*self.theta)
        #local_energy += 3 - 13*gamma/alpha
        #local_energy += (15*np.sqrt(3)*gamma/(2*alpha))*np.sin(2*self.theta)
        #local_energy += ext_field[1]*(-(2*alpha+5*gamma)*np.cos(self.theta) + 3*np.sqrt(3)*gamma*np.sin(self.theta))

        return local_energy_arr, local_energy

    def extEnergy(self, x, ext_moment):
        return -4 * alpha * ext_moment * np.cos(x)

    def dipoleEnergy(self, x):
        return 3 - np.cos(2*x)

    def paraEnergy(self, x, ext_moment):
        field_on_perm = -(3*np.sqrt(3)*np.sin(x - np.pi/3) + 2*np.cos(x))/alpha
        return -2 * gamma * (ext_moment + field_on_perm) * (3*np.cos(x + np.pi/3) + np.cos(x))
        #return -2 * gamma * (ext_moment) * (3*np.cos(x + np.pi/3) + np.cos(x))

    def gradientDescent(self, x0, ext_moment):
        x = x0
        count = 0
        while(True):
            rate = 0.001 / (1 + count)
            delta = rate*self.gradPotential(ext_moment, x)
            if abs(delta) < 1.0e-5:
                break
            x -= delta
            count += 1

        return x

    def gradPotential(self, ext_field, x):
        local_grad = self.gradExtEnergy(x, ext_field[1]) \
                + self.gradDipoleEnergy(x) \
                + self.gradParaEnergy(x, ext_field[1])

        return local_grad

    def gradExtEnergy(self, x, ext_moment):
        return 4 * alpha * ext_moment * np.sin(x)

    def gradDipoleEnergy(self, x):
        return 2 * np.sin(2*x)

    def gradParaEnergy(self, x, ext_moment):
        return 2*(self.gradExtPara(x, ext_moment) + self.gradDipolePara(x))
        #return 2*self.gradExtPara(x, ext_moment)

    def gradExtPara(self, x, ext_moment):
        return gamma*ext_moment*(3*np.sin(x + np.pi/3) + np.sin(x))

    def gradDipolePara(self, x):
        return (gamma/alpha) * ( (3*np.sqrt(3)*np.cos(x - np.pi/3) - 2*np.sin(x))*(3*np.cos(x + np.pi/3) + np.cos(x)) - (3*np.sqrt(3)*np.sin(x - np.pi/3) + 2*np.cos(x))*(3*np.sin(x + np.pi/3) + np.sin(x)) )



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
magnetic_field = patches.Circle(xy=(-2, -0.5), radius=0.4, fill=False)

axes[0].add_patch(permanent1)
axes[0].add_patch(permanent2)
axes[0].add_patch(paramagnetic)
axes[0].add_patch(magnetic_field)


axes[1].set_title('Potential Energy')
axes[1].set_xlabel('$\\theta$', fontsize=15)
axes[1].set_ylabel('$Potential Energy$', fontsize=15)
#axes[1].set_xticks(np.linspace(-2*num_cycle*np.pi, 2*np.pi, 2*(num_cycle)+1))
axes[1].set_xticks(np.linspace(-np.pi/3, np.pi/3, 5))
axes[1].set_xlim(-np.pi/4, np.pi/4)
theta_arr = np.linspace(-2*num_cycle*np.pi, 2*np.pi, 2000*(num_cycle+1))

ims = []

swimmer = Swimmer(np.array([0.0, 0.0, 0.0]), particle_angle=-0.119522)
b_ext = ExternalMagneticField(0)

for i in tqdm(range(int(max_iter))):
    #if (i+1) % 100 == 0: print("Drawing {}/{}".format(i+1, max_iter))
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
        0.6*b_ext.moment[0], 
        0.8*2*a_l*(swimmer.momentParticle(1)[0]/np.linalg.norm(swimmer.momentParticle(1))),
        0.8*2*a_l*(swimmer.momentParticle(2)[0]/np.linalg.norm(swimmer.momentParticle(2))),
        0.8*2*a_l*(swimmer.momentParticle(3)[0]/10)] )

    vec_y = np.array( [
        0.6*b_ext.moment[1], 
        0.8*2*a_l*(swimmer.momentParticle(1)[1]/np.linalg.norm(swimmer.momentParticle(1))),
        0.8*2*a_l*(swimmer.momentParticle(2)[1]/np.linalg.norm(swimmer.momentParticle(2))),
        0.8*2*a_l*(swimmer.momentParticle(3)[1]/10)] )
    im1 = axes[0].quiver(pos_x, pos_y, vec_x, vec_y, color='black', angles='xy', scale_units='xy', scale=1, pivot='mid', width=5.0e-3)

    potential, theta_0 = swimmer.potentialEnergy(b_ext.moment, theta_arr)
    im2 = axes[1].plot(theta_arr, potential, c='c')
    im2 += axes[1].plot(swimmer.theta, theta_0, marker='.', markersize=15, color='r')
    if i == 0:
        pole_x = swimmer.gradientDescent(0, b_ext.moment)
    else:
        pole_x = swimmer.gradientDescent(pole_x, b_ext.moment)

    pole_y, _ = swimmer.potentialEnergy(b_ext.moment, pole_x)

    im2 += axes[1].plot(pole_x, pole_y, marker='.', markersize=15, color='b')

    ims.append([im1]+im2)

    swimmer.calcTorque(b_ext.moment)
    swimmer.updateAngle(d_time)

    if i < 1/d_time:
        pass
    else:
        pass
        #b_ext.update(d_time)

print("収束値:{}".format(swimmer.theta))
print("勾配降下法: {}".format(pole_x))
ani = animation.ArtistAnimation(fig, ims, interval=(d_time*1e+3*5))
print("Saving animation ...")
ani.save('swimmer_test.mp4', writer='ffmpeg')
##plt.show()

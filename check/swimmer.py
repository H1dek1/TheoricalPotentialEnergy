#!/usr/bin/env python3
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches

FLAG = 1 # 0->なし, 1->あり


d_time = 1.0e-4
omega = 2*np.pi
num_cycle = 2
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
        self.npara2 = np.array([-np.cos(np.pi/3), np.sin(np.pi/3), 0])

    def calcParamagneticMoment(self, ext_field):
        the_other_moment = np.array([
            -self.permanent_moment[0],
            self.permanent_moment[1],
            self.permanent_moment[2]
            ])
        b_p = 3*np.dot(self.permanent_moment, -self.npara)*(-self.npara) - self.permanent_moment
        b_p += 3*np.dot(the_other_moment, -self.npara2)*(-self.npara2) - the_other_moment

        if FLAG == 0:
            self.para_moment = gamma * ext_field
        elif FLAG == 1:
            self.para_moment = gamma * (ext_field + b_p/alpha)



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

    def potentialEnergy(self, x_arr, ext_field):
        theta_potential = self.extEnergy(self.theta, ext_field) \
            + self.dipoleEnergy(self.theta)

        if type(x_arr) == np.ndarray:
            x_potential = []
            for x in x_arr:
                x_potential.append(self.extEnergy(x, ext_field) \
                    + self.dipoleEnergy(x))
                #x_potential.append(self.extEnergy(x, ext_field))
        else:
            x_potential = self.extEnergy(x_arr, ext_field) \
                    + self.dipoleEnergy(x_arr)
            #x_potential = self.extEnergy(x_arr, ext_field)

        return theta_potential, x_potential

    def extEnergy(self, theta, ext_field):
        tmp_moment = np.array([
            -np.sin(theta),
            np.cos(theta),
            0
            ])
        field = ext_field.copy()

        if FLAG == 0:
            field += (gamma/alpha) * ext_field[1] * np.array([3*np.sqrt(3)/4, 5/4, 0])
        elif FLAG == 1:
            field += (gamma/alpha) * (ext_field[1] - (3*np.sqrt(3)*np.sin(theta - np.pi/3) + 2*np.cos(theta))/alpha) * np.array([3*np.sqrt(3)/4, 5/4, 0])

        ext_energy = -4 * alpha * np.dot(tmp_moment, field)
        return ext_energy

    def dipoleEnergy(self, theta):
        dipole_energy = 3 - np.cos(2*theta)
        return dipole_energy

    def gradientDescent(self, x0, ext_field):
        x = x0
        while True:
            delta = self.gradEnergy(x, ext_field)
            if abs(delta) < 1.0e-5: break
            x -= 1.0e-4 * delta

        return x

    def gradEnergy(self, theta, ext_field):
        if FLAG == 0:
            grad = self.gradExtEnergy(theta, ext_field) + self.gradDipoleEnergy(theta)
        elif FLAG == 1:
            grad = self.gradExtEnergy2(theta, ext_field) + self.gradDipoleEnergy(theta)

        return grad

    #超常磁性に永久磁石の磁場を含まない場合
    def gradExtEnergy(self, theta, ext_field):
        tmp_vector = np.array([
            -np.cos(theta),
            -np.sin(theta),
            0
            ])
        field = ext_field.copy()
        field += (gamma/alpha) * ext_field[1] * np.array([3*np.sqrt(3)/4, 5/4, 0])
        grad_ext_energy = -4 * alpha * np.dot(tmp_vector, field)
        return grad_ext_energy

    def gradExtEnergy2(self, theta, ext_field):
        tmp_moment = np.array([
            -np.sin(theta),
            np.cos(theta),
            0
            ])
        d_tmp_moment = np.array([
            -np.cos(theta),
            -np.sin(theta),
            0
            ])
        field = ext_field.copy()
        field += (gamma/alpha) * (ext_field[1] - (3*np.sqrt(3)*np.sin(theta - np.pi/3) + 2*np.cos(theta))/alpha) * np.array([3*np.sqrt(3)/4, 5/4, 0])

        d_field = -(gamma/(alpha**2)) * (3*np.sqrt(3)*np.cos(theta - np.pi/3) - 2*np.sin(theta)) * np.array([3*np.sqrt(3)/4, 5/4, 0])

        grad_ext_energy = -4*alpha * (np.dot(tmp_moment, d_field) + np.dot(d_tmp_moment, field))
        return grad_ext_energy

    def gradDipoleEnergy(self, theta):
        grad_dipole_energy = 2 * np.sin(2*theta)
        return grad_dipole_energy





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
axes[1].set_xlabel('$\\theta$', fontsize=15)
axes[1].set_ylabel('Potential Energy', fontsize=15)
#axes[1].set_xlim(-0.3, 0)
#axes[1].set_ylim(-460, -440)
ims = []


if FLAG == 0:
    theta_arr = np.linspace(-2*np.pi, num_cycle*2*np.pi, 100*(1+int(num_cycle)))
elif FLAG == 1:
    theta_arr = np.linspace(-num_cycle*2*np.pi, 2*np.pi, 100*(1+int(num_cycle)))

pole_x = 0

pot_arr = []

for i in tqdm(range(int(max_iter))):
    #mutableな外部地場のモーメントを変更しないため
    b_ext = magnetic_field.moment.copy()
    b_ext.setflags(write=False)

    swimmer.calcParamagneticMoment(b_ext)
    swimmer.calcTorque(b_ext)
    #####
    if i%out_iter == 0:
        #subplot 1, 1
        positions = swimmer.particlePosition()
        moments = 0.5*swimmer.particleMoment()
        im1 = axes[0].quiver(positions[0], positions[1], moments[0], moments[1], \
                color='black', angles='xy', scale_units='xy', scale=1, pivot='mid', width=5.0e-3)

        #subplot 2, 1
        potential, potential_arr = swimmer.potentialEnergy(theta_arr, b_ext)
        im2 = axes[1].plot(theta_arr, potential_arr, c='blue')
        #im2 += axes[1].plot(theta_arr, 100*swimmer.dipoleEnergy(theta_arr), c='orange')
        im2 += axes[1].plot(swimmer.theta, potential, marker='.', markersize=15, color='r')
        pole_x = swimmer.gradientDescent(pole_x, b_ext)
        _, pole_potential = swimmer.potentialEnergy(pole_x, b_ext)
        im2 += axes[1].plot(pole_x, pole_potential, marker='.', markersize=10, color='g')
        ims.append([im1]+im2)
        pot_arr = potential_arr

    #####
    swimmer.update()
    magnetic_field.update()

index = np.where(pot_arr == np.min(pot_arr))
real_pole = theta_arr[index]
print("min index: {}".format(real_pole))

print("final particle angle: {}".format(swimmer.theta))
print("pole angle          : {}".format(pole_x))
ani = animation.ArtistAnimation(fig, ims, interval=(out_time*1.0e+3*5))
print('Saving animation ...')
ani.save('sample.mp4', writer='ffmpeg')

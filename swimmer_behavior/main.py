#!/usr/bin/env python3

import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches

from external_magnetic_field import ExternalMagneticField
from swimmer import Swimmer

def main():
    FLAG = True # True=NewModel, False=OldModel

    d_time = 1.0e-4
    omega = 2*np.pi
    num_cycle = 2
    max_iter = num_cycle / d_time
    out_time = 1.0e-2
    out_iter = int(out_time / d_time)
    sleep_iter = int(1 / d_time)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    matplotlibSetting(fig, axes)
    ims = []
    
    init_position = np.array([0, 0, 0])
    
    swimmer = Swimmer(init_position, 0, flag=FLAG)
    magnetic_field = ExternalMagneticField(angle=0, angle_velocity=omega)

    print('Aligning particles ...')
    
    #for i in range(int(sleep_iter)):
    #    swimmer.calcParamagneticMoment(magnetic_field.moment)
    #    swimmer.calcTorque(magnetic_field.moment)
    #    swimmer.update(d_time)
        #magnetic_field.update(d_time)
    
    
    if FLAG == False:
        theta_arr = np.linspace(-2*np.pi, num_cycle*2*np.pi, 100*(1+int(num_cycle)))
    elif FLAG == True:
        theta_arr = np.linspace(-num_cycle*2*np.pi - np.pi/2, 2*np.pi, 100*(1+int(num_cycle)))
    
    pole_x = 0
    print('Start Iteration')
    for i in tqdm(range(int(max_iter))):
    #for i in range(1):
        #mutableな外部地場のモーメントを変更しないため
        b_ext = magnetic_field.moment.copy()
        b_ext.setflags(write=False)
    
        swimmer.calcParamagneticMoment(b_ext)
        swimmer.calcTorque(b_ext)
        #####
        if i%out_iter == 0:
            #subplot (1, 1)
            positions = swimmer.particlePosition()
            moments = 0.5*swimmer.particleMoment()
            im1_1 = axes[0].quiver(positions[0], positions[1], moments[0], moments[1], \
                    color='black', angles='xy', scale_units='xy', scale=1, pivot='mid', width=5.0e-3, zorder=2)
            im1_2 = axes[0].quiver(-2.0, -0.5, b_ext[0], b_ext[1], \
                    color='black', angles='xy', scale_units='xy', scale=2, width=5.0e-3, zorder=2)
    
            #subplot (2, 1)
            potential, potential_arr = swimmer.potentialEnergy(theta_arr, b_ext)
            im2_1 = axes[1].plot(theta_arr, potential_arr, c='C0')
    
            im2_1 += axes[1].plot(swimmer.theta, potential, marker='.', markersize=10, color='r')
            pole_x = swimmer.gradientDescent(pole_x, b_ext)
            _, pole_potential = swimmer.potentialEnergy(pole_x, b_ext)
            #im2_1 += axes[1].plot(pole_x, pole_potential, marker='.', markersize=10, color='g')
            im2_2 = axes[1].axvline(swimmer.theta, color='r')

            ims.append([im1_1]+[im1_2]+im2_1+[im2_2])
    
        #####
        swimmer.update(d_time)
        magnetic_field.update(d_time)
    
    #print("final particle angle: {}".format(swimmer.theta))
    #print("pole angle          : {}".format(pole_x))
    ani = animation.ArtistAnimation(fig, ims, interval=(out_time*1.0e+3*5))
    print('Saving animation ...')
    ani.save('sample.mp4', writer='ffmpeg')
    print('Success!')


def matplotlibSetting(fig, axes):
    axes[0].set_xlabel('$x/l$', fontsize=15)
    axes[0].set_ylabel('$y/l$', fontsize=15)
    axes[0].set_xlim(-3, 3)
    axes[0].set_ylim(-1, 1.5)
    axes[0].set_aspect('equal')
    axes[0].text(-2.8, -0.9, '$B_{ext}$', fontsize=20)
    particle1 = patches.Circle(xy=(-0.5, 0), radius=Swimmer.a_l, fc='gray', ec='gray', fill=True, zorder=1)
    particle2 = patches.Circle(xy=(0.5, 0), radius=Swimmer.a_l, fc='gray', ec='gray', fill=True, zorder=1)
    particle3 = patches.Circle(xy=(0, np.sqrt(3)/2), radius=Swimmer.a_l, fc='orange', ec='orange', fill=True, zorder=1)
    axes[0].add_patch(particle1)
    axes[0].add_patch(particle2)
    axes[0].add_patch(particle3)
    axes[1].set_xlabel('$\\theta$', fontsize=15)
    axes[1].set_ylabel('Potential Energy', fontsize=15)
    #axes[1].set_xticks(np.arange(-2*np.pi, 5*np.pi, np.pi/2))
    #axes[1].set_xticklabels(['$-2\pi$', '$-3\pi/2$', '$-\pi$', '$-\pi/2$', '$0$', '$\pi/2$', '$\pi$', '$3\pi/2$', '$2\pi$', '$5\pi/2$', '$3\pi$', '$7\pi/2$', '$4\pi$', '$9\pi/2$'])
    axes[1].set_xticks(np.arange(-9*np.pi/2, 5*np.pi/2, np.pi/2))
    axes[1].set_xticklabels(['$-9\pi/2$', '$-4\pi$', '$-7\pi/2$', '$-3\pi$', '$-5\pi/2$', '$-2\pi$', '$-3\pi/2$', '$-\pi$', '$-\pi/2$', '$0$', '$\pi/2$', '$\pi$', '$3\pi/2$', '$2\pi$'])
    #axes[1].set_xlim(-0.3, 0)
    #axes[1].set_ylim(-460, -440)


if __name__ == '__main__':
    main()


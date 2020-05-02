#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import matplotlib.animation as animation

def potentialEnergy(a, c, x, f_ext):

    energy = 3 - 13*c/a
    energy += (c/(2*a) - 1)*np.cos(2*x)
    energy += (15*np.sqrt(3)*c/(2*a))*np.sin(2*x)
    energy += f_ext*(-(2*a+5*c)*np.cos(x) + 3*np.sqrt(3)*c*np.sin(x))

    return energy

def calcgrad(a, c, x, f_ext):
    grad = (2 - c/a) * np.sin(2*x)
    grad += (15*np.sqrt(3)*c/a) * np.cos(2*x)
    grad += f_ext * ((2*a + 5*c)*np.sin(x) + 3*np.sqrt(3)*c*np.cos(x))

    return grad

def calcgradgrad(a, c, x, f_ext):
    gradgrad = (4 - 2*c/a)*np.cos(2*x)
    gradgrad -= (30*np.sqrt(3)*c/a) * np.sin(2*x)
    gradgrad += f_ext * ( (2*a + 5*c)*np.cos(x) -3*np.sqrt(3)*c*np.sin(x))

    return gradgrad


def NewtonMethod(x0, a, c, f_ext):
    x = x0
    while(True):
        delta = 0.1*calcgrad(a, c, x, 1)/calcgradgrad(a, c, x, 1)
        if(abs(delta) < 1.0e-4):
            break
        ax.plot(x, potentialEnergy(a, c, x, 1), marker='.', markersize=6)
        x -= delta

    return x

def GradientDescent(x0, a, c, f_ext):
    x = x0
    counter = 0
    while(True):
        rate = 0.001 / (1 + counter)
        delta = rate*calcgrad(a, c, x, f_ext)
        #if counter > 100 and abs(delta) < 1.0e-5:
        if abs(delta) < 1.0e-5:
            break
        #ax.plot(x, potentialEnergy(a, c, x, 1), marker='.', markersize=6)
        x -= delta
        counter += 1;

    return x

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111)
ax.set_xlabel("$\\theta$")
ax.set_ylabel("PotentialEnergy")
ax.set_xticks([-2*np.pi, -3*np.pi/2, -np.pi, -np.pi/2, 0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
ax.set_xticklabels(["$-2\\pi$", "$-3\\pi/2$", "$-\\pi$", "$-\\pi/2$", 0, "$\\pi/2$", "$\\pi$", "$3\\pi/2$", "$2\\pi$"])
ax.grid()

alpha = 100
gamma = 10
x_init = 0.0

ax.set_title("$\\alpha={}, \\gamma={}$".format(alpha, gamma))
x = np.linspace(-2*np.pi, 2*np.pi, 1000)

ims = []
t = np.linspace(0, 2*np.pi, 100)
f_arr = np.cos(t)

for f in f_arr:
    y1 = potentialEnergy(alpha, gamma, x, f)
    im = ax.plot(x, y1, c='c')
    if f == f_arr[0]:
        pole_x = GradientDescent(x_init, alpha, gamma, f)
    else:
        pole_x = GradientDescent(pole_x, alpha, gamma, f)

    pole_y = potentialEnergy(alpha, gamma, pole_x, f)
    im += ax.plot(pole_x, pole_y, marker='.', markersize=20, color='r')
    ims.append(im)

#pole_x = NewtonMethod(x_init, alpha, gamma, 1)

#print(pole_x)
#print(pole_y)

#ax.scatter(pole_x, pole_y, s=51)
ani = animation.ArtistAnimation(fig, ims, interval=100)
ani.save('ani.mp4', writer="ffmpeg")
plt.show()

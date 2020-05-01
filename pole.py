#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

def potentialEnergy(alpha, gamma, x, f_ext):

    energy = 3 - 13*gamma/alpha
    energy += (gamma/(2*alpha) - 1)*np.cos(2*x)
    energy += (15*np.sqrt(3)*gamma/(2*alpha))*np.sin(2*x)
    energy += f_ext*(-(2*alpha+5*gamma)*np.cos(x) + 3*np.sqrt(3)*gamma*np.sin(x))

    return energy

def calcgrad(alpha, gamma, x, f_ext):
    grad = (2 - gamma/alpha) * np.sin(2*x)
    grad += (15*np.sqrt(3)*gamma/alpha) * np.cos(2*x)
    grad += f_ext * ((2*alpha + 5*gamma)*np.sin(x) + 3*np.sqrt(3)*gamma*np.cos(x))

    return grad

def calcgradgrad(alpha, gamma, x, f_ext):
    gradgrad = (4 - 2*gamma/alpha)*np.cos(2*x)
    gradgrad -= (30*np.sqrt(3)*gamma/alpha) * np.sin(2*x)
    gradgrad += f_ext * ( (2*alpha + 5*gamma)*np.cos(x) -3*np.sqrt(3)*gamma*np.sin(x))

    return gradgrad


def NewtonMethod(x0):
    x = x0
    while(True):
        delta = 0.1*calcgrad(alpha, gamma, x, 1)/calcgradgrad(alpha, gamma, x, 1)
        if(abs(delta) < 1.0e-4):
            break
        ax.plot(x, potentialEnergy(alpha, gamma, x, 1), marker='.', markersize=6)
        x -= delta

    return x


fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111)
plt.grid()

alpha = 100
gamma = 50

x = np.linspace(-np.pi, 2*np.pi, 100)

pole_x = NewtonMethod(1.2)
pole_y = potentialEnergy(alpha, gamma, pole_x, 1)
print("pole_x is {}".format(pole_x))
print("pole_y is {}".format(pole_y))

y1 = potentialEnergy(alpha, gamma, x, 1)
y2 = calcgrad(alpha, gamma, x, 1)
y3 = calcgradgrad(alpha, gamma, x, 1)

ax.plot(x, y1, color='b')
ax.plot(x, y2, color='r')
#ax.plot(x, y3)
ax.plot(pole_x, pole_y, marker='.', markersize=10)
ax.vlines(pole_x, -600, 600, "g")
plt.show()

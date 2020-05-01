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


def NewtonMethod(x0, alpha, gamma, f_ext):
    x = x0
    while(True):
        delta = 0.1*calcgrad(alpha, gamma, x, 1)/calcgradgrad(alpha, gamma, x, 1)
        if(abs(delta) < 1.0e-4):
            break
        #ax.plot(x, potentialEnergy(alpha, gamma, x, 1), marker='.', markersize=6)
        x -= delta

    return x


fig = plt.figure(figsize=(8, 8))
#ax = fig.add_subplot(211)
ax2 = fig.add_subplot(211)
ax3 = fig.add_subplot(212)
ax2.set_title("$\\theta(\\alpha, \\gamma)$", fontsize=15)

ax2.set_xlabel("$\\alpha$")
ax2.set_ylabel("$\\theta$")
ax2.set_yticks([-np.pi/5, -np.pi/10, 0])
ax2.set_yticklabels(["$-\\pi/5$", "$-\\pi/10$", "0"])
ax2.grid()


ax3.set_xlabel("$\\gamma$")
ax3.set_ylabel("$\\theta$")
ax3.set_yticks([-np.pi/5, -np.pi/10, 0])
ax3.set_yticklabels(["$-\\pi/5$", "$-\\pi/10$", "0"])
ax3.grid()

alpha_arr = np.linspace(10, 100, 10)
gamma_arr = np.linspace(1, 10, 10)
pole_x = []
x = np.linspace(-np.pi, 2*np.pi, 100)
key = []
key2 = []

for alpha in alpha_arr:
    tmp_row = []
    key2.append("$\\alpha = {}$".format(alpha))
    for gamma in gamma_arr:
        if alpha == alpha_arr[0]:
            key.append("$\\gamma = {}$".format(gamma))
        if gamma == gamma_arr[0]:
            tmp_row = np.append(tmp_row, NewtonMethod(0, alpha, gamma, 1))
        else:
            tmp_row = np.append(tmp_row, NewtonMethod(tmp_row[-1], alpha, gamma, 1))
    pole_x = np.append(pole_x, tmp_row, axis=0)
    #pole_y = potentialEnergy(alpha, gamma, pole_x, 1)
    #y1 = potentialEnergy(alpha, gamma, x, 1)
    #ax.plot(x, y1, color='b')
pole_x = np.reshape(pole_x, (len(alpha_arr), -1))
ax2.plot(alpha_arr, pole_x)
ax3.plot(gamma_arr, pole_x.T)

ax2.legend(key)
ax3.legend(key2)

#print("alpha_arr is {}".format(alpha_arr[0]))
#print("pole_x is {}".format(pole_x[0]))

print(alpha_arr.shape)
print(pole_x.shape)
plt.show()

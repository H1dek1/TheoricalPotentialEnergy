#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

def potentialEnergy(alpha, gamma, x, direction):
    f_x = direction

    energy = -2 * alpha * f_x * np.cos(x)
    energy += 3 - np.cos(2*x)
    energy -= 2 * gamma * (f_x - (3*np.sqrt(3)*np.sin(x - np.pi/3) + 2*np.cos(x))/alpha)*(3*np.cos(x + np.pi/3) + np.cos(x))

    return energy


fig, ax = plt.subplots(figsize=(10, 10))
plt.subplots_adjust(left=0.25, bottom=0.25)
ax.set_xlabel('$\\theta$', fontsize=18)
ax.set_ylabel('Potential Energy', fontsize=18)
ax.set_xlim(-2*np.pi, 2*np.pi)
ax.set_xticks(np.linspace(-2*np.pi, 2*np.pi, 5))
ax.set_xticklabels(['$-2\pi$', '$-\pi$', '0', '$\pi$', '$2\pi$',]) 
plt.ylim(-1000, 1000)
plt.grid()

x = np.linspace(-2*np.pi, 2*np.pi, 100)
alpha = 10
gamma = 1
d_alpha = 10
d_gamma = 1
y1 = potentialEnergy(alpha, gamma, x, 1)
y2 = potentialEnergy(alpha, gamma, x, -1)

l1, = plt.plot(x, y1, lw=2, label='$B_{ext}=(0, 1, 0)$')
l2, = plt.plot(x, y2, lw=2, label='$B_{ext}=(0, -1, 0)$')
plt.legend(fontsize=15)

axcolor = 'gold'
ax_alpha = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)
ax_gamma = plt.axes([0.25, 0.05, 0.65, 0.03], facecolor=axcolor)

sli_alpha = Slider(ax_alpha, 'Alpha', 10, 100, valinit=alpha, valstep=d_alpha)
sli_gamma = Slider(ax_gamma, 'Gamma', 1, 10, valinit=gamma, valstep=d_gamma)

def update(val):
    s_alpha = sli_alpha.val
    s_gamma = sli_gamma.val
    l1.set_ydata( potentialEnergy(s_alpha, s_gamma, x, 1) )
    l2.set_ydata( potentialEnergy(s_alpha, s_gamma, x, -1) )
    fig.canvas.draw_idle()

sli_alpha.on_changed(update)
sli_gamma.on_changed(update)

resetax = plt.axes([0.8, 0.0, 0.1, 0.04])
button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')

def reset(event):
    sli_alpha.reset()
    sli_gamma.reset()

button.on_clicked(reset)

plt.show()
